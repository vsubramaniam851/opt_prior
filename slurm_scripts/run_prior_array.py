import fcntl
import json
import math
import os
import shlex
import shutil
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path
from args import get_args
from run_prior import build_command, get_run_args

RUNS = {
    'B': [('resnet', 'baseline', 'rn50', None), ('vit', 'baseline', 'vit_tiny', None),
          ('resnet', 'CKA', 'rn50', None), ('vit', 'CKA', 'vit_tiny', 12),
          ('resnet', 'KNN', 'rn50', None), ('vit', 'KNN', 'vit_tiny', 12)],
    'C': [('resnet', 'CKA', 'rn101', None), ('resnet', 'CKA', 'rn152', None),
          ('vit', 'CKA', 'vit_tiny', 8), ('vit', 'CKA', 'vit_tiny', 24),
          ('resnet', 'KNN', 'rn101', None), ('resnet', 'KNN', 'rn152', None),
          ('vit', 'KNN', 'vit_tiny', 8), ('vit', 'KNN', 'vit_tiny', 24),
          ('vit', 'CKA', 'vit_base', None), ('vit', 'CKA', 'vit_large', None),
          ('vit', 'KNN', 'vit_base', None), ('vit', 'KNN', 'vit_large', None)],
    'D': [('resnet', 'CKA', 'rn50', None), ('resnet', 'KNN', 'rn50', None),
          ('vit', 'CKA', 'vit_tiny', 12), ('vit', 'KNN', 'vit_tiny', 12),
          ('resnet', 'CKA', 'rn152', None), ('resnet', 'KNN', 'rn152', None),
          ('vit', 'CKA', 'vit_tiny', 24), ('vit', 'KNN', 'vit_tiny', 24)]
}
TRAINING_KEYS = ['seed', 'ft_epochs', 'ft_batch', 'ft_accum', 'ft_optimizer', 'ft_lr', 'ft_wd', 'ft_clip',
                 'label_smoothing', 'mixup_alpha', 'scheduler', 'scheduler_type', 'warmup_epochs', 'use_amp']

def parse_with(parser, argv):
    original = sys.argv
    try:
        sys.argv = ['run_prior_array'] + argv
        return parser()
    finally:
        sys.argv = original

def env_int(env, name, default, minimum = 0):
    value = str(env.get(name, default))
    if not value.isascii() or not value.isdigit() or int(value) < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)

def make_run(group, index, env):
    if group not in RUNS or index < 0 or index >= len(RUNS[group]):
        raise ValueError('Use group B with indices 0-5 or group C with indices 0-11 or group D with indices 0-7')
    if not env.get('DATA_DIR'):
        raise ValueError('Set DATA_DIR to the ImageNet directory containing train and val')
    family, method, guide, depth = RUNS[group][index]
    default_epochs = 100 if family == 'resnet' else 300
    epochs = env_int(env, 'RESNET_EPOCHS' if family == 'resnet' else 'VIT_EPOCHS', default_epochs, 6 if family == 'resnet' else 21)
    phase = 'train' if group == 'B' and method != 'baseline' else 'full'
    if group == 'D':
        phase = env.get('PRIOR_PHASE', 'align')
        if phase not in ['align', 'train', 'full']:
            raise ValueError('PRIOR_PHASE must be align, train or full')
    argv = [family, method, '--guide', guide, '--data_dir', str(Path(env['DATA_DIR']).expanduser().resolve()),
            '--seed', str(env_int(env, 'SEED', 0)), '--teacher_seed', str(env_int(env, 'TEACHER_SEED', 1000)),
            '--workers', str(env_int(env, 'WORKERS', 8)), '--phase', phase, '--epochs', str(epochs)]
    if group == 'D':
        argv += ['--prior_descriptor', 'spatial']
    if depth is not None:
        argv += ['--depth', str(depth)]
    amp = env_int(env, 'AMP', 0)
    if amp not in [0, 1]:
        raise ValueError('Use AMP=0 or AMP=1')
    if amp:
        argv += ['--amp']
    args = parse_with(get_run_args, argv)
    initial_command = build_command(args)
    source = initial_command[initial_command.index('--exp_name') + 1]
    name = source if epochs == default_epochs else f'{source}_e{epochs}'
    argv += ['--exp_name', name]
    args.exp_name = name
    training = parse_with(get_args, build_command(args)[2:])
    return {'group': group, 'index': index, 'family': family, 'method': method, 'phase': phase, 'source': source,
            'name': name, 'epochs': epochs, 'argv': argv, 'training': vars(training)}

def read_json(path):
    with path.open() as f:
        return json.load(f)

def validate_initialization(root, name, run):
    folder = root / 'logs' / name
    checkpoint = root / 'saved_models' / f'orig_{name}.pt'
    if not checkpoint.is_file() or not (folder / 'prior_alignment.json').is_file() or not (folder / 'args.json').is_file():
        raise FileNotFoundError(f'Missing saved initialization for {name}. Complete its v2 alignment first; no alignment is started by B training tasks')
    metadata = read_json(folder / 'prior_alignment.json')
    if metadata.get('version') != 2 or not metadata.get('complete') or not metadata.get('guide_unchanged'):
        raise ValueError(f'{name} is not a completed, unchanged-guide v2 initialization')
    config = metadata.get('alignment_config', {})
    if not config or not metadata.get('student_aligned_hash'):
        raise ValueError(f'Missing alignment configuration or state fingerprint for {name}')
    mismatched = [key for key, value in config.items() if run['training'].get(key) != value]
    if mismatched:
        raise ValueError(f'Alignment configuration differs for {name}: {mismatched}. Use the same v2 settings, data path, WORKERS and AMP')
    return metadata

def preflight(root, run):
    data = Path(run['training']['data_dir'])
    if not all((data / split).is_dir() for split in ['train', 'val']):
        raise FileNotFoundError(f'Expected ImageNet class directories under {data}/train and {data}/val')
    name = run['name']
    folder = root / 'logs' / name
    if run['phase'] == 'align' and (folder / 'prior_alignment.json').is_file():
        validate_initialization(root, name, run)
        return 'complete_alignment'
    final = root / 'saved_models' / f'ft_final_{name}.pt'
    result = folder / ('ft_results_seed' + str(run['training']['seed']) + '.json')
    if final.is_file() and result.is_file():
        validate_initialization(root, name, run)
        path = folder / 'training_args.json' if (folder / 'training_args.json').is_file() else folder / 'args.json'
        previous = read_json(path)
        mismatched = [key for key in TRAINING_KEYS if previous.get(key) != run['training'].get(key)]
        accuracy = read_json(result).get('final_top1')
        if mismatched or not isinstance(accuracy, (float, int)) or not math.isfinite(accuracy):
            raise ValueError(f'Existing completion has different settings or invalid results: {name}, {mismatched}')
        return 'complete'
    saved_training = [folder / 'training_args.json', result, final, root / 'saved_models' / f'ft_{name}.pt']
    if any(path.exists() for path in saved_training):
        raise FileExistsError(f'Training already started or is incomplete for {name}. There is no optimizer-state resume; existing files will not be overwritten')
    if run['phase'] == 'train':
        source = run['source']
        destination_exists = any((folder / filename).exists() for filename in ['args.json', 'prior_alignment.json'])
        destination_exists = destination_exists or (root / 'saved_models' / f'orig_{name}.pt').exists()
        selected = name if destination_exists else source
        validate_initialization(root, selected, run)
        original = read_json(root / 'logs' / selected / 'args.json')
        if selected == name and not original.get('align_only', False):
            raise FileExistsError(f'{name} was launched with supervised training enabled. Do not start another job against its outputs')
        return 'copy' if selected != name else 'train'
    if any(path.exists() for path in [folder / 'args.json', folder / 'prior_alignment.json', root / 'saved_models' / f'orig_{name}.pt']):
        raise FileExistsError(f'{name} already exists. This full-run task will not overwrite or resume it')
    return 'full'

def copy_initialization(root, run):
    source, name = run['source'], run['name']
    folder = root / 'logs' / name
    folder.mkdir(parents = True, exist_ok = True)
    (root / 'saved_models').mkdir(exist_ok = True)
    for filename in ['args.json', 'prior_alignment.json']:
        shutil.copy2(root / 'logs' / source / filename, folder / filename)
    shutil.copy2(root / 'saved_models' / f'orig_{source}.pt', root / 'saved_models' / f'orig_{name}.pt')
    with (folder / 'initialization_source.json').open('w') as f:
        json.dump({'source_experiment': source, 'destination_experiment': name, 'epochs': run['epochs']}, f, indent = 2)

def conda_hook(env):
    return Path(env.get('CONDA_BASE', '/data/aia/scratch/vsub851/anaconda3')).expanduser() / 'etc' / 'profile.d' / 'conda.sh'

def execution_command(run, env):
    current = env_int(env, 'USE_CURRENT_ENV', 0)
    if current not in [0, 1]:
        raise ValueError('Use USE_CURRENT_ENV=0 or USE_CURRENT_ENV=1')
    program = [run.get('program', 'run_prior.py')] + run['argv']
    if current:
        return ['srun', '--unbuffered', sys.executable, '-u'] + program
    shell = '\n'.join(['source ' + shlex.quote(str(conda_hook(env))),
                       'conda activate ' + shlex.quote(env.get('CONDA_ENV', 'nas')),
                       'exec ' + shlex.join(['python', '-u'] + program)])
    return ['srun', '--unbuffered', 'bash', '-e', '-c', shell]

def execute(root, run, env):
    status = preflight(root, run)
    name = run['name']
    command = execution_command(run, env)
    group, index, epochs = run['group'], run['index'], run['epochs']
    print(f'Array {group}[{index}]: {name}, epochs={epochs}, action={status}', flush = True)
    print(shlex.join(command), flush = True)
    if status in ['complete', 'complete_alignment']:
        print('Already complete; skipping without changing results', flush = True)
        return
    if env.get('DRY_RUN', '0') == '1':
        print('Dry run: inputs checked; no files copied, no GPU process started', flush = True)
        return
    if not env.get('SLURM_JOB_ID') or not shutil.which('srun'):
        raise RuntimeError('Submit the sbatch script, or set DRY_RUN=1 for a local preflight')
    if env.get('USE_CURRENT_ENV', '0') != '1' and not conda_hook(env).is_file():
        raise FileNotFoundError('Conda shell hook not found. Set CONDA_BASE or activate the environment and export USE_CURRENT_ENV=1')
    locks = root / 'logs' / 'array_locks'
    locks.mkdir(parents = True, exist_ok = True)
    with (locks / f'{name}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another batch launcher holds the lock for {name}')
        status = preflight(root, run)
        if status in ['complete', 'complete_alignment']:
            return
        if status == 'copy':
            copy_initialization(root, run)
        child_env = dict(env)
        child_env['PYTHONUNBUFFERED'] = '1'
        child_env['SLURM_EXPORT_ENV'] = 'ALL'
        child_env.setdefault('WANDB_MODE', 'online' if child_env.get('WANDB_API_KEY') else 'offline')
        folder = root / 'logs' / name
        folder.mkdir(parents = True, exist_ok = True)
        launch = {key: value for key, value in run.items() if key != 'training'}
        launch.update({'command': command, 'job_id': env['SLURM_JOB_ID'], 'wandb_mode': child_env['WANDB_MODE'],
                       'started_utc': datetime.now(timezone.utc).isoformat()})
        job_id = env['SLURM_JOB_ID']
        with (folder / f'slurm_launch_{job_id}.json').open('w') as f:
            json.dump(launch, f, indent = 2)
        print('WANDB_MODE=' + child_env['WANDB_MODE'], flush = True)
        subprocess.run(command, cwd = root, env = child_env, check = True)
        expected = 'complete_alignment' if run['phase'] == 'align' else 'complete'
        if preflight(root, run) != expected:
            raise RuntimeError('The process exited without a complete final checkpoint and result JSON')
        print(f'Completed: {name}', flush = True)

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in RUNS:
        raise ValueError('Run from network_theseus: python3 -m slurm_scripts.run_prior_array B, C or D')
    env = dict(os.environ)
    index = env_int(env, 'SLURM_ARRAY_TASK_ID', -1)
    run = make_run(sys.argv[1], index, env)
    execute(Path(__file__).resolve().parents[1], run, env)

if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file = sys.stderr)
        sys.exit(1)
