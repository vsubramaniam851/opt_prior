import ast
import contextlib
import fcntl
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch
from slurm_scripts.run_prior_array import RUNS, copy_initialization, env_int, execute, execution_command
from slurm_scripts.run_prior_array import make_run, preflight, read_json, validate_initialization

ROOT = Path(__file__).resolve().parents[1]
FUNCTION = next(node for node in ast.parse((ROOT / 'prior_utils.py').read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'prior_config')
CONFIG_KEYS = ast.literal_eval(FUNCTION.body[0].value)

def write_json(path, data):
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open('w') as f:
        json.dump(data, f)

def write_initialization(root, run, source = False, align_only = True):
    name = run['source'] if source else run['name']
    folder = root / 'logs' / name
    config = {key: run['training'].get(key) for key in CONFIG_KEYS}
    write_json(folder / 'prior_alignment.json', {'version': 2, 'complete': True, 'guide_unchanged': True,
                                               'student_aligned_hash': 'synthetic-fingerprint', 'alignment_config': config})
    write_json(folder / 'args.json', dict(run['training'], exp_name = name, align_only = align_only))
    checkpoint = root / 'saved_models' / f'orig_{name}.pt'
    checkpoint.parent.mkdir(exist_ok = True)
    checkpoint.write_bytes(b'synthetic-initialization')

def write_completion(root, run):
    name = run['name']
    if not (root / 'logs' / name / 'prior_alignment.json').exists():
        write_initialization(root, run, align_only = False)
    folder = root / 'logs' / name
    args_path = folder / ('training_args.json' if run['phase'] == 'train' else 'args.json')
    write_json(args_path, run['training'])
    write_json(folder / ('ft_results_seed' + str(run['training']['seed']) + '.json'), {'final_top1': 61.25})
    (root / 'saved_models' / f'ft_final_{name}.pt').write_bytes(b'synthetic-final')

class ArrayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix = 'prior array ')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        data = self.root / 'image net'
        for split in ['train', 'val']:
            (data / split).mkdir(parents = True)
        self.env = {'DATA_DIR': str(data), 'USE_CURRENT_ENV': '1', 'SLURM_JOB_ID': '12345'}
    def test_b_mapping_and_phases(self):
        rows = []
        for index in range(6):
            run = make_run('B', index, self.env)
            rows.append((run['family'], run['method'], run['training']['model'], run['training']['teacher_depth'], run['phase']))
        self.assertEqual(rows, [('resnet', 'baseline', 'rn50', None, 'full'), ('vit', 'baseline', 'vit_tiny', 4, 'full'),
                               ('resnet', 'CKA', 'rn50', None, 'train'), ('vit', 'CKA', 'vit_tiny', 12, 'train'),
                               ('resnet', 'KNN', 'rn50', None, 'train'), ('vit', 'KNN', 'vit_tiny', 12, 'train')])
    def test_c_mapping_and_unique_names(self):
        runs = [make_run(group, i, self.env) for group in ['B', 'C'] for i in range(len(RUNS[group]))]
        self.assertEqual(len(set(run['name'] for run in runs)), 18)
        for run in runs[6:]:
            self.assertEqual(run['phase'], 'full')
            self.assertIn(run['training']['model'], ['rn101', 'rn152', 'vit_tiny', 'vit_base', 'vit_large'])
            if run['training']['model'] == 'vit_tiny':
                self.assertIn(run['training']['teacher_depth'], [8, 24])
    def test_unchanged_default_recipe(self):
        for group in ['B', 'C']:
            for index in range(len(RUNS[group])):
                run = make_run(group, index, self.env)
                args = run['training']
                resnet = run['family'] == 'resnet'
                self.assertEqual(run['name'], run['source'])
                self.assertEqual(run['epochs'], 100 if resnet else 300)
                self.assertEqual(args['ft_batch'], 256 if resnet else 128)
                self.assertEqual(args['ft_accum'], 1 if resnet else 4)
                self.assertEqual(args['ft_optimizer'], 'sgd' if resnet else 'adamw')
                self.assertEqual(args['ft_lr'], 0.1 if resnet else 5e-4)
                self.assertEqual(args['max_stage_steps'], 1000)
                self.assertEqual(args['rep_batch'], 128)
                self.assertEqual(args['rep_val_size'], 4096)
                self.assertEqual(args['prior_schedule'], 'cumulative')
                self.assertEqual(args['align_mix'], 1.0)
                self.assertEqual(args['prior_min_rank_ratio'], 0.0)
                self.assertEqual(args['cka_loss'], 'debiased')
                self.assertEqual(args['knn_loss'], 'soft')
                self.assertFalse(args['pretrained'])
                self.assertFalse(args['use_amp'])
    def test_short_budget_names_and_pairs(self):
        for group in ['B', 'C']:
            for index in range(len(RUNS[group])):
                run = make_run(group, index, dict(self.env, VIT_EPOCHS = '100'))
                self.assertEqual(run['epochs'], 100)
                self.assertEqual(run['name'], run['source'] + ('_e100' if run['family'] == 'vit' else ''))
                if run['family'] == 'vit':
                    self.assertEqual(run['training']['warmup_epochs'], 20)
    def test_seed_amp_and_workers(self):
        run = make_run('B', 3, dict(self.env, SEED = '2', TEACHER_SEED = '1001', AMP = '1', WORKERS = '6'))
        self.assertIn('_s2_g1001_bf16', run['name'])
        self.assertTrue(run['training']['use_amp'])
        self.assertEqual(run['training']['num_workers'], 6)
    def test_valid_preflight(self):
        for group in ['B', 'C']:
            for index in range(len(RUNS[group])):
                run = make_run(group, index, self.env)
                if run['phase'] == 'train':
                    write_initialization(self.root, run)
                self.assertEqual(preflight(self.root, run), 'train' if run['phase'] == 'train' else 'full')
    def test_missing_alignment_fails(self):
        with self.assertRaisesRegex(FileNotFoundError, 'alignment first'):
            preflight(self.root, make_run('B', 2, self.env))
    def test_configuration_mismatch_fails(self):
        run = make_run('B', 2, self.env)
        write_initialization(self.root, run)
        path = self.root / 'logs' / run['name'] / 'prior_alignment.json'
        for key, value in [('num_workers', 16), ('use_amp', True), ('data_dir', '/different/path'), ('prior_schedule', 'frozen')]:
            original = read_json(path)
            changed = json.loads(json.dumps(original))
            changed['alignment_config'][key] = value
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, key):
                preflight(self.root, run)
            write_json(path, original)
    def test_incomplete_or_changed_guide_fails(self):
        run = make_run('B', 2, self.env)
        write_initialization(self.root, run)
        path = self.root / 'logs' / run['name'] / 'prior_alignment.json'
        for key, value in [('complete', False), ('guide_unchanged', False), ('version', 1)]:
            original = read_json(path)
            write_json(path, dict(original, **{key: value}))
            with self.assertRaises(ValueError):
                preflight(self.root, run)
            write_json(path, original)
    def test_existing_training_fails(self):
        run = make_run('B', 2, self.env)
        write_initialization(self.root, run)
        path = self.root / 'logs' / run['name'] / 'training_args.json'
        write_json(path, run['training'])
        with self.assertRaisesRegex(FileExistsError, 'no optimizer-state resume'):
            preflight(self.root, run)
    def test_full_run_never_overwrites(self):
        for group, index in [('B', 0), ('C', 0)]:
            run = make_run(group, index, self.env)
            write_initialization(self.root, run)
            with self.assertRaisesRegex(FileExistsError, 'will not overwrite'):
                preflight(self.root, run)
    def test_completed_run_skipped_and_budget_checked(self):
        run = make_run('B', 2, self.env)
        write_completion(self.root, run)
        self.assertEqual(preflight(self.root, run), 'complete')
        path = self.root / 'logs' / run['name'] / 'training_args.json'
        write_json(path, dict(run['training'], ft_epochs = 50))
        with self.assertRaisesRegex(ValueError, 'different settings'):
            preflight(self.root, run)
    def test_partial_completion_never_skipped(self):
        run = make_run('B', 2, self.env)
        write_initialization(self.root, run)
        (self.root / 'saved_models' / ('ft_final_' + run['name'] + '.pt')).write_bytes(b'partial')
        with self.assertRaises(FileExistsError):
            preflight(self.root, run)
    def test_short_copy_preserves_source(self):
        run = make_run('B', 3, dict(self.env, VIT_EPOCHS = '100'))
        write_initialization(self.root, run, source = True)
        source = self.root / 'saved_models' / ('orig_' + run['source'] + '.pt')
        fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(preflight(self.root, run), 'copy')
        copy_initialization(self.root, run)
        destination = self.root / 'saved_models' / ('orig_' + run['name'] + '.pt')
        self.assertEqual(hashlib.sha256(destination.read_bytes()).hexdigest(), fingerprint)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), fingerprint)
        self.assertEqual(preflight(self.root, run), 'train')
        self.assertEqual(read_json(self.root / 'logs' / run['name'] / 'initialization_source.json')['source_experiment'], run['source'])
    def test_short_copy_can_use_completed_source(self):
        original = make_run('B', 3, self.env)
        write_completion(self.root, original)
        run = make_run('B', 3, dict(self.env, VIT_EPOCHS = '100'))
        self.assertEqual(preflight(self.root, run), 'copy')
    def test_dry_run_has_no_side_effects(self):
        run = make_run('B', 3, dict(self.env, VIT_EPOCHS = '100'))
        write_initialization(self.root, run, source = True)
        before = sorted(str(path) for path in self.root.rglob('*'))
        with contextlib.redirect_stdout(io.StringIO()), patch('subprocess.run') as process:
            execute(self.root, run, dict(self.env, DRY_RUN = '1'))
        process.assert_not_called()
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob('*')))
    def test_execution_and_completion_all_eighteen(self):
        for group in ['B', 'C']:
            for index in range(len(RUNS[group])):
                run = make_run(group, index, self.env)
                if run['phase'] == 'train':
                    write_initialization(self.root, run)
                def finish(command, cwd, env, check):
                    self.assertEqual(command, execution_command(run, self.env))
                    self.assertEqual(cwd, self.root)
                    self.assertTrue(check)
                    self.assertEqual(env['SLURM_EXPORT_ENV'], 'ALL')
                    self.assertEqual(env['PYTHONUNBUFFERED'], '1')
                    self.assertEqual(env['WANDB_MODE'], 'offline')
                    write_completion(self.root, run)
                with patch('shutil.which', return_value = '/mock/srun'), patch('subprocess.run', side_effect = finish) as process:
                    with contextlib.redirect_stdout(io.StringIO()):
                        execute(self.root, run, self.env)
                    process.assert_called_once()
                self.assertEqual(preflight(self.root, run), 'complete')
                with patch('subprocess.run') as process, contextlib.redirect_stdout(io.StringIO()):
                    execute(self.root, run, self.env)
                process.assert_not_called()
    def test_short_execution_reuses_initialization(self):
        run = make_run('B', 3, dict(self.env, VIT_EPOCHS = '100'))
        write_initialization(self.root, run, source = True)
        def finish(command, cwd, env, check):
            source = self.root / 'saved_models' / ('orig_' + run['source'] + '.pt')
            destination = self.root / 'saved_models' / ('orig_' + run['name'] + '.pt')
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(command[command.index('--phase') + 1], 'train')
            self.assertEqual(command[command.index('--epochs') + 1], '100')
            write_completion(self.root, run)
        with patch('shutil.which', return_value = '/mock/srun'), patch('subprocess.run', side_effect = finish):
            with contextlib.redirect_stdout(io.StringIO()):
                execute(self.root, run, self.env)
        self.assertEqual(preflight(self.root, run), 'complete')
        self.assertTrue((self.root / 'logs' / run['name'] / 'initialization_source.json').exists())
    def test_lock_blocks_duplicate(self):
        run = make_run('B', 0, self.env)
        path = self.root / 'logs' / 'array_locks' / (run['name'] + '.lock')
        path.parent.mkdir(parents = True)
        with path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch('shutil.which', return_value = '/mock/srun'), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'holds the lock'):
                    execute(self.root, run, self.env)
    def test_child_failure_propagates(self):
        run = make_run('B', 0, self.env)
        with patch('shutil.which', return_value = '/mock/srun'), patch('subprocess.run', side_effect = subprocess.CalledProcessError(7, ['srun'])):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                execute(self.root, run, self.env)
        self.assertFalse((self.root / 'saved_models' / ('ft_final_' + run['name'] + '.pt')).exists())
    def test_conda_command(self):
        run = make_run('B', 0, self.env)
        command = execution_command(run, dict(self.env, USE_CURRENT_ENV = '0'))
        self.assertEqual(command[:5], ['srun', '--unbuffered', 'bash', '-e', '-c'])
        lines = command[5].splitlines()
        self.assertEqual(shlex.split(lines[0]), ['source', '/data/aia/scratch/vsub851/anaconda3/etc/profile.d/conda.sh'])
        self.assertEqual(shlex.split(lines[1]), ['conda', 'activate', 'nas'])
        self.assertEqual(shlex.split(lines[2]), ['exec', 'python', '-u', 'run_prior.py'] + run['argv'])
        self.assertNotIn('--no-capture-output', command[5])
    def test_current_environment_bypasses_conda(self):
        run = make_run('B', 5, self.env)
        self.assertEqual(execution_command(run, self.env), ['srun', '--unbuffered', sys.executable, '-u', 'run_prior.py'] + run['argv'])
    def test_default_activation_and_quoting(self):
        base = self.root / 'fake conda'
        hook = base / 'etc' / 'profile.d' / 'conda.sh'
        hook.parent.mkdir(parents = True)
        bin_dir = self.root / 'activated env' / 'bin'
        bin_dir.mkdir(parents = True)
        (bin_dir / 'python').symlink_to(sys.executable)
        hook.write_text('conda() {\n    export PATH=' + shlex.quote(str(bin_dir)) + ':${PATH}\n    export PRIOR_TEST_ACTIVATED=1\n}\n')
        program = '\n'.join(['import json', 'import os', 'import sys', 'from pathlib import Path',
                             'Path(\'called.json\').write_text(json.dumps({\'argv\': sys.argv[1:], \'python\': sys.executable, \'activated\': os.environ.get(\'PRIOR_TEST_ACTIVATED\')}))'])
        (self.root / 'run_prior.py').write_text(program)
        env = dict(os.environ, **self.env, CONDA_BASE = str(base), CONDA_ENV = 'nas; exit 79')
        env['USE_CURRENT_ENV'] = '0'
        env.pop('PRIOR_TEST_ACTIVATED', None)
        for group in ['B', 'C']:
            for index in range(len(RUNS[group])):
                run = make_run(group, index, env)
                run['argv'][-1] += '; touch should_not_exist'
                command = execution_command(run, env)
                self.assertEqual(shlex.split(command[5].splitlines()[1]), ['conda', 'activate', env['CONDA_ENV']])
                result = subprocess.run(command[2:], cwd = self.root, env = env, capture_output = True, text = True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                called = read_json(self.root / 'called.json')
                self.assertEqual(called['argv'], run['argv'])
                self.assertEqual(called['python'], str(bin_dir / 'python'))
                self.assertEqual(called['activated'], '1')
                self.assertFalse((self.root / 'should_not_exist').exists())
    def test_activation_failure_does_not_start_python(self):
        base = self.root / 'failed conda'
        hook = base / 'etc' / 'profile.d' / 'conda.sh'
        hook.parent.mkdir(parents = True)
        hook.write_text('conda() { return 23; }\n')
        (self.root / 'run_prior.py').write_text('from pathlib import Path\nPath(\'called\').touch()\n')
        env = dict(self.env, USE_CURRENT_ENV = '0', CONDA_BASE = str(base))
        command = execution_command(make_run('B', 0, env), env)
        result = subprocess.run(command[2:], cwd = self.root, capture_output = True, text = True)
        self.assertEqual(result.returncode, 23)
        self.assertFalse((self.root / 'called').exists())
    def test_missing_hook_fails_before_writes(self):
        env = dict(self.env, USE_CURRENT_ENV = '0', CONDA_BASE = str(self.root / 'missing conda'))
        run = make_run('B', 0, env)
        with patch('shutil.which', return_value = '/mock/srun'), patch('subprocess.run') as process:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(FileNotFoundError, 'shell hook'):
                execute(self.root, run, env)
        process.assert_not_called()
        self.assertFalse((self.root / 'logs').exists())
        self.assertFalse((self.root / 'saved_models').exists())
    def test_legacy_failed_launch_log_allows_training_retry(self):
        run = make_run('B', 5, self.env)
        write_initialization(self.root, run)
        checkpoint = self.root / 'saved_models' / ('orig_' + run['name'] + '.pt')
        original = checkpoint.read_bytes()
        folder = self.root / 'logs' / run['name']
        write_json(folder / 'slurm_launch_12344.json', {'command': ['srun', 'conda', 'run', '--no-capture-output']})
        self.assertEqual(preflight(self.root, run), 'train')
        with patch('shutil.which', return_value = '/mock/srun'), patch('subprocess.run', side_effect = subprocess.CalledProcessError(2, ['srun'])):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                execute(self.root, run, self.env)
        self.assertEqual(checkpoint.read_bytes(), original)
        self.assertEqual(preflight(self.root, run), 'train')
        with patch('shutil.which', return_value = '/mock/srun'), patch('subprocess.run', side_effect = lambda *args, **kwargs: write_completion(self.root, run)):
            with contextlib.redirect_stdout(io.StringIO()):
                execute(self.root, run, self.env)
        self.assertEqual(preflight(self.root, run), 'complete')
        self.assertEqual(checkpoint.read_bytes(), original)
    def test_activation_failure_allows_default_path_retry(self):
        base = self.root / 'retry conda'
        hook = base / 'etc' / 'profile.d' / 'conda.sh'
        hook.parent.mkdir(parents = True)
        hook.write_text('conda() { return 23; }\n')
        env = dict(self.env, USE_CURRENT_ENV = '0', CONDA_BASE = str(base))
        run = make_run('B', 5, env)
        write_initialization(self.root, run)
        process_run = subprocess.run
        def launch(command, cwd, env, check):
            return process_run(command[2:], cwd = cwd, env = env, check = check)
        with patch('shutil.which', return_value = '/mock/srun'), patch('subprocess.run', side_effect = launch):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                execute(self.root, run, env)
        self.assertEqual(preflight(self.root, run), 'train')
        self.assertFalse((self.root / 'logs' / run['name'] / 'training_args.json').exists())
    def test_invalid_inputs(self):
        for env in [dict(self.env, SEED = '-1'), dict(self.env, AMP = '2'), dict(self.env, VIT_EPOCHS = '20'), dict(self.env, DATA_DIR = '')]:
            with self.assertRaises(ValueError):
                make_run('B', 3, env)
        for group, index in [('B', 6), ('C', 12), ('A', 0), ('B', -1)]:
            with self.assertRaises(ValueError):
                make_run(group, index, self.env)
        with self.assertRaises(ValueError):
            env_int({}, 'SLURM_ARRAY_TASK_ID', -1)
    def test_sbatch_shell_and_all_dry_runs(self):
        for name in ['run_prior.py', 'args.py']:
            shutil.copy2(ROOT / name, self.root / name)
        shutil.copytree(ROOT / 'slurm_scripts', self.root / 'slurm_scripts', ignore = shutil.ignore_patterns('__pycache__'))
        for group, script in [('B', 'prior_b_train.sbatch'), ('C', 'prior_c_depth.sbatch')]:
            script_path = self.root / 'slurm_scripts' / script
            subprocess.run(['bash', '-n', str(script_path)], check = True)
            text = script_path.read_text()
            self.assertIn('#SBATCH --time=3-00:00:00', text)
            self.assertIn('#SBATCH --array=0-' + str(5 if group == 'B' else 7) + '%4', text)
            self.assertIn('#SBATCH --no-requeue', text)
            for index in range(len(RUNS[group])):
                run = make_run(group, index, self.env)
                if run['phase'] == 'train':
                    write_initialization(self.root, run)
                env = dict(os.environ, **self.env, DRY_RUN = '1', SLURM_ARRAY_TASK_ID = str(index))
                result = subprocess.run(['bash', str(script_path)], cwd = self.root, env = env, capture_output = True, text = True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(run['name'], result.stdout)
                self.assertIn('Dry run:', result.stdout)

if __name__ == '__main__':
    unittest.main(verbosity = 2)
