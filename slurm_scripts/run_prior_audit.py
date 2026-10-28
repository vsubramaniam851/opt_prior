import os
import shlex
import subprocess

from pathlib import Path
from slurm_scripts.run_prior_array import env_int, execution_command, make_run

def audit_execution_command(job, env):
    command = execution_command(job, env)
    if command[:5] == ['srun', '--unbuffered', 'bash', '-e', '-c']:
        target = shlex.split(command[-1].splitlines()[-1])
    else:
        target = command
    expected = ['-u', job['program']] + job['argv']
    if target[-len(expected):] != expected:
        raise RuntimeError('Audit routing mismatch: update slurm_scripts/run_prior_array.py and slurm_scripts/run_prior_audit.py together. The shared launcher must honor the requested audit program')
    return command

def main():
    env = dict(os.environ)
    index = env_int(env, 'SLURM_ARRAY_TASK_ID', 0)
    group = env.get('AUDIT_GROUP', 'original')
    if group == 'original':
        mapping = [('B', 2), ('B', 4), ('B', 3), ('B', 5), ('C', 1), ('C', 5), ('C', 3), ('C', 7)]
        if index >= len(mapping):
            raise ValueError('Use audit array indices 0-7')
        run = make_run(*mapping[index], env)
    elif group == 'spatial':
        run = make_run('D', index, env)
    elif group == 'upper':
        run = make_run('E', index, env)
    elif group == 'bn_dose':
        run = make_run('F', index, env)
    else:
        raise ValueError('AUDIT_GROUP must be original, spatial, upper or bn_dose')
    root = Path(__file__).resolve().parents[1]
    name = run['name']
    output = root / 'geometry_audit' / (name + '.json')
    job = {'program': 'audit_prior_geometry.py', 'argv': ['--root', str(root), '--experiment', name,
           '--data_dir', env['DATA_DIR'], '--device', 'cuda', '--samples', str(env_int(env, 'AUDIT_SAMPLES', 1024, 24)),
           '--output', str(output), '--control_guide']}
    command = audit_execution_command(job, env)
    print(f'Audit {index}: {name}', flush = True)
    print(shlex.join(command), flush = True)
    if env.get('DRY_RUN') == '1':
        return
    if not env.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit the audit sbatch script or use DRY_RUN=1')
    subprocess.run(command, cwd = root, env = env, check = True)

if __name__ == '__main__':
    main()
