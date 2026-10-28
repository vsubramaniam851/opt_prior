import os
import shlex
import subprocess

from pathlib import Path
from slurm_scripts.run_prior_array import env_int, make_run
from slurm_scripts.run_prior_audit import audit_execution_command

def main():
    env = dict(os.environ)
    index = env_int(env, 'SLURM_ARRAY_TASK_ID', 0)
    if index not in [0, 1]:
        raise ValueError('Use mechanism array indices 0 and 1')
    source_env = dict(env, ALIGN_STEPS = '2000')
    run = make_run('E', index + 2, source_env)
    root = Path(__file__).resolve().parents[1]
    name = run['name']
    job = {'program': 'audit_prior_mechanism.py', 'argv': ['--root', str(root), '--experiment', name,
           '--data_dir', env['DATA_DIR'], '--device', 'cuda', '--samples', str(env_int(env, 'AUDIT_SAMPLES', 256, 24))]}
    command = audit_execution_command(job, env)
    print(f'Mechanism audit {index}: {name}', flush = True)
    print(shlex.join(command), flush = True)
    if env.get('DRY_RUN') == '1':
        return
    if not env.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit the sbatch script or use DRY_RUN=1')
    subprocess.run(command, cwd = root, env = env, check = True)

if __name__ == '__main__':
    main()
