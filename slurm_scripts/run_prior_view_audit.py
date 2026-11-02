import os
import shlex
import subprocess

from pathlib import Path
from slurm_scripts.run_prior_array import env_int, make_run
from slurm_scripts.run_prior_audit import audit_execution_command

def main():
    env = dict(os.environ)
    index = env_int(env, 'SLURM_ARRAY_TASK_ID', 0)
    run = make_run('G', index, env)
    root = Path(__file__).resolve().parents[1]
    name = run['name']
    job = {'program': 'audit_prior_views.py', 'argv': ['--root', str(root), '--experiment', name,
           '--data_dir', env['DATA_DIR'], '--device', 'cuda', '--samples', str(env_int(env, 'AUDIT_SAMPLES', 1024, 24)),
           '--output', str(root / 'view_audit' / (name + '.json'))]}
    command = audit_execution_command(job, env)
    print(f'View audit {index}: {name}', flush = True)
    print(shlex.join(command), flush = True)
    if env.get('DRY_RUN') == '1':
        return
    if not env.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit the audit sbatch script or use DRY_RUN=1')
    subprocess.run(command, cwd = root, env = env, check = True)

if __name__ == '__main__':
    main()
