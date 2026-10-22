import os
import subprocess

from pathlib import Path
from slurm_scripts.run_prior_array import env_int, execution_command, make_run

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
    else:
        raise ValueError('AUDIT_GROUP must be original or spatial')
    root = Path(__file__).resolve().parents[1]
    name = run['name']
    output = root / 'geometry_audit' / (name + '.json')
    job = {'program': 'audit_prior_geometry.py', 'argv': ['--root', str(root), '--experiment', name,
           '--data_dir', env['DATA_DIR'], '--device', 'cuda', '--samples', str(env_int(env, 'AUDIT_SAMPLES', 1024, 24)),
           '--output', str(output), '--control_guide']}
    print(f'Audit {index}: {name}', flush = True)
    if env.get('DRY_RUN') == '1':
        print(execution_command(job, env), flush = True)
        return
    if not env.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit the audit sbatch script or use DRY_RUN=1')
    subprocess.run(execution_command(job, env), cwd = root, env = env, check = True)

if __name__ == '__main__':
    main()
