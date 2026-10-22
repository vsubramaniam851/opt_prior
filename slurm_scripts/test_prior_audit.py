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
from slurm_scripts.run_prior_array import RUNS, execution_command, make_run
from slurm_scripts.run_prior_audit import audit_execution_command

ROOT = Path(__file__).resolve().parents[1]

class AuditLauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix = 'prior audit \' ')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        scripts = self.root / 'slurm_scripts'
        scripts.mkdir()
        for name in ['args.py', 'run_prior.py']:
            shutil.copy2(ROOT / name, self.root / name)
        for name in ['run_prior_array.py', 'run_prior_audit.py', 'prior_geometry_audit.sbatch']:
            shutil.copy2(ROOT / 'slurm_scripts' / name, scripts / name)
        probe = '\n'.join(['import json', 'import os', 'import sys', 'from pathlib import Path',
                           'Path(\'audit_called.json\').write_text(json.dumps({\'argv\': sys.argv[1:], \'program\': Path(__file__).name, \'activated\': os.environ.get(\'PRIOR_TEST_ACTIVATED\')}))'])
        (self.root / 'audit_prior_geometry.py').write_text(probe + '\n')
        self.bin = self.root / 'mock bin'
        self.bin.mkdir()
        (self.bin / 'srun').write_text('#!' + sys.executable + '\nimport os\nimport sys\nos.execvpe(sys.argv[2], sys.argv[2:], os.environ)\n')
        (self.bin / 'srun').chmod(0o755)
        (self.bin / 'python3').symlink_to(sys.executable)
        base = self.root / 'mock conda'
        hook = base / 'etc' / 'profile.d' / 'conda.sh'
        hook.parent.mkdir(parents = True)
        activated = self.root / 'mock environment' / 'bin'
        activated.mkdir(parents = True)
        (activated / 'python').symlink_to(sys.executable)
        hook.write_text('conda() {\n    export PATH=' + shlex.quote(str(activated)) + ':${PATH}\n    export PRIOR_TEST_ACTIVATED=1\n}\n')
        self.hook = hook
        self.env = dict(os.environ)
        for key in ['DRY_RUN', 'PRIOR_TEST_ACTIVATED', 'PYTHONPATH', 'PYTHONHOME']:
            self.env.pop(key, None)
        self.env.update({'DATA_DIR': str(self.root / 'image net'), 'SEED': '0', 'TEACHER_SEED': '1000', 'WORKERS': '8',
                         'AMP': '0', 'RESNET_EPOCHS': '100', 'VIT_EPOCHS': '300', 'PRIOR_PHASE': 'align',
                         'AUDIT_GROUP': 'original', 'AUDIT_SAMPLES': '1024', 'USE_CURRENT_ENV': '0',
                         'SLURM_ARRAY_TASK_ID': '1', 'SLURM_JOB_ID': 'mock-audit-1', 'CONDA_BASE': str(base),
                         'CONDA_ENV': 'nas; touch unexpected', 'PATH': str(self.bin) + os.pathsep + os.environ['PATH']})
    def job(self):
        return {'program': 'audit_prior_geometry.py', 'argv': ['--root', str(self.root), '--experiment', 'prior_v2_rn50_rn18_knn_s0_g1000',
                '--data_dir', self.env['DATA_DIR'], '--device', 'cuda', '--samples', '1024', '--control_guide']}
    def launch(self, env = None):
        return subprocess.run(['bash', 'slurm_scripts/prior_geometry_audit.sbatch'], cwd = self.root,
                              env = self.env if env is None else env, capture_output = True, text = True, timeout = 15)
    def test_audit_chooses_requested_program_in_both_environment_modes(self):
        for current in ['0', '1']:
            with self.subTest(current = current):
                command = audit_execution_command(self.job(), dict(self.env, USE_CURRENT_ENV = current))
                target = shlex.split(command[-1].splitlines()[-1]) if current == '0' else command
                self.assertEqual(target[-len(self.job()['argv']) - 2:], ['-u', 'audit_prior_geometry.py'] + self.job()['argv'])
                self.assertNotIn('run_prior.py', target)
                self.assertNotIn('--no-capture-output', command[-1])
    def test_training_program_stays_unchanged_for_all_arrays(self):
        for current in ['0', '1']:
            for group, rows in RUNS.items():
                for index in range(len(rows)):
                    with self.subTest(current = current, group = group, index = index):
                        run = make_run(group, index, self.env)
                        command = execution_command(run, dict(self.env, USE_CURRENT_ENV = current))
                        target = shlex.split(command[-1].splitlines()[-1]) if current == '0' else command
                        self.assertEqual(target[-len(run['argv']) - 2:], ['-u', 'run_prior.py'] + run['argv'])
    def test_reproduces_reported_argparse_error_with_legacy_routing(self):
        legacy_job = dict(self.job(), program = 'run_prior.py')
        command = execution_command(legacy_job, self.env)
        result = subprocess.run(command, cwd = self.root, env = self.env, capture_output = True, text = True, timeout = 15)
        self.assertEqual(result.returncode, 2)
        self.assertIn('argument family: invalid choice:', result.stderr)
        self.assertIn(str(self.root), result.stderr)
        self.assertFalse((self.root / 'audit_called.json').exists())
        self.assertFalse((self.root / 'logs').exists())
        self.assertFalse((self.root / 'saved_models').exists())
    def test_guard_rejects_legacy_helper_in_both_modes(self):
        for current in ['0', '1']:
            env = dict(self.env, USE_CURRENT_ENV = current)
            command = execution_command(dict(self.job(), program = 'run_prior.py'), env)
            with self.subTest(current = current), patch('slurm_scripts.run_prior_audit.execution_command', return_value = command):
                with self.assertRaisesRegex(RuntimeError, 'Audit routing mismatch'):
                    audit_execution_command(self.job(), env)
    def test_dry_runs_for_all_original_and_spatial_indices(self):
        originals = ['prior_v2_rn50_rn18_cka_s0_g1000', 'prior_v2_rn50_rn18_knn_s0_g1000',
                     'prior_v2_vit_tiny_d12_vit_tiny_cka_s0_g1000', 'prior_v2_vit_tiny_d12_vit_tiny_knn_s0_g1000',
                     'prior_v2_rn152_rn18_cka_s0_g1000', 'prior_v2_rn152_rn18_knn_s0_g1000',
                     'prior_v2_vit_tiny_d24_vit_tiny_cka_s0_g1000', 'prior_v2_vit_tiny_d24_vit_tiny_knn_s0_g1000']
        for current in ['0', '1']:
            for group in ['original', 'spatial']:
                for index in range(8):
                    env = dict(self.env, DRY_RUN = '1', SLURM_ARRAY_TASK_ID = str(index), AUDIT_GROUP = group, USE_CURRENT_ENV = current)
                    env.pop('SLURM_JOB_ID')
                    name = originals[index] if group == 'original' else make_run('D', index, env)['name']
                    with self.subTest(current = current, group = group, index = index):
                        result = self.launch(env)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn('audit_prior_geometry.py', result.stdout)
                        self.assertNotIn('-u run_prior.py', result.stdout)
                        self.assertIn(name, result.stdout)
                        self.assertFalse((self.root / 'audit_called.json').exists())
        self.assertFalse((self.root / 'geometry_audit').exists())
        self.assertFalse((self.root / 'logs').exists())
        self.assertFalse((self.root / 'saved_models').exists())
    def test_full_sbatch_path_for_all_audits_with_mock_srun_and_conda(self):
        for current in ['0', '1']:
            for group in ['original', 'spatial']:
                for index in range(8):
                    env = dict(self.env, SLURM_ARRAY_TASK_ID = str(index), AUDIT_GROUP = group, USE_CURRENT_ENV = current)
                    with self.subTest(current = current, group = group, index = index):
                        result = self.launch(env)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        called = json.loads((self.root / 'audit_called.json').read_text())
                        self.assertEqual(called['program'], 'audit_prior_geometry.py')
                        argv = called['argv']
                        self.assertEqual(argv[argv.index('--root') + 1], str(self.root))
                        self.assertEqual(argv[argv.index('--samples') + 1], '1024')
                        self.assertEqual(argv[argv.index('--data_dir') + 1], self.env['DATA_DIR'])
                        self.assertEqual(argv[argv.index('--device') + 1], 'cuda')
                        self.assertIn('--control_guide', argv)
                        name = argv[argv.index('--experiment') + 1]
                        self.assertEqual(name.endswith('_spatial'), group == 'spatial')
                        self.assertEqual(argv[argv.index('--output') + 1], str(self.root / 'geometry_audit' / (name + '.json')))
                        self.assertEqual(called['activated'], '1' if current == '0' else None)
                        self.assertFalse((self.root / 'unexpected').exists())
                        (self.root / 'audit_called.json').unlink()
    def test_mixed_versions_fail_during_dry_run_before_srun(self):
        helper = self.root / 'slurm_scripts' / 'run_prior_array.py'
        text = helper.read_text()
        current = '    program = [run.get(\'program\', \'run_prior.py\')] + run[\'argv\']'
        self.assertIn(current, text)
        helper.write_text(text.replace(current, '    program = [\'run_prior.py\'] + run[\'argv\']'))
        result = self.launch(dict(self.env, DRY_RUN = '1'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Audit routing mismatch', result.stderr)
        self.assertFalse((self.root / 'audit_called.json').exists())
    def test_activation_failure_does_not_call_audit(self):
        self.hook.write_text('conda() { return 23; }\n')
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exit status 23', result.stderr)
        self.assertFalse((self.root / 'audit_called.json').exists())
    def test_missing_hook_does_not_call_audit(self):
        self.hook.unlink()
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('conda.sh', result.stderr)
        self.assertFalse((self.root / 'audit_called.json').exists())
    def test_audit_failure_propagates_without_training(self):
        (self.root / 'audit_prior_geometry.py').write_text('raise SystemExit(19)\n')
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exit status 19', result.stderr)
        self.assertFalse((self.root / 'logs').exists())
        self.assertFalse((self.root / 'saved_models').exists())
    def test_missing_slurm_job_requires_dry_run(self):
        env = dict(self.env)
        env.pop('SLURM_JOB_ID')
        result = self.launch(env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Submit the audit sbatch script or use DRY_RUN=1', result.stderr)
        self.assertFalse((self.root / 'audit_called.json').exists())
    def test_invalid_indices_group_and_sample_count_fail(self):
        cases = [{'SLURM_ARRAY_TASK_ID': '-1'}, {'SLURM_ARRAY_TASK_ID': '8'},
                 {'AUDIT_GROUP': 'spatial', 'SLURM_ARRAY_TASK_ID': '8'}, {'AUDIT_GROUP': 'other'}, {'AUDIT_SAMPLES': '23'}]
        for values in cases:
            with self.subTest(values = values):
                result = self.launch(dict(self.env, DRY_RUN = '1', **values))
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / 'audit_called.json').exists())
    def test_audit_sbatch_syntax(self):
        result = subprocess.run(['bash', '-n', 'slurm_scripts/prior_geometry_audit.sbatch'], cwd = self.root, capture_output = True, text = True)
        self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == '__main__':
    unittest.main(verbosity = 2)
