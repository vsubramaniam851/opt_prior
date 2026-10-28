import contextlib
import copy
import io
import json
import math
import os
import subprocess
import tempfile
import unittest
import zipfile
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from PIL import Image
from types import SimpleNamespace
from unittest.mock import patch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.resnet import BasicBlock
from args import get_args
from audit_prior_mechanism import capture_endpoints, gradient_probe, hybrid_states, run_audit
from compare_prior import get_args as comparison_args
from compare_prior import analyze, discover, match_baselines, short_label, slurm_observations
from prior_utils import align_prior_stage, initialize_optimization_prior, prior_config, prior_fingerprint, prior_lr_factor
from prior_utils import prior_mode, prior_named_parameters, prior_parameter_diagnostics, prior_schedule, prior_state
from prior_utils import prior_trainable_parameters, validate_prior_args
from run_prior import build_command, get_run_args
from slurm_scripts.run_prior_array import make_run, parse_with, validate_initialization
from slurm_scripts.run_prior_audit import audit_execution_command, main as geometry_main
from slurm_scripts.run_prior_mechanism import main as mechanism_main
from test_compare_prior import fixture
from test_optimization_prior import test_args

class TinyResNet(nn.Module):
    def __init__(self, width = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU()
        self.maxpool = nn.Identity()
        for index in range(1, 5):
            setattr(self, f'layer{index}', nn.Sequential(BasicBlock(width, width)))
        self.fc = nn.Linear(width, 1000)

class BNDoseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(52)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for split in ['train', 'val']:
            (self.root / 'data' / split).mkdir(parents = True)
        self.env = {'DATA_DIR': str(self.root / 'data'), 'USE_CURRENT_ENV': '1'}

    def models(self):
        return TinyResNet(8), TinyResNet(4)

    def settings(self, **extra):
        return test_args(prior_descriptor = 'spatial', prior_start_stage = 3, prior_update_scope = 'suffix',
                         prior_bn_affine = extra.pop('prior_bn_affine', 'train'), prior_final_steps = extra.pop('prior_final_steps', None),
                         **extra)

    def loaders(self):
        images = torch.randn(16, 3, 8, 8, generator = torch.Generator().manual_seed(32))
        loader = DataLoader(TensorDataset(images, torch.zeros(16)), batch_size = 8, shuffle = False)
        return loader, loader

    def args(self, *extra):
        args = parse_with(get_run_args, ['resnet', 'CKA', '--data_dir', str(self.root / 'data'), *extra])
        return parse_with(get_args, build_command(args)[2:])

    def test_mechanism_audit_full_json_roundtrip(self):
        teacher, student = self.models()
        original = copy.deepcopy(student)
        initial_hash = prior_fingerprint(student)
        with torch.no_grad():
            student.layer4[0].conv2.weight.add_(0.01)
            student.layer4[0].bn2.weight.add_(0.01)
        name = 'synthetic_mechanism'
        folder = self.root / 'logs' / name
        folder.mkdir(parents = True)
        (self.root / 'saved_models').mkdir()
        data = self.root / 'data/train/class0'
        data.mkdir()
        rng = np.random.default_rng(75)
        for index in range(4):
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype = np.uint8)).save(data / f'{index}.png')
        cfg = dict(vars(self.settings()))
        cfg.update({'rep_batch': 4, 'data_dir': str(self.root / 'data')})
        meta = {'version': 2, 'complete': True, 'guide_unchanged': True, 'student': 'rn18', 'guide': 'rn50',
                'guide_seed': 1000, 'student_seed': 0, 'alignment_config': cfg, 'validation_indices': list(range(4)),
                'guide_initial_hash': prior_fingerprint(teacher), 'student_initial_hash': initial_hash,
                'student_aligned_hash': prior_fingerprint(student)}
        (folder / 'prior_alignment.json').write_text(json.dumps(meta))
        torch.save(student.state_dict(), self.root / 'saved_models' / f'orig_{name}.pt')
        args = SimpleNamespace(root = str(self.root), experiment = name, device = 'cpu', threads = 2,
                               workers = 0, samples = 4, data_dir = None, output = None)
        with patch('audit_prior_mechanism.make_prior_model', side_effect = lambda model, seed: copy.deepcopy(teacher if model == 'rn50' else original)), contextlib.redirect_stdout(io.StringIO()):
            result = run_audit(args)
        self.assertTrue(result['states_unchanged'])
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertEqual(len(result['measurements']), 48)
        self.assertEqual(len(result['gradient_probes']), 2)
        self.assertEqual(json.loads((self.root / 'mechanism_audit' / (name + '.json')).read_text()), result)

    def test_initializer_restores_all_parameters_after_fixed_affine(self):
        teacher, student = self.models()
        args = self.args('--prior_start_stage', '3', '--prior_update_scope', 'suffix', '--prior_descriptor', 'spatial',
                         '--prior_bn_affine', 'fixed', '--steps', '2', '--prior_final_steps', '1')
        args.knn_k = 3
        args.pool_hw = 2
        args.rep_val_interval = 1
        args.exp_name = 'fixed_initializer'
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            loaders = (*self.loaders(), list(range(16)))
            with patch('prior_utils.make_prior_model', return_value = student), patch('prior_utils.make_prior_loaders', return_value = loaders), contextlib.redirect_stdout(io.StringIO()):
                result = initialize_optimization_prior(args, teacher, torch.device('cpu'))
            self.assertTrue(all(param.requires_grad for param in result.parameters()))
            metadata = json.loads((self.root / 'logs/fixed_initializer/prior_alignment.json').read_text())
            self.assertTrue(metadata['guide_unchanged'])
            self.assertEqual(metadata['stages']['stage3']['steps'], 2)
            self.assertEqual(metadata['stages']['stage4']['steps'], 1)
            self.assertTrue(all(value['bn_affine_unchanged'] for value in metadata['stages'].values()))
        finally:
            os.chdir(previous)

    def test_mechanism_included_in_comparison_bundle(self):
        fixture(self.root, 'baseline', method = 'baseline')
        fixture(self.root, 'aligned')
        folder = self.root / 'mechanism_audit'
        folder.mkdir()
        data = {'experiment': 'aligned', 'states_unchanged': True, 'gradient_probes': []}
        (folder / 'aligned.json').write_text(json.dumps(data))
        output = self.root / 'review'
        args = comparison_args(['--root', str(self.root), '--output', str(output), '--bundle', '--include_geometry', '--no_plots'])
        with contextlib.redirect_stdout(io.StringIO()):
            analyze(args)
        with zipfile.ZipFile(output / 'prior_results_bundle.zip') as archive:
            self.assertEqual(json.loads(archive.read('mechanism_audit/aligned.json')), data)

    def test_bn_derivative_matches_autograd(self):
        x = torch.randn(6, 3, 4, 4, dtype = torch.float64, requires_grad = True)
        bn = nn.BatchNorm2d(3, affine = True, track_running_stats = False).double()
        bn.weight.data.copy_(torch.tensor([0.7, 1.3, 0.9]))
        g = torch.randn_like(x)
        grad, = torch.autograd.grad((bn(x) * g).sum(), x)
        dims = (0, 2, 3)
        sigma = (x.var(dim = dims, unbiased = False, keepdim = True) + bn.eps).sqrt()
        z = (x - x.mean(dim = dims, keepdim = True)) / sigma
        expected = bn.weight.view(1, -1, 1, 1) / sigma * (g - g.mean(dim = dims, keepdim = True) - z * (g * z).mean(dim = dims, keepdim = True))
        torch.testing.assert_close(grad, expected, atol = 1e-12, rtol = 1e-12)
        self.assertGreater(grad.norm().item(), 0)

    def test_freezing_affine_does_not_block_conv_gradients(self):
        for method in ['CKA', 'KNN']:
            teacher, student = self.models()
            settings = self.settings(repdist = method, prior_bn_affine = 'fixed')
            before = prior_state(student)
            report = gradient_probe(settings, teacher, student, next(iter(self.loaders()[0]))[0], before)
            conv = report['branches']['total']['groups']['conv']
            self.assertEqual(conv['trainable_tensors'], conv['nonzero_gradient_tensors'])
            self.assertGreater(conv['gradient_rms'], 0)
            for name in ['bn_scale', 'bn_bias']:
                self.assertEqual(report['branches']['total']['groups'][name]['trainable_parameters'], 0)
            self.assertEqual(prior_fingerprint(student), self.hash_state(student, before))

    def hash_state(self, model, weights):
        other = copy.deepcopy(model)
        other.load_state_dict(weights)
        return prior_fingerprint(other)

    def test_readout_hooks_include_true_pre_relu_skip_sum(self):
        teacher, student = self.models()
        settings = self.settings()
        images = next(iter(self.loaders()[0]))[0]
        modes = capture_endpoints(student, images, [2, 3], settings)
        self.assertEqual(set(modes), {'pre_bn_branch', 'pre_relu_sum', 'post_relu_stage'})
        self.assertEqual(len(modes['pre_bn_branch']), 2)
        self.assertFalse(torch.equal(modes['pre_relu_sum'][1][1], modes['pre_bn_branch'][1][1]))
        self.assertEqual(len(student.layer4[-1].relu._forward_pre_hooks), 0)

    def test_hybrids_only_swap_bn_affine(self):
        _, student = self.models()
        initial = prior_state(student)
        with torch.no_grad():
            for name, param, kind in prior_named_parameters(student):
                if kind != 'other':
                    param.add_(0.03)
        aligned = prior_state(student)
        states = hybrid_states(initial, aligned, student)
        kinds = {name: kind for name, _, kind in prior_named_parameters(student)}
        for name, value in states['aligned_convs_initial_bn'].items():
            expected = initial[name] if kinds.get(name, '').startswith('bn_') else aligned[name]
            self.assertTrue(torch.equal(value, expected))
        for name, value in states['initial_convs_aligned_bn'].items():
            expected = aligned[name] if kinds.get(name, '').startswith('bn_') else initial[name]
            self.assertTrue(torch.equal(value, expected))

    def test_gradient_probe_readonly_and_branches_sum(self):
        teacher, student = self.models()
        settings = self.settings()
        initial = prior_state(student)
        teacher_hash = prior_fingerprint(teacher)
        report = gradient_probe(settings, teacher, student, next(iter(self.loaders()[0]))[0], initial)
        branches = report['branches']
        self.assertAlmostEqual(branches['total']['weighted_loss'], sum(branches[name]['weighted_loss'] for name in ['spatial', 'global']), places = 6)
        self.assertTrue(all(row['connected'] and row['gradient_norm'] > 0 for row in report['convolution_gradients']))
        self.assertEqual(prior_fingerprint(student), self.hash_state(student, initial))
        self.assertEqual(prior_fingerprint(teacher), teacher_hash)
        self.assertTrue(all(param.grad is None for param in student.parameters()))
        json.dumps(report, allow_nan = False)

    def test_lr_horizon_exactly_matches_legacy(self):
        for horizon in [1, 5, 1000, 2000]:
            for step in [0, horizon // 2, horizon - 1]:
                warmup = max(1, int(horizon * 0.1))
                old = 0.1 + 0.9 * step / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, horizon - warmup)))
                self.assertEqual(prior_lr_factor(step, horizon), old)
        for step, horizon in [(-1, 5), (5, 5), (0, 0)]:
            with self.assertRaises(ValueError):
                prior_lr_factor(step, horizon)

    def train(self, teacher, student, args, index, trace):
        real = torch.optim.AdamW
        class RecordingAdam(real):
            def step(self, *argv, **kwargs):
                result = super().step(*argv, **kwargs)
                trace.append((self.param_groups[0]['lr'], prior_fingerprint(student)))
                return result
        metric = {'loss': 0.9, 'ucka': 0.1}
        def evaluate(*argv):
            metric['loss'] -= 0.1
            return dict(metric)
        stage = prior_schedule(teacher, student, 'cumulative', 3, 'suffix')[index - 2]
        with patch('prior_utils.torch.optim.AdamW', RecordingAdam), patch('prior_utils.prior_evaluate', side_effect = evaluate), contextlib.redirect_stdout(io.StringIO()):
            return align_prior_stage(args, teacher, student, *self.loaders(), stage, torch.device('cpu'))

    def test_final_cap_preserves_exact_update_prefix_and_schedule(self):
        teacher, student = self.models()
        full_trace, short_trace = [], []
        args = self.settings(max_stage_steps = 8, rep_scheduler = True, rep_val_interval = 2)
        self.train(copy.deepcopy(teacher), copy.deepcopy(student), args, 3, full_trace)
        args.prior_final_steps = 2
        report = self.train(copy.deepcopy(teacher), copy.deepcopy(student), args, 3, short_trace)
        self.assertEqual(full_trace[:2], short_trace)
        self.assertEqual(report['steps'], 2)

    def test_final_cap_does_not_shorten_first_phase(self):
        teacher, student = self.models()
        trace = []
        args = self.settings(max_stage_steps = 4, rep_scheduler = True, rep_val_interval = 2, prior_final_steps = 1)
        report = self.train(teacher, student, args, 2, trace)
        self.assertEqual(report['steps'], 4)
        self.assertEqual(len(trace), 4)

    def test_gradient_logging_does_not_change_updates(self):
        teacher, student = self.models()
        args = self.settings(max_stage_steps = 3, rep_val_interval = 1)
        a, b = [], []
        self.train(copy.deepcopy(teacher), copy.deepcopy(student), args, 3, a)
        args.prior_log_gradients = True
        result = self.train(copy.deepcopy(teacher), copy.deepcopy(student), args, 3, b)
        self.assertEqual(a, b)
        self.assertEqual(len(result['gradient_history']), 3)
        self.assertIn('conv', result['parameter_changes_by_type'])

    def test_fixed_affine_and_lower_prefix_preserved(self):
        for method in ['CKA', 'KNN']:
            teacher, student = self.models()
            args = self.settings(repdist = method, prior_bn_affine = 'fixed', max_stage_steps = 3, rep_val_interval = 1, prior_log_gradients = True)
            before = prior_state(student)
            report = self.train(teacher, student, args, 3, [])
            self.assertTrue(report['bn_affine_unchanged'])
            self.assertTrue(report['lower_prefix_unchanged'])
            self.assertGreater(report['parameter_changes_by_type']['conv']['relative_change'], 0)
            for name, param, kind in prior_named_parameters(student):
                if kind.startswith('bn_'):
                    self.assertTrue(torch.equal(param.detach(), before[name]))
            student.requires_grad_(True)
            self.assertTrue(all(param.requires_grad for param in student.parameters()))

    def test_optional_config_and_old_defaults(self):
        original = prior_config(self.args())
        self.assertNotIn('prior_bn_affine', original)
        self.assertNotIn('prior_final_steps', original)
        logged = self.args('--prior_log_gradients')
        self.assertEqual(original, prior_config(logged))
        args = self.args('--prior_bn_affine', 'fixed', '--steps', '2000', '--prior_final_steps', '250')
        validate_prior_args(args)
        self.assertEqual(prior_config(args)['prior_final_steps'], 250)
        self.assertEqual(prior_config(args)['prior_bn_affine'], 'fixed')

    def test_invalid_controls_rejected(self):
        for extra in [['--prior_final_steps', '0'], ['--prior_final_steps', '1001']]:
            with self.assertRaises(ValueError):
                self.args(*extra)
        for family, method, extra in [('vit', 'CKA', ['--prior_bn_affine', 'fixed']), ('resnet', 'baseline', ['--prior_final_steps', '250'])]:
            with self.assertRaises(ValueError):
                build_command(parse_with(get_run_args, [family, method, '--data_dir', str(self.root), *extra]))

    def test_f_mapping_names_and_existing_recipe(self):
        expected = [(250, 'train'), (1000, 'train'), (None, 'fixed'), (250, 'fixed'), (1000, 'fixed')]
        names = set()
        e = make_run('E', 2, self.env)
        for index in range(10):
            run = make_run('F', index, self.env)
            cfg = run['training']
            validate_prior_args(SimpleNamespace(**cfg))
            self.assertEqual((cfg['prior_final_steps'], cfg['prior_bn_affine']), expected[index // 2])
            self.assertEqual(cfg['repdist'], 'CKA' if index % 2 == 0 else 'KNN')
            self.assertEqual(cfg['max_stage_steps'], 2000)
            self.assertEqual(cfg['prior_start_stage'], 3)
            self.assertEqual(cfg['prior_update_scope'], 'suffix')
            self.assertTrue(cfg['prior_log_gradients'])
            self.assertEqual(run['phase'], 'align')
            for name in ['ft_epochs', 'ft_lr', 'ft_batch', 'ft_wd', 'ft_optimizer', 'ft_accum', 'label_smoothing', 'mixup_alpha']:
                self.assertEqual(cfg[name], e['training'][name])
            self.assertNotEqual(run['name'], e['name'])
            names.add(run['name'])
        self.assertEqual(len(names), 10)

    def test_f_reload_checks_missing_controls(self):
        run = make_run('F', 4, self.env)
        name = run['name']
        folder = self.root / 'logs' / name
        folder.mkdir(parents = True)
        (self.root / 'saved_models').mkdir()
        (self.root / 'saved_models' / f'orig_{name}.pt').write_bytes(b'synthetic')
        metadata = {'version': 2, 'complete': True, 'guide_unchanged': True, 'student_aligned_hash': 'synthetic',
                    'alignment_config': prior_config(SimpleNamespace(**run['training']))}
        (folder / 'args.json').write_text(json.dumps(run['training']))
        path = folder / 'prior_alignment.json'
        path.write_text(json.dumps(metadata))
        validate_initialization(self.root, name, run)
        del metadata['alignment_config']['prior_bn_affine']
        path.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            validate_initialization(self.root, name, run)

    def test_geometry_and_mechanism_routing(self):
        for index in range(10):
            env = dict(self.env, DRY_RUN = '1', AUDIT_GROUP = 'bn_dose', SLURM_ARRAY_TASK_ID = str(index))
            with patch.dict(os.environ, env, clear = True), contextlib.redirect_stdout(io.StringIO()) as output:
                geometry_main()
            self.assertIn('audit_prior_geometry.py', output.getvalue())
            self.assertIn(make_run('F', index, env)['name'], output.getvalue())
        for index in [0, 1]:
            env = dict(self.env, DRY_RUN = '1', SLURM_ARRAY_TASK_ID = str(index))
            with patch.dict(os.environ, env, clear = True), contextlib.redirect_stdout(io.StringIO()) as output:
                mechanism_main()
            self.assertIn('audit_prior_mechanism.py', output.getvalue())
            self.assertIn(make_run('E', index + 2, dict(env, ALIGN_STEPS = '2000'))['name'], output.getvalue())

    def test_comparison_controls_and_labels_separate(self):
        fixture(self.root, 'baseline', method = 'baseline')
        for name, affine, cap in [('short', 'train', 250), ('fixed', 'fixed', None)]:
            folder = fixture(self.root, name)
            for filename in ['args.json', 'training_args.json']:
                path = folder / filename
                if path.exists():
                    config = json.loads(path.read_text())
                    config.update({'prior_bn_affine': affine, 'prior_final_steps': cap})
                    path.write_text(json.dumps(config))
            path = folder / 'prior_alignment.json'
            meta = json.loads(path.read_text())
            if affine != 'train':
                meta['alignment_config']['prior_bn_affine'] = affine
            if cap is not None:
                meta['alignment_config']['prior_final_steps'] = cap
            path.write_text(json.dumps(meta))
        runs = discover(self.root, ['*'], [])
        match_baselines(runs)
        aligned = [run for run in runs if run['method'] != 'baseline']
        self.assertEqual(len({run['condition_id'] for run in aligned}), 2)
        self.assertEqual(len({short_label(run) for run in aligned}), 2)
        self.assertTrue(all(run['pair_status'] == 'matched' for run in aligned))

    def test_f_partial_logs_and_sbatch_syntax(self):
        fixture(self.root, 'partial', complete = False)
        log = self.root / 'logs/slurm-prior-f-bn-dose-1_0.out'
        log.write_text('Array F[0]: partial, epochs=100, action=full\nStep 0, Validation Loss 4.5, Top-1 Accuracy 18.2, Top-5 Accuracy 39.1, Samples 50000\n')
        runs = discover(self.root, ['*'], [])
        self.assertEqual(slurm_observations(self.root, runs)[0]['experiment'], 'partial')
        for filename in ['prior_f_bn_dose.sbatch', 'prior_mechanism_audit.sbatch']:
            subprocess.run(['bash', '-n', str(Path(__file__).parent / 'slurm_scripts' / filename)], check = True)

if __name__ == '__main__':
    unittest.main(verbosity = 2)
