import contextlib
import fcntl
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
import numpy as np
import torch

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from audit_prior_geometry import audit, spectrum
from compare_prior import analyze, depth_comparisons, discover, get_args as comparison_args, match_baselines, slurm_observations
from prior_utils import align_prior_stage, check_prior_run, make_prior_model, prior_config, prior_features, prior_fingerprint
from prior_utils import prior_metric, prior_schedule, prior_state
from rep_sims import CKA
from restart_prior_baseline import restart_baseline
from run_prior import build_command, get_run_args
from slurm_scripts.run_prior_array import RUNS, make_run, preflight
from slurm_scripts.test_prior_array import write_initialization
from test_compare_prior import fixture
from test_optimization_prior import test_args
from vits import ViT

class IterationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(83)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def baseline(self):
        folder = fixture(self.root, 'prior_v2_vit_tiny_baseline_s0', method = 'baseline', family = 'vit', complete = False)
        (self.root / 'saved_models').mkdir()
        name = folder.name
        for prefix in ['orig_', 'ft_']:
            (self.root / 'saved_models' / (prefix + name + '.pt')).write_bytes(prefix.encode())
        (self.root / 'saved_models' / name).mkdir()
        return name, folder

    def test_pooled_default_preserves_float32_outputs(self):
        x = torch.randn(8, 5, 7, 7)
        actual = prior_features(x)
        expected = [torch.nn.functional.adaptive_avg_pool2d(x, 4).flatten(1), x.mean(dim = (2, 3))]
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))
        tokens = torch.randn(8, 17, 6)
        self.assertTrue(torch.equal(prior_features(tokens)[1], tokens[:, 1:].mean(1)))

    def test_old_config_unchanged_until_opt_in(self):
        args = test_args()
        old = prior_config(args)
        self.assertNotIn('prior_descriptor', old)
        args.prior_descriptor = 'pooled'
        self.assertEqual(old, prior_config(args))
        args.prior_descriptor = 'spatial'
        self.assertEqual(prior_config(args)['prior_descriptor'], 'spatial')

    def test_reload_cannot_silently_drop_spatial_descriptor(self):
        args = test_args()
        args.optimization_prior = True
        args.reload_ft = True
        args.exp_name = 'spatial'
        args.prior_descriptor = 'spatial'
        folder = self.root / 'logs' / args.exp_name
        folder.mkdir(parents = True)
        (self.root / 'saved_models').mkdir()
        (self.root / 'saved_models' / 'orig_spatial.pt').write_bytes(b'test')
        (folder / 'prior_alignment.json').write_text(json.dumps({'version': 2, 'complete': True, 'alignment_config': prior_config(args)}))
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            check_prior_run(args)
            args.prior_descriptor = 'pooled'
            with self.assertRaisesRegex(ValueError, 'prior_descriptor'):
                check_prior_run(args)
        finally:
            os.chdir(previous)

    def test_spatial_constant_offsets_removed_only_in_residual(self):
        x = torch.randn(8, 5, 4, 4, dtype = torch.float64)
        offset = torch.randn(8, 5, 1, 1, dtype = torch.float64)
        a, b = prior_features(x, 4, 'spatial'), prior_features(x + offset, 4, 'spatial')
        self.assertTrue(torch.allclose(a[0], b[0], atol = 1e-12))
        self.assertTrue(torch.allclose(b[1] - a[1], offset.flatten(1), atol = 1e-12))

    def test_positive_per_image_scaling_removed(self):
        x = torch.randn(8, 5, 4, 4, dtype = torch.float64)
        scale = torch.rand(8, 1, 1, 1, dtype = torch.float64) + 0.5
        a, b = prior_features(x, 4, 'spatial'), prior_features(scale * x, 4, 'spatial')
        self.assertTrue(torch.allclose(a[0], b[0], atol = 1e-12))

    def test_vit_keeps_cls_and_retains_spatial_pattern(self):
        x = torch.randn(8, 17, 6)
        a = prior_features(x, 4, 'spatial')
        self.assertEqual(a[0].shape, (8, 96))
        self.assertTrue(torch.equal(a[1], x[:, 0]))
        changed = x.clone()
        changed[0, 1:] = changed[0, 1:].roll(1, dims = 0)
        b = prior_features(changed, 4, 'spatial')
        self.assertFalse(torch.allclose(a[0], b[0]))
        self.assertTrue(torch.allclose(x[:, 1:].mean(1), changed[:, 1:].mean(1)))

    def test_invalid_descriptor_and_grid_rejected(self):
        with self.assertRaises(ValueError):
            prior_features(torch.randn(8, 5, 1, 1), 4, 'spatial')
        with self.assertRaises(ValueError):
            prior_features(torch.randn(8, 12, 4), 4, 'spatial')
        with self.assertRaises(ValueError):
            prior_features(torch.randn(8, 4), 4, 'unknown')

    def test_spatial_cka_gradient_check(self):
        x = torch.randn(6, 3, 2, 2, dtype = torch.float64, requires_grad = True)
        y = torch.randn(6, 4, 2, 2, dtype = torch.float64)
        args = test_args(repdist = 'CKA')
        fun = lambda a: prior_metric(prior_features(a, 2, 'spatial'), prior_features(y, 2, 'spatial'), args)
        self.assertTrue(torch.autograd.gradcheck(fun, (x,), eps = 1e-6, atol = 2e-5))

    def test_spatial_knn_gradient_check(self):
        x = torch.randn(6, 5, 3, dtype = torch.float64, requires_grad = True)
        y = torch.randn(6, 5, 4, dtype = torch.float64)
        args = test_args(repdist = 'KNN', knn_k = 2)
        fun = lambda a: prior_metric(prior_features(a, 2, 'spatial'), prior_features(y, 2, 'spatial'), args)
        self.assertTrue(torch.autograd.gradcheck(fun, (x,), eps = 1e-6, atol = 2e-5))

    def test_zero_residual_has_finite_losses_and_gradients(self):
        for method in ['CKA', 'KNN']:
            x = torch.zeros(8, 3, 4, 4, requires_grad = True)
            y = torch.randn(8, 4, 4, 4)
            loss = prior_metric(prior_features(x, 4, 'spatial'), prior_features(y, 4, 'spatial'), test_args(repdist = method))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(x.grad.isfinite().all())

    def test_shared_dc_shortcut_detected(self):
        n = 64
        factor = torch.randn(n, 1, 1, 1)
        x = 30 * factor * torch.randn(1, 16, 1, 1) + torch.randn(n, 16, 4, 4)
        y = 30 * factor * torch.randn(1, 24, 1, 1) + torch.randn(n, 24, 4, 4)
        metric = CKA('cpu', unbiased = True)
        raw = metric.linear_CKA(prior_features(x)[0], prior_features(y)[0]).item()
        spatial = metric.linear_CKA(prior_features(x, 4, 'spatial')[0], prior_features(y, 4, 'spatial')[0]).item()
        self.assertGreater(raw, 0.99)
        self.assertLess(abs(spatial), 0.15)
        print({'synthetic_shared_dc_ucka': raw, 'independent_spatial_pattern_ucka': spatial})

    def test_cumulative_spatial_alignment_keeps_guide_unchanged(self):
        for method in ['CKA', 'KNN']:
            teacher = ViT(image_size = 32, patch_size = 8, hidden_dim = 24, num_layers = 4, num_heads = 3, mlp_dim = 48)
            student = ViT(image_size = 32, patch_size = 8, hidden_dim = 16, num_layers = 2, num_heads = 2, mlp_dim = 32)
            images = torch.randn(24, 3, 32, 32)
            loader = DataLoader(TensorDataset(images, torch.full((24,), float('nan'))), batch_size = 8)
            validation = DataLoader(TensorDataset(torch.randn(16, 3, 32, 32), torch.full((16,), float('nan'))), batch_size = 8)
            args = test_args(repdist = method, prior_descriptor = 'spatial', max_stage_steps = 3, rep_val_interval = 3)
            before = prior_fingerprint(teacher)
            with patch.object(teacher.model.heads, 'forward', side_effect = AssertionError('Guide classifier used')):
                with patch.object(student.model.heads, 'forward', side_effect = AssertionError('Student classifier used')):
                    for stage in prior_schedule(teacher, student):
                        result = align_prior_stage(args, teacher, student, loader, validation, stage, torch.device('cpu'))
                        self.assertTrue(np.isfinite(result['final']['loss']))
            self.assertEqual(prior_fingerprint(teacher), before)

    def test_restart_preview_has_no_side_effects(self):
        name, folder = self.baseline()
        before = sorted(str(p) for p in self.root.rglob('*'))
        result = restart_baseline(self.root, name)
        self.assertEqual(result['action'], 'preview')
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob('*')))

    def test_restart_archives_only_incomplete_baseline(self):
        name, folder = self.baseline()
        other = self.root / 'saved_models' / 'orig_other.pt'
        other.write_bytes(b'other')
        result = restart_baseline(self.root, name, True, True)
        self.assertEqual(result['action'], 'archived')
        self.assertFalse(folder.exists())
        backup = Path(result['backup'])
        self.assertEqual((backup / 'saved_models' / ('ft_' + name + '.pt')).read_bytes(), b'ft_')
        self.assertEqual(other.read_bytes(), b'other')
        self.assertFalse(discover(self.root, ['*'], []))

    def test_restart_requires_stopped_confirmation(self):
        name, folder = self.baseline()
        with self.assertRaises(ValueError):
            restart_baseline(self.root, name, True, False)
        self.assertTrue(folder.exists())

    def test_restart_rejects_active_lock(self):
        name, folder = self.baseline()
        locks = self.root / 'logs' / 'array_locks'
        locks.mkdir()
        with (locks / (name + '.lock')).open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(RuntimeError):
                restart_baseline(self.root, name, True, True)
        self.assertTrue(folder.exists())

    def test_restart_rejects_completed_and_aligned_runs(self):
        name, folder = self.baseline()
        final = self.root / 'saved_models' / ('ft_final_' + name + '.pt')
        final.write_bytes(b'final')
        with self.assertRaises(FileExistsError):
            restart_baseline(self.root, name, True, True)
        final.unlink()
        args = json.loads((folder / 'args.json').read_text())
        args['baseline'] = False
        (folder / 'args.json').write_text(json.dumps(args))
        with self.assertRaises(ValueError):
            restart_baseline(self.root, name, True, True)

    def test_restart_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            restart_baseline(self.root, '../elsewhere')

    def test_restart_rolls_back_failed_move(self):
        name, folder = self.baseline()
        move = shutil.move
        calls = [0]
        def fail_second(source, target):
            calls[0] += 1
            if calls[0] == 2:
                raise OSError('Synthetic disk error')
            return move(source, target)
        with patch('restart_prior_baseline.shutil.move', side_effect = fail_second):
            with self.assertRaises(OSError):
                restart_baseline(self.root, name, True, True)
        self.assertTrue(folder.is_dir())
        self.assertTrue((self.root / 'saved_models' / ('orig_' + name + '.pt')).is_file())

    def test_d_array_separates_names_and_phases(self):
        env = {'DATA_DIR': str(self.root)}
        old = {make_run(group, i, env)['name'] for group in ['B', 'C'] for i in range(len(RUNS[group]))}
        names = set()
        for i in range(8):
            run = make_run('D', i, env)
            self.assertEqual(run['phase'], 'align')
            self.assertEqual(run['training']['prior_descriptor'], 'spatial')
            self.assertTrue(run['training']['align_only'])
            names.add(run['name'])
            training = make_run('D', i, dict(env, PRIOR_PHASE = 'train'))
            self.assertEqual(training['name'], run['name'])
            self.assertTrue(training['training']['reload_ft'])
        self.assertEqual(len(names), 8)
        self.assertFalse(old.intersection(names))

    def test_d_preflight_alignment_then_training(self):
        for split in ['train', 'val']:
            (self.root / split).mkdir()
        env = {'DATA_DIR': str(self.root)}
        run = make_run('D', 0, env)
        self.assertEqual(preflight(self.root, run), 'full')
        write_initialization(self.root, run)
        self.assertEqual(preflight(self.root, run), 'complete_alignment')
        training = make_run('D', 0, dict(env, PRIOR_PHASE = 'train'))
        self.assertEqual(preflight(self.root, training), 'train')

    def test_vit_depth_comparison_without_baseline(self):
        fixture(self.root, 'd8', family = 'vit', depth = 8, top1 = 59.6)
        fixture(self.root, 'd12', family = 'vit', depth = 12, top1 = 60.452)
        runs = discover(self.root, ['*'], [])
        match_baselines(runs)
        rows = depth_comparisons(runs)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['deeper_minus_shallower_pp'], 0.852)
        self.assertTrue(all(run['delta_pp'] is None for run in runs))

    def test_different_descriptor_not_combined_in_depth_contrasts(self):
        folder = fixture(self.root, 'd8', family = 'vit', depth = 8)
        folder = fixture(self.root, 'd12', family = 'vit', depth = 12)
        for filename in ['args.json', 'prior_alignment.json']:
            path = folder / filename
            value = json.loads(path.read_text())
            if filename == 'args.json':
                value['prior_descriptor'] = 'spatial'
            else:
                value['alignment_config']['prior_descriptor'] = 'spatial'
            path.write_text(json.dumps(value))
        runs = discover(self.root, ['*'], [])
        match_baselines(runs)
        self.assertEqual(depth_comparisons(runs), [])

    def test_depth_contrasts_require_matching_initialization(self):
        fixture(self.root, 'd8', family = 'vit', depth = 8)
        folder = fixture(self.root, 'd12', family = 'vit', depth = 12)
        path = folder / 'prior_alignment.json'
        value = json.loads(path.read_text())
        value['student_initial_hash'] = 'different_initialization'
        path.write_text(json.dumps(value))
        runs = discover(self.root, ['*'], [])
        match_baselines(runs)
        self.assertEqual(depth_comparisons(runs), [])

    def test_new_array_slurm_log_and_nonfatal_errors_are_retained(self):
        fixture(self.root, 'spatial')
        path = self.root / 'logs' / 'slurm-prior-d-spatial-100_0.out'
        path.write_text('Array D[0]: spatial, epochs=100\nStep 25, Validation Loss 2.1, Top-1 Accuracy 56.4, Top-5 Accuracy 81.2, Samples 50000\n')
        path.with_suffix('.err').write_text('[ERROR] nvtiff_stream creation failed\nEpoch completed afterward\n')
        rows = slurm_observations(self.root, discover(self.root, ['*'], []))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['validation'][0]['top1'], 56.4)
        self.assertEqual(len(rows[0]['errors']), 1)
        self.assertIn('Epoch completed afterward', rows[0]['tail'])

    def test_geometry_audit_is_bundled_only_when_requested(self):
        fixture(self.root, 'aligned')
        folder = self.root / 'geometry_audit'
        folder.mkdir()
        (folder / 'aligned.json').write_text(json.dumps({'experiment': 'aligned', 'rows': []}))
        (folder / 'unrelated.json').write_text(json.dumps({'experiment': 'unrelated', 'rows': []}))
        arguments = ['--root', str(self.root), '--output', str(self.root / 'review'), '--bundle', '--no_plots']
        with contextlib.redirect_stdout(io.StringIO()):
            result = analyze(comparison_args(arguments))
        with zipfile.ZipFile(result['bundle']) as archive:
            self.assertNotIn('geometry_audit/aligned.json', archive.namelist())
        with contextlib.redirect_stdout(io.StringIO()):
            result = analyze(comparison_args(arguments + ['--include_geometry']))
        with zipfile.ZipFile(result['bundle']) as archive:
            self.assertIn('geometry_audit/aligned.json', archive.namelist())
            self.assertNotIn('geometry_audit/unrelated.json', archive.namelist())

    def test_spectrum_reports_known_rank(self):
        x = torch.randn(8, 1) @ torch.randn(1, 10)
        result = spectrum(x)
        self.assertAlmostEqual(result['participation_rank'], 1, places = 5)
        self.assertAlmostEqual(result['top1_variance_fraction'], 1, places = 5)
        self.assertEqual(spectrum(torch.zeros(8, 3))['participation_rank'], 0)

    def test_audit_on_actual_images_and_saved_weights(self):
        folder = self.root / 'logs' / 'audit_test'
        folder.mkdir(parents = True)
        (self.root / 'saved_models').mkdir()
        data = self.root / 'images' / 'train' / 'class0'
        data.mkdir(parents = True)
        generator = np.random.default_rng(17)
        for i in range(8):
            Image.fromarray(generator.integers(0, 256, size = (240, 240, 3), dtype = np.uint8)).save(data / f'{i}.png')
        teacher = make_prior_model('vit_tiny', 1000, 8)
        student = make_prior_model('vit_tiny', 0)
        metadata = {'version': 2, 'complete': True, 'guide_unchanged': True, 'baseline': False, 'student': 'vit_tiny',
                    'guide': 'vit_tiny', 'guide_seed': 1000, 'student_seed': 0, 'guide_depth_override': 8,
                    'guide_initial_hash': prior_fingerprint(teacher), 'student_initial_hash': prior_fingerprint(student),
                    'student_aligned_hash': prior_fingerprint(student), 'normalization': 'batch',
                    'validation_indices': list(range(8)), 'alignment_config': {'rep_batch': 8, 'knn_k': 3,
                    'pool_hw': 4, 'data_dir': str(data.parents[1])}}
        (folder / 'prior_alignment.json').write_text(json.dumps(metadata))
        torch.save(student.state_dict(), self.root / 'saved_models' / 'orig_audit_test.pt')
        args = SimpleNamespace(root = str(self.root), experiment = 'audit_test', data_dir = None, samples = 8,
                               device = 'cpu', workers = 0, threads = 2, control_guide = True, output = None)
        result = audit(args)
        self.assertEqual(result['samples'], 8)
        self.assertTrue(result['states_unchanged'])
        self.assertEqual(len(result['rows']), 80)
        self.assertTrue((self.root / 'geometry_audit' / 'audit_test.json').is_file())
        del teacher, student

if __name__ == '__main__':
    unittest.main(verbosity = 2)
