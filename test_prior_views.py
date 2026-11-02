import contextlib
import copy
import io
import json
import os
import subprocess
import tempfile
import unittest
import zipfile
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from args import get_args
from audit_prior_views import run_audit
from compare_prior import analyze, discover, get_args as comparison_args, match_baselines, short_label, slurm_observations, view_comparisons
from prior_utils import align_prior_stage, initialize_optimization_prior, make_prior_loaders, prior_batch, prior_config
from prior_utils import prior_evaluate, prior_features, prior_fingerprint, prior_mode, prior_schedule, prior_state
from prior_utils import prior_trainable_parameters, validate_prior_args
from prior_views import PriorViewDataset, view_diagnostics, view_kernel, view_log_neighbors, view_loss, view_spec, view_target
from rep_sims import CKA, DifferentiableKNN
from run_prior import build_command, get_run_args
from slurm_scripts.run_prior_array import make_run, parse_with, preflight, validate_initialization
from slurm_scripts.run_prior_audit import main as geometry_main
from slurm_scripts.run_prior_view_audit import main as view_main
from test_compare_prior import fixture
from test_optimization_prior import test_args
from test_prior_bn_dose import TinyResNet

class PairTensors(Dataset):
    def __init__(self, images, labels = None):
        self.images = images
        self.labels = torch.zeros(len(images)) if labels is None else labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        return (image, image.flip(-1) * 0.9 + 0.02), self.labels[index]

class ViewTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(45)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for split in ['train', 'val']:
            (self.root / 'data' / split).mkdir(parents = True)
        self.env = {'DATA_DIR': str(self.root / 'data'), 'USE_CURRENT_ENV': '1'}

    def settings(self, **extra):
        return test_args(prior_descriptor = 'spatial_only', prior_start_stage = 3, prior_update_scope = 'suffix',
                         prior_view_mode = extra.pop('prior_view_mode', 'shared'), prior_view_crop_min = 0.5,
                         prior_view_seed = 41000, rep_batch = 8, **extra)

    def matrices(self, grad = False):
        x = [torch.randn(8, 5, dtype = torch.float64, requires_grad = grad) for _ in range(2)]
        y = [torch.randn(8, 9, dtype = torch.float64, requires_grad = grad) for _ in range(2)]
        return x, y

    def images(self, count = 12):
        folder = self.root / 'data/train/class0'
        folder.mkdir(exist_ok = True)
        rng = np.random.default_rng(21)
        for i in range(count):
            Image.fromarray(rng.integers(0, 256, (32, 36, 3), dtype = np.uint8)).save(folder / f'{i:04d}.png')

    def test_separate_matches_original_cka_value_and_gradient(self):
        x, y = self.matrices(True)
        for mode in ['normal', 'debiased']:
            args = self.settings(cka_loss = mode)
            loss = view_loss(x, y, args, 'separate')
            reference = sum(1 - CKA('cpu', unbiased = mode == 'debiased').linear_CKA(a, b.detach()) for a, b in zip(x, y)) / 2
            self.assertTrue(torch.allclose(loss, reference, atol = 1e-12))
            grad = torch.autograd.grad(loss, x, retain_graph = True)
            expected = torch.autograd.grad(reference, x, retain_graph = True)
            for a, b in zip(grad, expected):
                self.assertTrue(torch.allclose(a, b, atol = 1e-10))

    def test_separate_matches_original_knn_value_and_gradient(self):
        x, y = self.matrices(True)
        args = self.settings(repdist = 'KNN')
        loss = view_loss(x, y, args, 'separate')
        metric = DifferentiableKNN('cpu', 3, temperature = 0.2)
        reference = sum(metric.diff_knn_distribution_loss(a, b) for a, b in zip(x, y)) / 2
        self.assertTrue(torch.allclose(loss, reference, atol = 1e-12))
        actual, expected = torch.autograd.grad(loss, x), torch.autograd.grad(reference, x)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.allclose(a, b, atol = 1e-10))

    def test_shared_cka_matches_explicit_average_kernel(self):
        x, y = self.matrices()
        args = self.settings()
        target = sum(view_kernel(t) for t in y) / 2
        target /= target.norm()
        reference = sum(1 - (view_kernel(a) * target).sum() for a in x) / 2
        self.assertTrue(torch.allclose(view_loss(x, y, args), reference, atol = 1e-12))

    def test_shared_knn_averages_probabilities_not_scores(self):
        x, y = self.matrices()
        args = self.settings(repdist = 'KNN')
        q = sum(view_log_neighbors(t, 3, 0.2).exp() for t in y) / 2
        reference = sum((q * (q.log() - view_log_neighbors(a, 3, 0.2))).sum(1).mean() for a in x) / 2
        self.assertTrue(torch.allclose(view_loss(x, y, args), reference, atol = 1e-12))
        _, target = view_target(y, args)
        self.assertEqual(tuple(target.shape), (8, 7))
        self.assertTrue(torch.allclose(target.exp().sum(1), torch.ones(8, dtype = torch.float64)))

    def test_cka_shared_gradcheck(self):
        x, y = self.matrices()
        args = self.settings()
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: view_loss([a, b], y, args), tuple(v.requires_grad_() for v in x)))

    def test_knn_shared_gradcheck(self):
        x, y = self.matrices()
        args = self.settings(repdist = 'KNN')
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: view_loss([a, b], y, args), tuple(v.requires_grad_() for v in x)))

    def test_identical_views_reduce_to_one_view(self):
        x, y = self.matrices()
        for method in ['CKA', 'KNN']:
            args = self.settings(repdist = method)
            a = view_loss([x[0], x[0]], [y[0], y[0]], args, 'shared')
            b = view_loss([x[0], x[0]], [y[0], y[0]], args, 'separate')
            self.assertTrue(torch.allclose(a, b, atol = 1e-12))

    def test_shared_and_separate_are_different(self):
        x, y = self.matrices()
        for method in ['CKA', 'KNN']:
            args = self.settings(repdist = method)
            self.assertGreater(abs((view_loss(x, y, args, 'shared') - view_loss(x, y, args, 'separate')).item()), 1e-5)

    def test_view_swap_and_image_permutation_invariance(self):
        x, y = self.matrices()
        order = torch.randperm(8)
        for method in ['CKA', 'KNN']:
            for mode in ['separate', 'shared']:
                args = self.settings(repdist = method, prior_view_mode = mode)
                a = view_loss(x, y, args)
                b = view_loss(x[::-1], y[::-1], args)
                c = view_loss([v[order] for v in x], [v[order] for v in y], args)
                self.assertTrue(torch.allclose(a, b, atol = 1e-12))
                self.assertTrue(torch.allclose(a, c, atol = 1e-12))

    def test_teacher_detached_and_both_student_views_receive_gradients(self):
        for method in ['CKA', 'KNN']:
            x, y = self.matrices(True)
            loss = view_loss(x, y, self.settings(repdist = method))
            loss.backward()
            self.assertTrue(all(t.grad is None for t in y))
            self.assertTrue(all(a.grad is not None and torch.isfinite(a.grad).all() and a.grad.norm() > 0 for a in x))

    def test_kernel_removes_diagonal_and_preserves_float64(self):
        x, _ = self.matrices()
        kernel = view_kernel(x[0])
        self.assertEqual(kernel.dtype, torch.float64)
        self.assertTrue(torch.equal(kernel.diag(), torch.zeros(8, dtype = torch.float64)))
        self.assertTrue(torch.allclose(kernel.sum(1), torch.zeros(8, dtype = torch.float64), atol = 1e-12))

    def test_uniform_neighborhoods_are_measured_not_hidden(self):
        x = [torch.zeros(8, 4), torch.zeros(8, 4)]
        diagnostics = view_diagnostics(x, x, self.settings(repdist = 'KNN'))
        self.assertAlmostEqual(diagnostics['target_neighbor_entropy_normalized'], 1.0, places = 6)
        self.assertAlmostEqual(diagnostics['target_effective_neighbors'], 7.0, places = 5)
        self.assertAlmostEqual(diagnostics['guide_kernel_coherence'], 0.0)

    def test_invalid_loss_inputs_rejected(self):
        x, y = self.matrices()
        for mode in ['single', 'unknown']:
            with self.assertRaises(ValueError):
                view_loss(x, y, self.settings(), mode)
        with self.assertRaises(ValueError):
            view_loss(x[:1], y, self.settings())
        with self.assertRaises(ValueError):
            view_loss(x, [y[0][:4], y[1]], self.settings())
        with self.assertRaises(ValueError):
            view_log_neighbors(x[0], 8, 0.2)

    def test_spatial_only_matches_old_spatial_branch(self):
        for z in [torch.randn(8, 5, 4, 4), torch.randn(8, 17, 5)]:
            a = prior_features(z, 4, 'spatial_only')
            b = prior_features(z, 4, 'spatial')
            self.assertEqual(len(a), 1)
            self.assertTrue(torch.equal(a[0], b[0]))

    def test_fixed_validation_views_preserve_rng_and_are_distinct(self):
        image = Image.fromarray(np.random.default_rng(3).integers(0, 256, (30, 40, 3), dtype = np.uint8))
        dataset = PriorViewDataset([(image, float('nan'))], transforms.Compose([transforms.RandomResizedCrop(16), transforms.ToTensor()]), 41000)
        state = torch.random.get_rng_state().clone()
        a, _ = dataset[0]
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        torch.randn(10)
        b, _ = dataset[0]
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
        self.assertFalse(torch.equal(a[0], a[1]))
        reserve = PriorViewDataset(dataset.dataset, dataset.transform, 1041003)
        c, _ = reserve[0]
        self.assertFalse(torch.equal(a[0], c[0]))

    def test_fixed_views_do_not_depend_on_worker_count(self):
        image = Image.fromarray(np.random.default_rng(5).integers(0, 256, (30, 40, 3), dtype = np.uint8))
        dataset = PriorViewDataset([(image, i) for i in range(6)], transforms.Compose([transforms.RandomResizedCrop(16), transforms.ToTensor()]), 41000)
        a = list(DataLoader(dataset, batch_size = 3, num_workers = 0))
        b = list(DataLoader(dataset, batch_size = 3, num_workers = 2))
        self.assertTrue(all(torch.equal(x[0][v], y[0][v]) for x, y in zip(a, b) for v in range(2)))

    def test_holdout_image_disjoint_and_tail_preserved(self):
        self.images(15)
        args = self.settings(data_dir = str(self.root / 'data'), rep_val_size = 7, rep_split_seed = 0, seed = 0, num_workers = 0)
        args.rep_batch = 4
        train, val, indices = make_prior_loaders(args)
        self.assertFalse(set(train.dataset.indices) & set(indices))
        self.assertEqual(sum(len(x) for x in val.batch_sampler), 7)
        a = list(val)
        b = list(val)
        self.assertTrue(all(torch.equal(x[0][v], y[0][v]) for x, y in zip(a, b) for v in range(2)))

    def test_both_objectives_receive_identical_seeded_training_views(self):
        self.images(15)
        outputs = []
        for mode in ['separate', 'shared']:
            torch.manual_seed(10000)
            args = self.settings(data_dir = str(self.root / 'data'), rep_val_size = 7, rep_split_seed = 0,
                                 seed = 0, num_workers = 0, prior_view_mode = mode)
            args.rep_batch = 4
            train, _, _ = make_prior_loaders(args)
            outputs.append(list(train))
        self.assertTrue(all(torch.equal(x[0][v], y[0][v]) for x, y in zip(*outputs) for v in range(2)))

    def test_forward_calls_and_bn_batch_sizes_match_between_modes(self):
        teacher, student = TinyResNet(8), TinyResNet(4)
        images = (torch.randn(8, 3, 8, 8), torch.randn(8, 3, 8, 8))
        hashes = [prior_fingerprint(teacher), prior_fingerprint(student)]
        for mode in ['separate', 'shared']:
            sizes = {'guide': [], 'student': []}
            handles = [model.bn1.register_forward_pre_hook(lambda module, inputs, state=state: sizes[state].append(inputs[0].size(0)))
                       for state, model in [('guide', teacher), ('student', student)]]
            with prior_mode(teacher), prior_mode(student), patch.object(teacher.fc, 'forward', side_effect = AssertionError('classifier')), patch.object(student.fc, 'forward', side_effect = AssertionError('classifier')):
                loss, endpoints, _, _ = prior_batch(self.settings(prior_view_mode = mode), teacher, student, images, [2, 3], [2, 3])
                loss.backward()
            self.assertEqual(sizes, {'guide': [8, 8], 'student': [8, 8]})
            self.assertEqual(len(endpoints), 2)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            for handle in handles:
                handle.remove()
        self.assertEqual(hashes, [prior_fingerprint(teacher), prior_fingerprint(student)])

    def test_two_view_validation_is_label_free_and_reproducible(self):
        teacher, student = TinyResNet(8), TinyResNet(4)
        images = torch.randn(16, 3, 8, 8)
        args = self.settings()
        a = DataLoader(PairTensors(images), batch_size = 8)
        b = DataLoader(PairTensors(images, torch.full((16,), float('nan'))), batch_size = 8)
        metrics_a = prior_evaluate(args, teacher, student, a, 3, 3, torch.device('cpu'))
        metrics_b = prior_evaluate(args, teacher, student, b, 3, 3, torch.device('cpu'))
        self.assertEqual(metrics_a, metrics_b)
        self.assertEqual(metrics_a['samples'], 16)
        self.assertIn('endpoint4_student_cross_view_ucka', metrics_a)
        self.assertIn('target_neighbor_entropy_normalized', metrics_a)

    def test_progressive_two_view_alignment_and_lower_prefix(self):
        images = torch.randn(24, 3, 8, 8)
        train = DataLoader(PairTensors(images[:16]), batch_size = 8)
        val = DataLoader(PairTensors(images[16:]), batch_size = 8)
        for method in ['CKA', 'KNN']:
            for mode in ['separate', 'shared']:
                torch.manual_seed(18)
                teacher, student = TinyResNet(8), TinyResNet(4)
                initial = prior_fingerprint(teacher)
                args = self.settings(repdist = method, prior_view_mode = mode, max_stage_steps = 8, rep_val_interval = 4,
                                     progressive_lrs = [1e-3], prior_log_gradients = True)
                results = []
                with contextlib.redirect_stdout(io.StringIO()):
                    for stage in prior_schedule(teacher, student, 'cumulative', 3, 'suffix'):
                        result = align_prior_stage(args, teacher, student, train, val, stage, torch.device('cpu'))
                        self.assertTrue(result['lower_prefix_unchanged'])
                        self.assertEqual(result['views_per_image'], 2)
                        self.assertLessEqual(result['final']['loss'], result['initial']['loss'])
                        results.append(result)
                self.assertEqual(initial, prior_fingerprint(teacher))
                print({'two_view_synthetic': method + '_' + mode, 'accepted': [r['accepted'] for r in results],
                       'initial_loss': [r['initial']['loss'] for r in results], 'selected_loss': [r['final']['loss'] for r in results]})

    def test_initializer_metadata_and_downstream_unfreeze(self):
        run = make_run('G', 2, self.env)
        args = SimpleNamespace(**run['training'])
        args.max_stage_steps, args.rep_val_interval, args.knn_k, args.rep_batch = 2, 1, 3, 8
        args.exp_name = 'two_view_init'
        teacher, student = TinyResNet(8), TinyResNet(4)
        loader = DataLoader(PairTensors(torch.randn(16, 3, 8, 8)), batch_size = 8)
        old = Path.cwd()
        try:
            os.chdir(self.root)
            with patch('prior_utils.make_prior_model', return_value = student), patch('prior_utils.make_prior_loaders', return_value = (loader, loader, list(range(16)))), contextlib.redirect_stdout(io.StringIO()):
                result = initialize_optimization_prior(args, teacher, torch.device('cpu'))
        finally:
            os.chdir(old)
        self.assertTrue(all(p.requires_grad for p in result.parameters()))
        meta = json.loads((self.root / 'logs/two_view_init/prior_alignment.json').read_text())
        self.assertEqual(meta['view_spec']['mode'], 'shared')
        self.assertTrue(meta['guide_unchanged'])
        self.assertEqual(list(meta['stages']), ['stage3', 'stage4'])

    def test_g_array_mapping_unique_names_and_unchanged_supervised_recipe(self):
        baseline = make_run('B', 0, self.env)
        names = []
        for index in range(4):
            run = make_run('G', index, self.env)
            cfg = run['training']
            self.assertEqual(cfg['prior_view_mode'], 'separate' if index < 2 else 'shared')
            self.assertEqual(cfg['repdist'], 'CKA' if index % 2 == 0 else 'KNN')
            self.assertEqual(cfg['prior_descriptor'], 'spatial_only')
            self.assertEqual(cfg['max_stage_steps'], 2000)
            self.assertEqual(cfg['prior_bn_affine'], 'train')
            self.assertEqual(run['phase'], 'align')
            for key in ['ft_epochs', 'ft_batch', 'ft_lr', 'ft_wd', 'ft_optimizer', 'ft_accum', 'label_smoothing']:
                self.assertEqual(cfg[key], baseline['training'][key])
            validate_prior_args(SimpleNamespace(**cfg))
            names.append(run['name'])
        self.assertEqual(len(set(names)), 4)

    def test_config_defaults_preserve_old_fingerprints(self):
        run = make_run('F', 0, self.env)
        args = SimpleNamespace(**run['training'])
        current = prior_config(args)
        for key in ['prior_view_mode', 'prior_view_crop_min', 'prior_view_seed']:
            self.assertNotIn(key, current)
        del args.prior_view_mode, args.prior_view_crop_min, args.prior_view_seed
        self.assertEqual(current, prior_config(args))

    def test_reloading_a_different_view_mode_is_rejected(self):
        run = make_run('G', 0, self.env)
        name = run['name']
        folder = self.root / 'logs' / name
        folder.mkdir(parents = True)
        (self.root / 'saved_models').mkdir()
        (self.root / 'saved_models' / f'orig_{name}.pt').write_bytes(b'fixture')
        meta = {'version': 2, 'complete': True, 'guide_unchanged': True, 'student_aligned_hash': 'fixture',
                'alignment_config': prior_config(SimpleNamespace(**run['training']))}
        (folder / 'args.json').write_text(json.dumps(dict(run['training'], align_only = True)))
        (folder / 'prior_alignment.json').write_text(json.dumps(meta))
        self.assertEqual(preflight(self.root, run), 'complete_alignment')
        changed = copy.deepcopy(run)
        changed['training']['prior_view_mode'] = 'shared'
        with self.assertRaises(ValueError):
            validate_initialization(self.root, name, changed)

    def test_launcher_rejects_two_views_with_pooled_and_baseline(self):
        for method, descriptor in [('CKA', 'pooled'), ('baseline', 'spatial_only')]:
            args = parse_with(get_run_args, ['resnet', method, '--data_dir', '/data', '--prior_view_mode', 'shared', '--prior_descriptor', descriptor])
            with self.assertRaises(ValueError):
                build_command(args)
        args = parse_with(get_run_args, ['resnet', 'CKA', '--data_dir', '/data', '--prior_view_crop_min', 'nan'])
        with self.assertRaises(ValueError):
            build_command(args)

    def test_view_and_standard_audit_routing(self):
        for index in range(4):
            env = dict(self.env, AUDIT_GROUP = 'views', DRY_RUN = '1', SLURM_ARRAY_TASK_ID = str(index))
            for main, program in [(geometry_main, 'audit_prior_geometry.py'), (view_main, 'audit_prior_views.py')]:
                output = io.StringIO()
                with patch.dict(os.environ, env, clear = True), contextlib.redirect_stdout(output):
                    main()
                self.assertIn(program, output.getvalue())
                self.assertIn(make_run('G', index, env)['name'], output.getvalue())
                self.assertNotIn('-u run_prior.py', output.getvalue())

    def test_view_audit_json_roundtrip_and_state_checks(self):
        self.images(4)
        teacher, original = TinyResNet(8), TinyResNet(4)
        student = copy.deepcopy(original)
        with torch.no_grad():
            student.layer4[0].conv2.weight.add_(0.001)
        name = 'view_audit_fixture'
        folder = self.root / 'logs' / name
        folder.mkdir(parents = True)
        (self.root / 'saved_models').mkdir()
        cfg = vars(self.settings(data_dir = str(self.root / 'data')))
        cfg['rep_batch'] = 4
        meta = {'version': 2, 'complete': True, 'guide_unchanged': True, 'student': 'rn18', 'guide': 'rn50',
                'guide_seed': 1000, 'student_seed': 0, 'alignment_config': cfg, 'view_spec': view_spec(SimpleNamespace(**cfg)),
                'validation_indices': list(range(4)), 'guide_initial_hash': prior_fingerprint(teacher),
                'student_initial_hash': prior_fingerprint(original), 'student_aligned_hash': prior_fingerprint(student)}
        (folder / 'prior_alignment.json').write_text(json.dumps(meta))
        torch.save(student.state_dict(), self.root / 'saved_models' / f'orig_{name}.pt')
        args = SimpleNamespace(root = str(self.root), experiment = name, device = 'cpu', threads = 2,
                               workers = 0, samples = 4, data_dir = None, output = None)
        with patch('audit_prior_views.make_prior_model', side_effect = lambda model, seed, depth=None: copy.deepcopy(teacher if model == 'rn50' else original)), contextlib.redirect_stdout(io.StringIO()):
            report = run_audit(args)
        self.assertEqual(len(report['rows']), 32)
        self.assertEqual(len(report['batch_rows']), 32)
        self.assertTrue(report['states_unchanged'])
        self.assertEqual(report['optimizer_steps'], 0)
        self.assertEqual(json.loads((self.root / 'view_audit' / (name + '.json')).read_text()), report)

    def test_comparison_groups_and_bundles_view_audits(self):
        fixture(self.root, 'baseline', method = 'baseline')
        for mode in ['separate', 'shared']:
            folder = fixture(self.root, mode)
            args = json.loads((folder / 'args.json').read_text())
            args.update(prior_descriptor = 'spatial_only', prior_view_mode = mode, prior_view_crop_min = 0.5, prior_view_seed = 41000)
            (folder / 'args.json').write_text(json.dumps(args))
            meta = json.loads((folder / 'prior_alignment.json').read_text())
            meta['alignment_config'] = prior_config(SimpleNamespace(**args))
            (folder / 'prior_alignment.json').write_text(json.dumps(meta))
            path = self.root / 'view_audit' / f'{mode}.json'
            path.parent.mkdir(exist_ok = True)
            path.write_text(json.dumps({'experiment': mode, 'rows': []}))
        runs = discover(self.root, ['*'], [])
        match_baselines(runs)
        aligned = [r for r in runs if r['method'] != 'baseline']
        self.assertTrue(all(r['pair_status'] == 'matched' for r in aligned))
        self.assertEqual(len({r['condition_id'] for r in aligned}), 2)
        self.assertTrue(all('views=' in short_label(r) for r in aligned))
        pairs = view_comparisons(runs)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['shared_minus_separate_pp'], 0.0)
        copied = copy.deepcopy(runs)
        next(r for r in copied if r['experiment'] == 'shared')['alignment_controls']['prior_view_crop_min'] = 0.8
        self.assertEqual(view_comparisons(copied), [])
        duplicate = copy.deepcopy(next(r for r in runs if r['experiment'] == 'separate'))
        self.assertEqual(view_comparisons(runs + [duplicate]), [])
        args = parse_with(comparison_args, ['--root', str(self.root), '--output', str(self.root / 'review'), '--bundle', '--no_plots', '--include_geometry'])
        with contextlib.redirect_stdout(io.StringIO()):
            analyze(args)
        with zipfile.ZipFile(self.root / 'review/prior_results_bundle.zip') as z:
            self.assertIn('view_audit/shared.json', z.namelist())
            self.assertIn('view_audit/separate.json', z.namelist())

    def test_bfloat16_relations_are_evaluated_in_float32(self):
        x, y = self.matrices()
        for method in ['CKA', 'KNN']:
            inputs = [a.to(torch.bfloat16).requires_grad_() for a in x]
            loss = view_loss(inputs, [b.to(torch.bfloat16) for b in y], self.settings(repdist = method))
            self.assertEqual(loss.dtype, torch.float32)
            loss.backward()
            self.assertTrue(all(torch.isfinite(a.grad).all() for a in inputs))

    def test_shared_target_is_not_coordinate_feature_averaging(self):
        x, y = self.matrices()
        sign_changed = [y[0], -y[1]]
        for method in ['CKA', 'KNN']:
            args = self.settings(repdist = method)
            self.assertTrue(torch.allclose(view_loss(x, y, args), view_loss(x, sign_changed, args), atol = 1e-12))

    def test_g_partial_slurm_logs_are_included(self):
        fixture(self.root, 'two_view', complete = False)
        path = self.root / 'logs/slurm-prior-g-views-123_2.out'
        path.write_text('Array G[2]: two_view, epochs=100, action=train\nStep 5, Validation Loss 2.0, Top-1 Accuracy 40.0, Top-5 Accuracy 65.0, Samples 50000\n')
        path.with_suffix('.err').write_text('RuntimeError: synthetic interruption\n')
        rows = slurm_observations(self.root, discover(self.root, ['*'], []))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['validation'][0]['top1'], 40.0)
        self.assertIn('synthetic interruption', rows[0]['errors'][0])

    def test_g_shell_syntax_and_real_dry_run(self):
        root = Path(__file__).resolve().parent
        for script in ['prior_g_views.sbatch', 'prior_view_audit.sbatch']:
            subprocess.run(['bash', '-n', str(root / 'slurm_scripts' / script)], check = True)
            env = dict(os.environ, **self.env, DRY_RUN = '1', SLURM_ARRAY_TASK_ID = '2')
            result = subprocess.run(['bash', 'slurm_scripts/' + script], cwd = root, env = env, capture_output = True, text = True, check = True)
            self.assertIn('viewsshared', result.stdout)

if __name__ == '__main__':
    unittest.main(verbosity = 2)
