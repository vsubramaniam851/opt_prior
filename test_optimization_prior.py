import ast
import contextlib
import copy
import math
import json
import os
import sys
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from args import get_args
from run_prior import get_run_args, build_command
from rep_sims import CKA, DifferentiableKNN
from prior_utils import make_prior_model, prior_fingerprint, prior_mode, prior_schedule, prior_forward
from prior_utils import prior_state, align_prior_stage, make_prior_loaders, validate_prior_args
from prior_utils import prior_batch, prior_evaluate, prior_endpoints, prior_rank_ok, initialize_optimization_prior
from prior_utils import prior_config, check_prior_run, check_prior_checkpoint, seed_prior
from vits import ViT

def test_args(**kwargs):
    values = {'repdist': 'CKA', 'cka_loss': 'debiased', 'knn_loss': 'soft', 'knn_k': 3, 'knn_temp': 0.2,
              'prior_norm': 'batch', 'use_amp': False, 'pool_hw': 2, 'progressive_lrs': [3e-4],
              'progressive_epochs': [1], 'max_stage_steps': 12, 'rep_wd': 0.0, 'rep_scheduler': False,
              'anchor_to_init': False, 'reload_ft': False, 'rep_val_interval': 6, 'prior_schedule': 'cumulative', 'prior_min_rank_ratio': 0.0, 'align_mix': 1.0}
    values.update(kwargs)
    return SimpleNamespace(**values)

def synthetic_loaders():
    generator = torch.Generator().manual_seed(93)
    basis = torch.randn(8, 3, 32, 32, generator = generator)
    basis = torch.nn.functional.avg_pool2d(basis, 5, stride = 1, padding = 2)
    coefficients = torch.randn(64, 8, generator = generator)
    images = torch.einsum('nk,kchw->nchw', coefficients, basis)
    images += 0.02 * torch.randn(images.shape, generator = generator)
    labels = torch.arange(64) % 5
    return DataLoader(TensorDataset(images[:40], labels[:40]), batch_size = 8), DataLoader(TensorDataset(images[40:], labels[40:]), batch_size = 8)

def downstream_namespace(train_loader, val_loader, logs):
    source = Path(__file__).with_name('finetune.py').read_text()
    names = ['imagenet_finetune', 'validate', 'accuracy', 'adjust_learning_rate']
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'torch': torch, 'nn': nn, 'optim': optim, 'np': np, 'math': math, 'contextlib': contextlib,
                 'copy': copy, 'device': torch.device('cpu'), 'wandb': SimpleNamespace(log = logs.append),
                 'tqdm': lambda loader, **kwargs: loader,
                 'get_imagenet_dataloaders': lambda *args, **kwargs: (train_loader, val_loader)}
    data_source = Path(__file__).with_name('dataloader.py').read_text()
    nodes += [node for node in ast.parse(data_source).body if isinstance(node, ast.FunctionDef) and node.name == 'mixup_batch']
    exec(compile(ast.Module(body = nodes, type_ignores = []), 'finetune.py', 'exec'), namespace)
    return namespace

class DaliBatches:
    def __init__(self, images, labels, batch_size):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.size = images.size(0)

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def __iter__(self):
        for start in range(0, self.size, self.batch_size):
            yield [{'images': self.images[start:start + self.batch_size].permute(0, 2, 3, 1),
                    'labels': self.labels[start:start + self.batch_size, None]}]

class PriorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_launcher_arguments_for_all_guides(self):
        cases = [('resnet', guide, []) for guide in ['rn18', 'rn50', 'rn101', 'rn152']]
        cases += [('vit', 'vit_tiny', ['--depth', str(depth)]) for depth in [4, 8, 12, 24]]
        cases += [('vit', guide, []) for guide in ['vit_base', 'vit_large']]
        for family, guide, extra in cases:
            for method in ['baseline', 'CKA', 'KNN']:
                for phase in ['align', 'train', 'full']:
                    argv = ['run_prior.py', family, method, '--data_dir', '/a path/imagenet', '--guide', guide, '--phase', phase] + extra
                    with patch.object(sys, 'argv', argv):
                        launch = get_run_args()
                    command = build_command(launch)
                    with patch.object(sys, 'argv', command[1:]):
                        args = get_args()
                    validate_prior_args(args)
                    self.assertFalse(args.pretrained)
                    self.assertEqual(args.prior_schedule, 'cumulative')
                    self.assertEqual(args.align_mix, 1.0)
                    self.assertEqual(args.prior_min_rank_ratio, 0.0)
                    self.assertEqual(args.data_dir, '/a path/imagenet')
                    self.assertEqual(args.align_only, phase == 'align')
                    self.assertEqual(args.reload_ft, phase == 'train')
                    self.assertEqual(args.baseline, method == 'baseline')
                    self.assertEqual(args.repdist, 'CKA' if method == 'baseline' else method)

    def test_launcher_ablation_flags_and_names(self):
        names = []
        for extra in [[], ['--prior_schedule', 'frozen'], ['--align_mix', '0.5'], ['--prior_min_rank_ratio', '0.5'], ['--prior_norm', 'eval']]:
            with patch.object(sys, 'argv', ['run_prior.py', 'resnet', 'CKA', '--data_dir', '/imagenet'] + extra):
                launch = get_run_args()
            command = build_command(launch)
            with patch.object(sys, 'argv', command[1:]):
                args = get_args()
            validate_prior_args(args)
            self.assertEqual(args.prior_schedule, launch.prior_schedule)
            self.assertEqual(args.align_mix, launch.align_mix)
            self.assertEqual(args.prior_min_rank_ratio, launch.prior_min_rank_ratio)
            names.append(args.exp_name)
        self.assertEqual(len(names), len(set(names)))
        for extra in [['--guide', 'vit_tiny'], ['--depth', '12']]:
            with patch.object(sys, 'argv', ['run_prior.py', 'resnet', 'CKA', '--data_dir', '/imagenet'] + extra):
                launch = get_run_args()
            with self.assertRaises(ValueError):
                build_command(launch)

    def test_cka_gradcheck_and_invariance(self):
        x = torch.randn(8, 5, dtype = torch.float64, requires_grad = True)
        y = torch.randn(8, 9, dtype = torch.float64)
        for unbiased in [False, True]:
            metric = CKA('cpu', unbiased = unbiased)
            self.assertTrue(torch.autograd.gradcheck(lambda a: metric.linear_CKA(a, y), (x,), eps = 1e-6, atol = 1e-5))
            q = torch.linalg.qr(torch.randn(5, 5, dtype = torch.float64)).Q
            transformed = x.detach() @ q * 100 + 10
            self.assertAlmostEqual(metric.linear_CKA(x, transformed).item(), 1.0, places = 8)
            z = torch.zeros_like(x, requires_grad = True)
            self.assertTrue(torch.autograd.grad(1 - metric.linear_CKA(z, y), z)[0].isfinite().all())

    def test_cka_high_dimension_bias(self):
        torch.manual_seed(0)
        x = torch.randn(64, 8192)
        y = torch.randn(64, 8192)
        biased = CKA('cpu').linear_CKA(x, y).item()
        debiased = CKA('cpu', unbiased = True).linear_CKA(x, y).item()
        self.assertGreater(biased, 0.98)
        self.assertLess(abs(debiased), 0.1)
        print({'independent_biased_cka': biased, 'independent_debiased_cka': debiased})

    def test_knn_gradcheck_and_identity(self):
        x = torch.randn(8, 5, dtype = torch.float64, requires_grad = True)
        y = torch.randn(8, 9, dtype = torch.float64)
        metric = DifferentiableKNN('cpu', 3, temperature = 0.4)
        self.assertTrue(torch.autograd.gradcheck(lambda a: metric.diff_knn_distribution_loss(a, y), (x,), eps = 1e-6, atol = 1e-5))
        loss = metric.diff_knn_distribution_loss(x, x.detach())
        self.assertLess(abs(loss.item()), 1e-10)
        self.assertLess(torch.autograd.grad(loss, x)[0].norm().item(), 1e-8)
        metric.return_distributions = True
        loss, p, q = metric.diff_knn_distribution_loss(x, y)
        self.assertTrue((p > 0).all())
        self.assertTrue((q > 0).all())
        self.assertTrue(torch.allclose(p.sum(1), torch.ones(8, dtype = p.dtype)))
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, x)[0]).all())

    def test_knn_euclidean_and_small_batch(self):
        x = torch.randn(8, 5)
        metric = DifferentiableKNN('cpu', 3, temperature = 0.4, metric = 'euclidean')
        self.assertEqual(metric._sim(x).shape, (8, 8))
        self.assertTrue(torch.allclose(metric._sim(x), -torch.cdist(x, x).square(), atol = 1e-5))
        with self.assertRaises(ValueError):
            metric.diff_knn_distribution_loss(x[:3], x[:3])
        with self.assertRaises(ValueError):
            CKA('cpu', unbiased = True).linear_CKA(x[:3], x[:3])

    def test_guide_buffers_and_modes(self):
        model = make_prior_model('rn50', 1000)
        initial_hash = prior_fingerprint(model)
        modes = [m.training for m in model.modules()]
        with prior_mode(model), torch.no_grad():
            result = prior_forward(model, torch.randn(8, 3, 32, 32), 3)
            self.assertTrue(result.isfinite().all())
        self.assertEqual(initial_hash, prior_fingerprint(model))
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(all(m.track_running_stats for m in model.modules() if isinstance(m, nn.BatchNorm2d)))

    def test_independent_student_initialization(self):
        before = torch.get_rng_state()
        first = make_prior_model('rn18', 3)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        guide = make_prior_model('rn101', 1000)
        second = make_prior_model('rn18', 3)
        self.assertEqual(prior_fingerprint(first), prior_fingerprint(second))
        self.assertNotEqual(prior_fingerprint(first), prior_fingerprint(guide))

    def test_vit_schedule_and_depth_only(self):
        student = make_prior_model('vit_tiny', 3)
        guide = make_prior_model('vit_tiny', 1000, 12)
        self.assertEqual(student.model.hidden_dim, guide.model.hidden_dim)
        self.assertEqual([(i, j) for i, j, _ in prior_schedule(guide, student)], [(0, 2), (1, 5), (2, 8), (3, 11)])

    def test_holdout_is_disjoint_and_keeps_tail(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / 'train' / 'class0'
            folder.mkdir(parents = True)
            for i in range(41):
                Image.fromarray(np.full((8, 8, 3), i, dtype = np.uint8)).save(folder / f'{i:03d}.png')
            args = test_args(data_dir = root, rep_batch = 8, rep_val_size = 17, rep_split_seed = 0, seed = 0, num_workers = 0)
            train, val, indices = make_prior_loaders(args)
            self.assertFalse(set(train.dataset.indices).intersection(indices))
            self.assertEqual(sum(len(batch) for batch in val.batch_sampler), 17)
            self.assertEqual([len(batch) for batch in val.batch_sampler], [8, 9])
            _, _, again = make_prior_loaders(args)
            self.assertEqual(indices, again)

    def test_progressive_resnet_and_vit_both_losses(self):
        train, val = synthetic_loaders()
        for family in ['resnet', 'vit']:
            for method in ['CKA', 'KNN']:
                if family == 'resnet':
                    teacher = make_prior_model('rn50', 1000)
                    student = make_prior_model('rn18', 0)
                else:
                    torch.manual_seed(1000)
                    teacher = ViT(image_size = 32, patch_size = 8, hidden_dim = 48, num_layers = 4, num_heads = 6, mlp_dim = 192, num_classes = 5)
                    torch.manual_seed(0)
                    student = ViT(image_size = 32, patch_size = 8, hidden_dim = 24, num_layers = 2, num_heads = 3, mlp_dim = 96, num_classes = 5)
                teacher_hash = prior_fingerprint(teacher)
                summaries = []
                for stage in prior_schedule(teacher, student):
                    initial = prior_state(student)
                    ids = {id(p) for p in stage[2]}
                    frozen = [name for name, p in student.named_parameters() if id(p) not in ids]
                    result = align_prior_stage(test_args(repdist = method), teacher, student, train, val, stage, torch.device('cpu'))
                    for name in frozen:
                        self.assertTrue(torch.equal(initial[name], student.state_dict()[name]))
                    self.assertLessEqual(result['final']['loss'], result['initial']['loss'])
                    self.assertEqual(len(result['active_endpoints']), stage[0] + 1)
                    if stage[0] > 0 and result['accepted']:
                        key = 'conv1.weight' if family == 'resnet' else 'model.conv_proj.weight'
                        self.assertFalse(torch.equal(initial[key], student.state_dict()[key]))
                    self.assertEqual(result['history'][-1]['step'], 12)
                    self.assertEqual(result['final']['samples'], 24)
                    summaries.append(result)
                self.assertEqual(teacher_hash, prior_fingerprint(teacher))
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))
                self.assertTrue(any(result['accepted'] for result in summaries))
                print({'architecture': family, 'loss': method, 'stages': [dict(initial=s['initial']['loss'], final=s['final']['loss'], accepted=s['accepted']) for s in summaries]})

    def small_models(self):
        torch.manual_seed(1000)
        teacher = ViT(image_size = 32, patch_size = 8, hidden_dim = 48, num_layers = 4, num_heads = 6, mlp_dim = 192, num_classes = 5)
        torch.manual_seed(0)
        student = ViT(image_size = 32, patch_size = 8, hidden_dim = 24, num_layers = 2, num_heads = 3, mlp_dim = 96, num_classes = 5)
        return teacher, student

    def test_cumulative_graph_and_trainable_prefix(self):
        train, _ = synthetic_loaders()
        images = next(iter(train))[0]
        for method in ['CKA', 'KNN']:
            teacher, student = self.small_models()
            cumulative = prior_schedule(teacher, student)
            frozen = prior_schedule(teacher, student, 'frozen')
            first = {id(p) for p in cumulative[0][2]}
            self.assertTrue(first.issubset({id(p) for p in cumulative[1][2]}))
            self.assertFalse(first.intersection({id(p) for p in frozen[1][2]}))
            indices, targets = prior_endpoints(teacher, student, 1, 'cumulative')
            self.assertEqual(indices, [0, 1])
            self.assertEqual(targets, [1, 3])
            args = test_args(repdist = method)
            with prior_mode(teacher), prior_mode(student):
                loss, losses, _, _ = prior_batch(args, teacher, student, images, indices, targets)
                self.assertTrue(torch.allclose(loss, torch.stack(losses).mean()))
                gradient = torch.autograd.grad(losses[-1], student.model.conv_proj.weight, retain_graph = True)[0]
                self.assertGreater(gradient.norm().item(), 0.0)
                loss.backward()
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            self.assertGreater(student.model.encoder.layers[0].mlp[0].weight.grad.norm().item(), 0.0)
            self.assertTrue(all(p.grad is None for p in student.model.heads.parameters()))

    def test_multi_endpoint_forward_runs_each_block_once(self):
        teacher, student = self.small_models()
        counts = [0] * 4
        handles = []
        for index, block in enumerate(teacher.model.encoder.layers):
            def count(module, inputs, output, index = index):
                counts[index] += 1
            handles.append(block.register_forward_hook(count))
        try:
            with patch.object(teacher.model.heads, 'forward', side_effect = AssertionError('Guide head was called')):
                with prior_mode(teacher), torch.no_grad():
                    outputs = prior_forward(teacher, torch.randn(8, 3, 32, 32), [1, 3], 2)
            self.assertEqual(counts, [1, 1, 1, 1])
            self.assertEqual(len(outputs), 2)
            self.assertEqual(len(outputs[0]), 2)
        finally:
            for handle in handles:
                handle.remove()

    def test_validation_uses_cumulative_loss_and_preserves_rng(self):
        _, val = synthetic_loaders()
        teacher, student = self.small_models()
        for method in ['CKA', 'KNN']:
            args = test_args(repdist = method)
            before = torch.get_rng_state().clone()
            metrics = prior_evaluate(args, teacher, student, val, 1, 3, torch.device('cpu'))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertAlmostEqual(metrics['loss'], (metrics['endpoint1_loss'] + metrics['endpoint2_loss']) / 2, places = 6)
            self.assertEqual(metrics['samples'], 24)
            again = prior_evaluate(args, teacher, student, val, 1, 3, torch.device('cpu'))
            self.assertEqual(metrics, again)

    def test_labels_and_classifier_heads_are_unused(self):
        train, val = synthetic_loaders()
        bad_train = DataLoader(TensorDataset(train.dataset.tensors[0], torch.full((40,), float('nan'))), batch_size = 8)
        bad_val = DataLoader(TensorDataset(val.dataset.tensors[0], torch.full((24,), float('nan'))), batch_size = 8)
        for method in ['CKA', 'KNN']:
            teacher, student = self.small_models()
            guide_hash = prior_fingerprint(teacher)
            original = prior_state(student)
            results = []
            for train_loader, val_loader in [(train, val), (bad_train, bad_val)]:
                student.load_state_dict(original)
                with patch.object(teacher.model.heads, 'forward', side_effect = AssertionError('Guide head was called')):
                    with patch.object(student.model.heads, 'forward', side_effect = AssertionError('Student head was called')):
                        result = align_prior_stage(test_args(repdist = method, max_stage_steps = 2, rep_val_interval = 1), teacher, student,
                                                   train_loader, val_loader, prior_schedule(teacher, student)[1], torch.device('cpu'))
                results.append((prior_fingerprint(student), result['final']))
            self.assertEqual(results[0], results[1])
            self.assertEqual(prior_fingerprint(teacher), guide_hash)

    def test_rank_gate_is_opt_in_and_per_endpoint(self):
        initial = {'endpoint1_student_rank': 10.0, 'endpoint2_student_rank': 10.0}
        low = {'endpoint1_student_rank': 0.1, 'endpoint2_student_rank': 20.0}
        self.assertTrue(prior_rank_ok(test_args(), low, initial))
        self.assertFalse(prior_rank_ok(test_args(prior_min_rank_ratio = 0.5), low, initial))
        self.assertTrue(prior_rank_ok(test_args(prior_min_rank_ratio = 0.5), initial, initial))

    def test_selection_accepts_cumulative_tradeoff_without_rank_gate(self):
        train, val = synthetic_loaders()
        initial = {'loss': 1.0, 'endpoint1_loss': 0.5, 'endpoint2_loss': 1.5,
                   'endpoint1_student_rank': 10.0, 'endpoint2_student_rank': 10.0, 'samples': 24}
        candidate = {'loss': 0.8, 'endpoint1_loss': 0.8, 'endpoint2_loss': 0.8,
                     'endpoint1_student_rank': 1.0, 'endpoint2_student_rank': 1.0, 'samples': 24}
        for ratio in [0.0, 0.5]:
            teacher, student = self.small_models()
            start = prior_fingerprint(student)
            final = candidate if ratio == 0 else initial
            with patch('prior_utils.prior_evaluate', side_effect = [dict(initial), dict(candidate), dict(final)]):
                result = align_prior_stage(test_args(max_stage_steps = 1, prior_min_rank_ratio = ratio), teacher, student, train, val,
                                           prior_schedule(teacher, student)[1], torch.device('cpu'))
            self.assertEqual(result['accepted'], ratio == 0)
            self.assertEqual(start == prior_fingerprint(student), ratio != 0)

    def test_full_update_interpolation_and_rejection(self):
        train, val = synthetic_loaders()
        for alpha, reject in [(1.0, False), (0.5, False), (0.5, True), (0.0, True)]:
            teacher, student = self.small_models()
            initial = prior_state(student)
            states = []
            def evaluate(*args):
                states.append(prior_state(student))
                loss = 1.0 if len(states) == 1 or (len(states) == 3 and reject) else 0.5
                return {'loss': loss, 'endpoint1_student_rank': 10.0, 'samples': 24}
            with patch('prior_utils.prior_evaluate', side_effect = evaluate):
                result = align_prior_stage(test_args(max_stage_steps = 1, align_mix = alpha), teacher, student, train, val,
                                           prior_schedule(teacher, student)[0], torch.device('cpu'))
            self.assertEqual(len(states), 3)
            for name, value in student.state_dict().items():
                mixed = initial[name] + alpha * (states[1][name] - initial[name])
                self.assertTrue(torch.allclose(states[2][name], mixed, atol = 1e-7))
                self.assertTrue(torch.equal(value, initial[name] if reject else states[2][name]))
            self.assertEqual(result['accepted'], not reject)

    def test_frozen_prefix_ablation(self):
        train, val = synthetic_loaders()
        for method in ['CKA', 'KNN']:
            teacher, student = self.small_models()
            original = prior_state(student)
            stage = prior_schedule(teacher, student, 'frozen')[1]
            trainable = {id(p) for p in stage[2]}
            result = align_prior_stage(test_args(repdist = method, prior_schedule = 'frozen', max_stage_steps = 2, rep_val_interval = 1),
                                       teacher, student, train, val, stage, torch.device('cpu'))
            self.assertEqual(result['active_endpoints'], [[2, 4]])
            for name, parameter in student.named_parameters():
                if id(parameter) not in trainable:
                    self.assertTrue(torch.equal(parameter, original[name]))
                    self.assertIsNone(parameter.grad)

    def test_initializer_metadata_and_downstream_unfreezing(self):
        train, val = synthetic_loaders()
        teacher = make_prior_model('rn50', 1000)
        args = test_args(seed = 0, teacher_seed = 1000, teacher_depth = None, model = 'rn50', baseline = False,
                         replacement_type = 'progressive_align_resnet', exp_name = 'integration', max_stage_steps = 1)
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as root:
            try:
                os.chdir(root)
                with patch('prior_utils.make_prior_loaders', return_value = (train, val, list(range(40, 64)))):
                    student = initialize_optimization_prior(args, teacher, torch.device('cpu'))
                metadata = json.loads(Path('logs/integration/prior_alignment.json').read_text())
                self.assertTrue(metadata['guide_unchanged'])
                self.assertFalse(metadata['classification_labels_used'])
                self.assertFalse(metadata['rank_gate_enabled'])
                self.assertEqual(metadata['align_mix'], 1.0)
                self.assertEqual(metadata['schedule'], 'cumulative')
                self.assertEqual(len(metadata['stages']), 4)
                self.assertTrue(all(p.requires_grad and p.grad is None for p in student.parameters()))
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))
                saved = torch.load('saved_models/integration/prior_stage4.pt', weights_only = True)
                self.assertTrue(all(torch.equal(value, saved[name]) for name, value in student.state_dict().items()))
            finally:
                os.chdir(previous)

    def test_single_batch_holdout_and_training_split(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / 'train' / 'class0'
            folder.mkdir(parents = True)
            for i in range(16):
                Image.fromarray(np.full((8, 8, 3), i, dtype = np.uint8)).save(folder / f'{i:03d}.png')
            args = test_args(data_dir = root, rep_batch = 8, rep_val_size = 8, rep_split_seed = 0, seed = 0, num_workers = 0)
            train, val, indices = make_prior_loaders(args)
            self.assertEqual(len(train), 1)
            self.assertEqual(len(val), 1)
            self.assertEqual(sum(len(batch) for batch in val.batch_sampler), 8)
            before = torch.get_rng_state().clone()
            next(iter(val))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_checkpoint_configuration_and_overwrite_guards(self):
        args = test_args(optimization_prior = True, exp_name = 'reload_test', data_dir = 'unused')
        teacher, student = self.small_models()
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as root:
            try:
                os.chdir(root)
                check_prior_run(args)
                Path('logs/reload_test').mkdir(parents = True)
                Path('saved_models').mkdir()
                metadata = {'version': 2, 'complete': True, 'alignment_config': prior_config(args),
                            'student_aligned_hash': prior_fingerprint(student)}
                Path('logs/reload_test/prior_alignment.json').write_text(json.dumps(metadata))
                torch.save(student.state_dict(), 'saved_models/orig_reload_test.pt')
                with self.assertRaises(FileExistsError):
                    check_prior_run(args)
                args.reload_ft = True
                check_prior_run(args)
                check_prior_checkpoint(args, student)
                args.align_mix = 0.5
                with self.assertRaisesRegex(ValueError, 'align_mix'):
                    check_prior_run(args)
                args.align_mix = 1.0
                with torch.no_grad():
                    student.model.conv_proj.weight.add_(0.1)
                with self.assertRaises(ValueError):
                    check_prior_checkpoint(args, student)
                Path('saved_models/ft_final_reload_test.pt').touch()
                with self.assertRaises(FileExistsError):
                    check_prior_run(args)
            finally:
                os.chdir(previous)

    def test_baseline_checkpoint_reloading_does_not_rewrite_metadata(self):
        teacher = make_prior_model('rn18', 1000)
        args = test_args(seed = 0, teacher_seed = 1000, teacher_depth = None, model = 'rn18', baseline = True,
                         replacement_type = 'progressive_align_resnet', exp_name = 'baseline')
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as root:
            try:
                os.chdir(root)
                student = initialize_optimization_prior(args, teacher, torch.device('cpu'), initialize_only = True)
                path = Path('logs/baseline/prior_alignment.json')
                metadata = path.read_text()
                self.assertTrue(json.loads(metadata)['complete'])
                self.assertEqual(json.loads(metadata)['student_initial_hash'], prior_fingerprint(student))
                args.reload_ft = True
                initialize_optimization_prior(args, teacher, torch.device('cpu'), initialize_only = True)
                self.assertEqual(path.read_text(), metadata)
            finally:
                os.chdir(previous)

    def test_dali_reader_and_iterator_configuration_contract(self):
        source = Path(__file__).with_name('dataloader.py').read_text()
        names = ['create_imagenet_dataloader', 'get_imagenet_dataloaders']
        nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
        calls = []
        node = SimpleNamespace()
        node.gpu = lambda: node
        def read_files(**kwargs):
            calls.append(kwargs)
            return node, node
        class FakePipeline:
            def __init__(self, **kwargs):
                self.arguments = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def set_outputs(self, *args):
                return None

            def build(self):
                return None

        fn = SimpleNamespace(readers = SimpleNamespace(file = read_files),
                             decoders = SimpleNamespace(image = lambda *args, **kwargs: node, image_random_crop = lambda *args, **kwargs: node),
                             resize = lambda *args, **kwargs: node, crop_mirror_normalize = lambda *args, **kwargs: node,
                             random = SimpleNamespace(coin_flip = lambda **kwargs: node))
        namespace = {'os': os, 'Pipeline': FakePipeline, 'fn': fn, 'types': SimpleNamespace(RGB = 'rgb', FLOAT = 'float', NHWC = 'nhwc'),
                     'DALIGenericIterator': lambda pipes, **kwargs: kwargs, 'LastBatchPolicy': SimpleNamespace(DROP = 'drop', PARTIAL = 'partial'),
                     'IMAGENET_MEAN': [0, 0, 0], 'IMAGENET_STD': [1, 1, 1]}
        exec(compile(ast.Module(body = nodes, type_ignores = []), 'dataloader.py', 'exec'), namespace)
        train, val = namespace['get_imagenet_dataloaders'](data_dir = '/data/imagenet', seed = 20000)
        self.assertEqual(train['last_batch_policy'], 'drop')
        self.assertEqual(val['last_batch_policy'], 'partial')
        self.assertTrue(train['auto_reset'] and val['auto_reset'])
        self.assertTrue(all(call['pad_last_batch'] for call in calls))
        self.assertEqual([call['random_shuffle'] for call in calls], [True, False])
        self.assertEqual([call['file_root'] for call in calls], ['/data/imagenet/train', '/data/imagenet/val'])
        self.assertEqual(train['reader_name'], 'Reader')
        self.assertEqual(val['reader_name'], 'Reader')

    def test_downstream_adamw_mixup_and_cosine_schedule(self):
        train = DaliBatches(torch.randn(10, 3, 4, 4), torch.arange(10) % 5, 2)
        val = DaliBatches(torch.randn(5, 3, 4, 4), torch.arange(5), 2)
        logs = []
        namespace = downstream_namespace(train, val, logs)
        args = test_args(optimization_prior = True, imagenet = 'imagenet-1k', data_dir = 'unused', seed = 0,
                         ft_batch = 2, measure_alignment = False, tune_linear = False, fc_only = False,
                         ft_optimizer = 'adamw', ft_lr = 5e-4, ft_wd = 0.05, label_smoothing = 0.1, ft_accum = 2,
                         use_scheduler = True, ft_epochs = 2, evaluate = 'epoch', scheduler_type = 'cosine', warmup_epochs = 0,
                         mixup_alpha = 0.8, preserve_alignment = False, compute_norms = False, ft_clip = 1.0, exp_name = 'adamw')
        model = nn.Sequential(nn.Flatten(), nn.Linear(48, 5))
        model.requires_grad_(False)
        initial = prior_fingerprint(model)
        rates = []
        original_step = optim.AdamW.step
        def step(optimizer, *args, **kwargs):
            rates.append(optimizer.param_groups[0]['lr'])
            return original_step(optimizer, *args, **kwargs)
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as root:
            try:
                os.chdir(root)
                Path('saved_models').mkdir()
                with patch.object(optim.AdamW, 'step', step):
                    model, _, result, _ = namespace['imagenet_finetune'](args, model, None)
                self.assertEqual(len(rates), 6)
                self.assertAlmostEqual(rates[0], args.ft_lr)
                self.assertTrue(all(a > b for a, b in zip(rates, rates[1:])))
                self.assertNotEqual(initial, prior_fingerprint(model))
                self.assertEqual(len(result['top1_accs']), 2)
                self.assertEqual(result['final_top1'], result['top1_accs'][-1])
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                self.assertEqual([log['val_samples'] for log in logs if 'val_samples' in log], [5, 5])
            finally:
                os.chdir(previous)

    def test_full_and_split_orchestration_match(self):
        train_rep, val_rep = synthetic_loaders()
        images = train_rep.dataset.tensors[0][:10]
        labels = train_rep.dataset.tensors[1][:10]
        train_ft = DaliBatches(images, labels, 2)
        val_ft = DaliBatches(val_rep.dataset.tensors[0][:5], val_rep.dataset.tensors[1][:5], 2)
        logs = []
        namespace = downstream_namespace(train_ft, val_ft, logs)
        namespace.update({'os': os, 'json': json, 'seed_prior': seed_prior, 'validate_prior_args': validate_prior_args,
                          'make_prior_model': make_prior_model, 'check_prior_run': check_prior_run, 'check_prior_checkpoint': check_prior_checkpoint,
                          'initialize_optimization_prior': initialize_optimization_prior,
                          'wandb': SimpleNamespace(log = logs.append, init = lambda **kwargs: None, finish = lambda: None)})
        for filename, name in [('layer_replace.py', 'layer_replace'), ('theseus.py', 'theseus')]:
            source = Path(__file__).with_name(filename).read_text()
            nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
            exec(compile(ast.Module(body = nodes, type_ignores = []), filename, 'exec'), namespace)
        with patch.object(sys, 'argv', ['run_prior.py', 'resnet', 'CKA', '--data_dir', 'unused']):
            command = build_command(get_run_args())
        with patch.object(sys, 'argv', command[1:]):
            args = get_args()
        args.rep_batch = 8
        args.knn_k = 3
        args.rep_val_size = 24
        args.max_stage_steps = 1
        args.ft_batch = 2
        args.ft_accum = 2
        args.ft_epochs = 1
        args.warmup_epochs = 0
        args.num_workers = 0
        namespace['get_args'] = lambda: args
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as root:
            try:
                os.chdir(root)
                with patch('prior_utils.make_prior_loaders', return_value = (train_rep, val_rep, list(range(40, 64)))):
                    for baseline in [False, True]:
                        args.baseline = baseline
                        args.exp_name = f'full_{baseline}'
                        args.align_only = False
                        args.reload_ft = False
                        namespace['theseus']()
                        complete = torch.load(f'saved_models/ft_final_{args.exp_name}.pt', weights_only = True)
                        args.exp_name = f'split_{baseline}'
                        args.align_only = True
                        namespace['theseus']()
                        path = Path(f'logs/{args.exp_name}/args.json')
                        original_args = path.read_text()
                        args.align_only = False
                        args.reload_ft = True
                        namespace['theseus']()
                        split = torch.load(f'saved_models/ft_final_{args.exp_name}.pt', weights_only = True)
                        self.assertTrue(all(torch.equal(value, split[name]) for name, value in complete.items()))
                        self.assertEqual(path.read_text(), original_args)
                        self.assertTrue(Path(f'logs/{args.exp_name}/training_args.json').exists())
            finally:
                os.chdir(previous)

    def test_downstream_accumulation_validation_and_final_checkpoint(self):
        train = DaliBatches(torch.randn(10, 3, 4, 4), torch.arange(10) % 5, 2)
        val = DaliBatches(torch.randn(5, 3, 4, 4), torch.arange(5), 2)
        logs = []
        namespace = downstream_namespace(train, val, logs)
        args = test_args(optimization_prior = True, imagenet = 'imagenet-1k', data_dir = 'unused', seed = 0,
                         ft_batch = 2, measure_alignment = False, tune_linear = False, fc_only = False,
                         ft_optimizer = 'sgd', ft_lr = 0.01, ft_wd = 0.0, label_smoothing = 0.2, ft_accum = 2,
                         use_scheduler = False, ft_epochs = 1, evaluate = 'epoch', scheduler_type = 'cosine',
                         mixup_alpha = 0.0, preserve_alignment = False, compute_norms = False, ft_clip = 0.0, exp_name = 'test')
        model = nn.Sequential(nn.Flatten(), nn.Linear(48, 5))
        reference = copy.deepcopy(model)
        for p in model.parameters():
            p.grad = torch.ones_like(p) * 100
        optimizer = optim.SGD(reference.parameters(), lr = args.ft_lr, momentum = 0.9, nesterov = True)
        for start in range(0, train.size, 4):
            optimizer.zero_grad(set_to_none = True)
            loss = nn.CrossEntropyLoss(label_smoothing = 0.2)(reference(train.images[start:start + 4]), train.labels[start:start + 4])
            loss.backward()
            optimizer.step()
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as root:
            try:
                os.chdir(root)
                Path('saved_models').mkdir()
                model, _, results, _ = namespace['imagenet_finetune'](args, model, None)
                self.assertTrue(Path('saved_models/ft_final_test.pt').exists())
                checkpoint = torch.load('saved_models/ft_final_test.pt', weights_only = True)
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, checkpoint[name]))
                    self.assertTrue(torch.allclose(value, reference.state_dict()[name], atol = 1e-6))
                self.assertEqual(len(results['top1_accs']), 1)
                self.assertEqual(next(log['val_samples'] for log in logs if 'val_samples' in log), 5)
            finally:
                os.chdir(previous)

if __name__ == '__main__':
    unittest.main(verbosity = 2)
