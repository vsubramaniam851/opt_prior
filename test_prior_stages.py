import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import torch

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from torch.utils.data import DataLoader, TensorDataset
from args import get_args
from compare_prior import discover, match_baselines, short_label, slurm_observations
from prior_utils import align_prior_stage, check_prior_run, initialize_optimization_prior, make_prior_model
from prior_utils import prior_batch, prior_config, prior_endpoints, prior_fingerprint, prior_mode, prior_parts
from prior_utils import prior_schedule, prior_segment_changes, prior_state, validate_prior_args
from run_prior import build_command, get_run_args
from slurm_scripts.run_prior_array import execute, make_run, parse_with, preflight
from slurm_scripts.run_prior_audit import main as audit_main
from slurm_scripts.test_prior_array import write_initialization
from test_compare_prior import fixture
from test_optimization_prior import test_args
from vits import ViT

class StageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(71)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def models(self, family = 'vit'):
        if family == 'resnet':
            return make_prior_model('rn50', 1000), make_prior_model('rn18', 0)
        return (ViT(image_size = 32, patch_size = 8, hidden_dim = 24, num_layers = 12, num_heads = 3, mlp_dim = 48),
                ViT(image_size = 32, patch_size = 8, hidden_dim = 16, num_layers = 4, num_heads = 2, mlp_dim = 32))

    def launch(self, *extra):
        args = parse_with(get_run_args, ['resnet', 'CKA', '--data_dir', str(self.root), *extra])
        return args, parse_with(get_args, build_command(args)[2:])

    def test_default_schedule_parameter_lists_match_legacy(self):
        for family in ['resnet', 'vit']:
            teacher, student = self.models(family)
            stem, blocks, globals_ = prior_parts(student)
            for schedule in ['cumulative', 'frozen']:
                stages = prior_schedule(teacher, student, schedule)
                self.assertEqual(len(stages), 4)
                for index, target, params in stages:
                    modules = stem + blocks[:index + 1] if schedule == 'cumulative' else (stem + [blocks[index]] if index == 0 else [blocks[index]])
                    old = [p for module in modules for p in module.parameters()]
                    if schedule == 'cumulative' or index == 0:
                        old += globals_
                    self.assertEqual([id(p) for p in old], [id(p) for p in params])

    def test_targets_are_actual_depths_not_reindexed(self):
        for family, expected in [('resnet', [(2, 2), (3, 3)]), ('vit', [(2, 8), (3, 11)])]:
            teacher, student = self.models(family)
            for scope in ['prefix', 'suffix']:
                stages = prior_schedule(teacher, student, 'cumulative', 3, scope)
                self.assertEqual([(i, j) for i, j, _ in stages], expected)
                self.assertEqual(prior_endpoints(teacher, student, 2, 'cumulative', 3), ([2], [expected[0][1]]))
                self.assertEqual(prior_endpoints(teacher, student, 3, 'cumulative', 3), ([2, 3], [x[1] for x in expected]))

    def test_suffix_masks_and_prefix_masks(self):
        for family in ['resnet', 'vit']:
            teacher, student = self.models(family)
            stem, blocks, globals_ = prior_parts(student)
            for scope in ['prefix', 'suffix']:
                for index, target, params in prior_schedule(teacher, student, 'cumulative', 3, scope):
                    modules = stem + blocks[:index + 1] if scope == 'prefix' else blocks[2:index + 1]
                    expected = [p for module in modules for p in module.parameters()] + (globals_ if scope == 'prefix' else [])
                    self.assertEqual({id(p) for p in params}, {id(p) for p in expected})
                    self.assertEqual(len(params), len({id(p) for p in params}))

    def test_final_only_and_frozen_start(self):
        teacher, student = self.models()
        stages = prior_schedule(teacher, student, 'cumulative', 4, 'suffix')
        self.assertEqual([(x[0], x[1]) for x in stages], [(3, 11)])
        self.assertEqual(prior_endpoints(teacher, student, 3, 'cumulative', 4), ([3], [11]))
        stages = prior_schedule(teacher, student, 'frozen', 3)
        self.assertEqual(len(stages), 2)
        _, blocks, _ = prior_parts(student)
        for index, _, params in stages:
            self.assertEqual([id(p) for p in params], [id(p) for p in blocks[index].parameters()])

    def test_invalid_start_and_scope_rejected(self):
        teacher, student = self.models()
        for start in [0, -1, 5, True, 2.5]:
            with self.assertRaises(ValueError):
                prior_schedule(teacher, student, start_stage = start)
        with self.assertRaises(ValueError):
            prior_schedule(teacher, student, update_scope = 'all')
        with self.assertRaises(ValueError):
            prior_schedule(teacher, student, 'frozen', 3, 'suffix')
        for start in ['0', '5']:
            args = parse_with(get_run_args, ['resnet', 'CKA', '--data_dir', str(self.root), '--prior_start_stage', start])
            with self.assertRaises(ValueError):
                build_command(args)

    def test_launcher_flags_names_and_training_recipe(self):
        _, old = self.launch()
        names = set()
        for family in ['resnet', 'vit']:
            for method in ['CKA', 'KNN']:
                for scope in ['prefix', 'suffix']:
                    for phase in ['align', 'train', 'full']:
                        argv = [family, method, '--data_dir', str(self.root), '--prior_start_stage', '3',
                                '--prior_update_scope', scope, '--prior_descriptor', 'spatial', '--phase', phase, '--steps', '2000']
                        launch = parse_with(get_run_args, argv)
                        command = build_command(launch)
                        args = parse_with(get_args, command[2:])
                        validate_prior_args(args)
                        self.assertEqual(args.prior_start_stage, 3)
                        self.assertEqual(args.prior_update_scope, scope)
                        self.assertEqual(args.max_stage_steps, 2000)
                        self.assertEqual(args.align_only, phase == 'align')
                        self.assertEqual(args.reload_ft, phase == 'train')
                        self.assertIn('_from3', args.exp_name)
                        self.assertEqual(args.exp_name.endswith('_suffix'), scope == 'suffix')
                        names.add(args.exp_name)
                        if family == 'resnet':
                            for key in ['ft_batch', 'ft_epochs', 'ft_lr', 'ft_wd', 'ft_optimizer', 'ft_accum', 'label_smoothing', 'mixup_alpha']:
                                self.assertEqual(getattr(args, key), getattr(old, key))
        self.assertEqual(len(names), 8)

    def test_baseline_and_frozen_invalid_scope_rejected(self):
        for argv in [['resnet', 'baseline', '--prior_start_stage', '3'],
                     ['resnet', 'CKA', '--prior_schedule', 'frozen', '--prior_update_scope', 'suffix']]:
            args = parse_with(get_run_args, argv + ['--data_dir', str(self.root)])
            with self.assertRaises(ValueError):
                build_command(args)
        _, args = self.launch('--prior_start_stage', '3')
        args.prior_start_stage = 5
        with self.assertRaises(ValueError):
            validate_prior_args(args)

    def test_default_config_unchanged_and_optional_fields_recorded(self):
        args = test_args()
        old = prior_config(args)
        args.prior_start_stage = 1
        args.prior_update_scope = 'prefix'
        self.assertEqual(prior_config(args), old)
        args.prior_start_stage = 3
        args.prior_update_scope = 'suffix'
        new = prior_config(args)
        self.assertEqual(new['prior_start_stage'], 3)
        self.assertEqual(new['prior_update_scope'], 'suffix')
        for key, value in old.items():
            self.assertEqual(new[key], value)

    def test_reload_rejects_dropped_stage_scope_and_preserves_old_defaults(self):
        _, args = self.launch('--phase', 'train')
        (self.root / 'saved_models').mkdir()
        folder = self.root / 'logs' / args.exp_name
        folder.mkdir(parents = True)
        (self.root / 'saved_models' / ('orig_' + args.exp_name + '.pt')).write_bytes(b'test')
        path = folder / 'prior_alignment.json'
        path.write_text(json.dumps({'version': 2, 'complete': True, 'alignment_config': prior_config(args)}))
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            check_prior_run(args)
            args.prior_start_stage = 3
            with self.assertRaisesRegex(ValueError, 'prior_start_stage'):
                check_prior_run(args)
            args.prior_update_scope = 'suffix'
            path.write_text(json.dumps({'version': 2, 'complete': True, 'alignment_config': prior_config(args)}))
            check_prior_run(args)
            args.prior_update_scope = 'prefix'
            with self.assertRaisesRegex(ValueError, 'prior_update_scope'):
                check_prior_run(args)
        finally:
            os.chdir(previous)

    def test_live_resnet_gradient_update_masks_for_both_losses(self):
        for method in ['CKA', 'KNN']:
            for scope in ['prefix', 'suffix']:
                torch.manual_seed(71)
                teacher, student = self.models('resnet')
                teacher.requires_grad_(False)
                before_guide = prior_fingerprint(teacher)
                initial = prior_state(student)
                teacher.fc.forward = lambda *args: self.fail('Guide head was evaluated')
                student.fc.forward = lambda *args: self.fail('Student head was evaluated')
                for index, target, params in prior_schedule(teacher, student, 'cumulative', 3, scope):
                    student.requires_grad_(False)
                    student.zero_grad(set_to_none = True)
                    for p in params:
                        p.requires_grad_(True)
                    args = test_args(repdist = method, prior_descriptor = 'spatial', prior_start_stage = 3, prior_update_scope = scope)
                    a, b = prior_endpoints(teacher, student, index, 'cumulative', 3)
                    images = torch.randn(8, 3, 64, 64)
                    with prior_mode(teacher), prior_mode(student):
                        loss, losses, _, _ = prior_batch(args, teacher, student, images, a, b)
                        self.assertTrue(torch.equal(loss, torch.stack(losses).mean()))
                        loss.backward()
                        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in params))
                        optimizer = torch.optim.SGD(params, lr = 1e-3)
                        optimizer.step()
                    changes = prior_segment_changes(student, initial)
                    if scope == 'suffix':
                        for key in ['stem', 'stage1', 'stage2']:
                            self.assertFalse(changes[key]['changed'])
                    else:
                        self.assertTrue(changes['stem']['changed'])
                    self.assertTrue(changes['stage3']['changed'])
                self.assertTrue(changes['stage4']['changed'])
                self.assertEqual(prior_fingerprint(teacher), before_guide)
                for name, value in student.state_dict().items():
                    if 'running_' in name or 'num_batches_tracked' in name or name.startswith('fc.'):
                        self.assertTrue(torch.equal(value, initial[name]), name)
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_upper_initialization_orchestration_and_metadata(self):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            for method in ['CKA', 'KNN']:
                for scope in ['prefix', 'suffix']:
                    teacher, student = self.models()
                    initial = prior_state(student)
                    initial_hash = prior_fingerprint(student)
                    guide_hash = prior_fingerprint(teacher)
                    student.model.heads.forward = lambda *args: self.fail('Student head was evaluated')
                    teacher.model.heads.forward = lambda *args: self.fail('Guide head was evaluated')
                    x = torch.randn(32, 3, 32, 32)
                    labels = torch.full((32,), float('nan'))
                    train = DataLoader(TensorDataset(x[:24], labels[:24]), batch_size = 8)
                    val = DataLoader(TensorDataset(x[24:], labels[24:]), batch_size = 8)
                    args = test_args(repdist = method, prior_descriptor = 'spatial', prior_start_stage = 3, prior_update_scope = scope,
                                     max_stage_steps = 4, rep_val_interval = 2, exp_name = f'{method}_{scope}', replacement_type = 'progressive_align_vits',
                                     student_vit = 'vit_tiny', seed = 0, teacher_seed = 1000, model = 'vit_tiny', teacher_depth = 12, baseline = False)
                    with patch('prior_utils.make_prior_model', return_value = student), patch('prior_utils.make_prior_loaders', return_value = (train, val, list(range(24, 32)))):
                        returned = initialize_optimization_prior(args, teacher, 'cpu')
                    metadata = json.loads((self.root / 'logs' / args.exp_name / 'prior_alignment.json').read_text())
                    self.assertEqual(list(metadata['stages']), ['stage3', 'stage4'])
                    self.assertTrue(metadata['complete'])
                    self.assertFalse(metadata['classification_labels_used'])
                    self.assertTrue(metadata['guide_unchanged'])
                    self.assertEqual(metadata['guide_initial_hash'], guide_hash)
                    self.assertEqual(metadata['student_initial_hash'], initial_hash)
                    self.assertTrue(all(p.requires_grad and p.grad is None for p in returned.parameters()))
                    for name, record in metadata['stages'].items():
                        self.assertEqual(record['steps'], 4)
                        self.assertEqual(record['prior_start_stage'], 3)
                        self.assertEqual(record['prior_update_scope'], scope)
                        self.assertIn('parameter_changes_by_segment', record)
                        self.assertEqual(record['active_endpoints'], [[3, 9]] if name == 'stage3' else [[3, 9], [4, 12]])
                        if scope == 'suffix':
                            self.assertTrue(record['lower_prefix_unchanged'])
                    if scope == 'suffix':
                        changes = prior_segment_changes(returned, initial)
                        for key in ['stem', 'stage1', 'stage2']:
                            self.assertFalse(changes[key]['changed'])
                    print({'synthetic_upper_alignment': args.exp_name, 'accepted': [r['accepted'] for r in metadata['stages'].values()],
                           'initial_loss': [r['initial']['loss'] for r in metadata['stages'].values()],
                           'selected_loss': [r['final']['loss'] for r in metadata['stages'].values()]})
        finally:
            os.chdir(previous)

    def test_e_array_names_budget_and_phase_routes(self):
        names = set()
        for phase in ['align', 'train', 'full']:
            for steps in ['1000', '2000']:
                for i in range(4):
                    run = make_run('E', i, {'DATA_DIR': str(self.root), 'PRIOR_PHASE': phase, 'ALIGN_STEPS': steps})
                    self.assertEqual(run['phase'], phase)
                    self.assertEqual(run['method'], 'CKA' if i % 2 == 0 else 'KNN')
                    self.assertEqual(run['training']['prior_start_stage'], 3)
                    self.assertEqual(run['training']['prior_update_scope'], 'prefix' if i < 2 else 'suffix')
                    self.assertEqual(run['training']['max_stage_steps'], int(steps))
                    self.assertTrue(run['name'].endswith('_steps' + steps))
                    self.assertEqual(run['training']['ft_epochs'], 100)
                    names.add(run['name'])
        self.assertEqual(len(names), 8)

    def test_e_checkpoint_validation_no_silent_policy_reuse(self):
        for split in ['train', 'val']:
            (self.root / split).mkdir()
        run = make_run('E', 2, {'DATA_DIR': str(self.root), 'PRIOR_PHASE': 'train'})
        write_initialization(self.root, run)
        folder = self.root / 'logs' / run['name']
        path = folder / 'prior_alignment.json'
        metadata = json.loads(path.read_text())
        with self.assertRaisesRegex(ValueError, 'prior_start_stage'):
            preflight(self.root, run)
        metadata['alignment_config'] = prior_config(SimpleNamespace(**run['training']))
        path.write_text(json.dumps(metadata))
        self.assertEqual(preflight(self.root, run), 'train')
        metadata['alignment_config'].pop('prior_update_scope')
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'prior_update_scope'):
            preflight(self.root, run)

    def test_e_alignment_then_training_uses_saved_initialization(self):
        for split in ['train', 'val']:
            (self.root / split).mkdir()
        env = {'DATA_DIR': str(self.root), 'USE_CURRENT_ENV': '1', 'SLURM_JOB_ID': 'mock_upper'}
        for index in range(4):
            for phase in ['align', 'train']:
                run = make_run('E', index, dict(env, PRIOR_PHASE = phase))
                def complete(*args, **kwargs):
                    self.assertIn('run_prior.py', args[0])
                    self.assertEqual(args[0][-len(run['argv']):], run['argv'])
                    folder = self.root / 'logs' / run['name']
                    if phase == 'align':
                        write_initialization(self.root, run)
                        path = folder / 'prior_alignment.json'
                        metadata = json.loads(path.read_text())
                        metadata['alignment_config'] = prior_config(SimpleNamespace(**run['training']))
                        path.write_text(json.dumps(metadata))
                    else:
                        self.assertTrue((self.root / 'saved_models' / ('orig_' + run['name'] + '.pt')).is_file())
                        (folder / 'training_args.json').write_text(json.dumps(run['training']))
                        (folder / 'ft_results_seed0.json').write_text(json.dumps({'final_top1': 60.0}))
                        (self.root / 'saved_models' / ('ft_final_' + run['name'] + '.pt')).write_bytes(b'synthetic-test-only')
                    return SimpleNamespace(returncode = 0)
                with patch('slurm_scripts.run_prior_array.subprocess.run', side_effect = complete), \
                     patch('slurm_scripts.run_prior_array.shutil.which', return_value = '/mock/srun'), contextlib.redirect_stdout(io.StringIO()):
                    execute(self.root, run, env)
                self.assertEqual(preflight(self.root, run), 'complete_alignment' if phase == 'align' else 'complete')

    def test_upper_audit_routing_and_names(self):
        for index in range(4):
            for current in ['0', '1']:
                env = {'DATA_DIR': str(self.root), 'AUDIT_GROUP': 'upper', 'SLURM_ARRAY_TASK_ID': str(index),
                       'DRY_RUN': '1', 'USE_CURRENT_ENV': current, 'ALIGN_STEPS': '2000'}
                output = io.StringIO()
                with patch.dict(os.environ, env, clear = True), contextlib.redirect_stdout(output):
                    audit_main()
                self.assertIn('audit_prior_geometry.py', output.getvalue())
                self.assertNotIn('-u run_prior.py', output.getvalue())
                self.assertIn(make_run('E', index, env)['name'], output.getvalue())

    def test_comparisons_split_stage_policies_and_reuse_baseline(self):
        fixture(self.root, 'baseline', method = 'baseline')
        for name, start, scope in [('all_stages', 1, 'prefix'), ('top_prefix', 3, 'prefix'), ('top_suffix', 3, 'suffix')]:
            folder = fixture(self.root, name)
            path = folder / 'args.json'
            args = json.loads(path.read_text())
            args.update({'prior_start_stage': start, 'prior_update_scope': scope})
            path.write_text(json.dumps(args))
            path = folder / 'prior_alignment.json'
            meta = json.loads(path.read_text())
            if start != 1:
                meta['alignment_config']['prior_start_stage'] = start
            if scope != 'prefix':
                meta['alignment_config']['prior_update_scope'] = scope
            path.write_text(json.dumps(meta))
        runs = discover(self.root, ['*'], [])
        match_baselines(runs)
        aligned = [r for r in runs if r['method'] != 'baseline']
        self.assertEqual(len({r['condition_id'] for r in aligned}), 3)
        self.assertTrue(all(r['pair_status'] == 'matched' for r in aligned))
        self.assertTrue(all(r['paired_baseline'] == 'baseline' for r in aligned))

    def test_omitted_stage_policy_is_blocked_in_comparison(self):
        folder = fixture(self.root, 'bad_policy')
        path = folder / 'prior_alignment.json'
        meta = json.loads(path.read_text())
        meta['alignment_config']['prior_start_stage'] = 3
        path.write_text(json.dumps(meta))
        run = discover(self.root, ['*'], [])[0]
        self.assertFalse(run['eligible'])
        self.assertTrue(any(x['code'] == 'alignment_config_mismatch' for x in run['issues']))

    def test_e_logs_remain_visible_to_comparison(self):
        folder = fixture(self.root, 'upper_partial', complete = False)
        log = self.root / 'logs' / 'slurm-prior-e-upper-123_2.out'
        log.write_text('Array E[2]: upper_partial, epochs=100, action=full\nStep 0, Validation Loss 4.5, Top-1 Accuracy 18.2, Top-5 Accuracy 39.1, Samples 50000\n')
        log.with_suffix('.err').write_text('RuntimeError: synthetic test failure\n')
        runs = discover(self.root, ['*'], [])
        observations = slurm_observations(self.root, runs)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]['experiment'], folder.name)
        self.assertEqual(observations[0]['validation'][0]['top1'], 18.2)
        self.assertIn('synthetic test failure', observations[0]['errors'][0])
        self.assertIsNone(runs[0]['final_top1'])

    def test_plot_labels_distinguish_descriptors_and_stage_policies(self):
        fixture(self.root, 'aligned')
        run = discover(self.root, ['*'], [])[0]
        labels = set()
        for descriptor, start, scope in [('pooled', 1, 'prefix'), ('spatial', 1, 'prefix'),
                                          ('spatial', 3, 'prefix'), ('spatial', 3, 'suffix')]:
            changed = copy.deepcopy(run)
            changed['alignment_controls'].update({'prior_descriptor': descriptor, 'prior_start_stage': start, 'prior_update_scope': scope})
            labels.add(short_label(changed))
        self.assertEqual(len(labels), 4)

    def test_new_sbatch_syntax(self):
        path = Path(__file__).parent / 'slurm_scripts' / 'prior_e_upper.sbatch'
        subprocess.run(['bash', '-n', str(path)], check = True)
        text = path.read_text()
        self.assertIn('#SBATCH --array=0-3%2', text)
        self.assertIn('run_prior_array E', text)

if __name__ == '__main__':
    unittest.main(verbosity = 2)
