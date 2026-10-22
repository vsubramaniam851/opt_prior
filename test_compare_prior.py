import ast
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import tokenize
import unittest
import zipfile

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from args import get_args as get_training_args
from run_prior import build_command, get_run_args
from compare_prior import Figure, analyze, compact_result, condition_summary, depth_comparisons, discover
from compare_prior import get_args, load_run, match_baselines, slurm_observations, stage_tables, write_json

ROOT = Path(__file__).resolve().parent

def parse(parser, argv):
    with patch.object(sys, 'argv', ['test'] + argv):
        return parser()

def real_config(family, method, seed, guide, depth, epochs):
    argv = [family, method, '--data_dir', '/synthetic/imagenet', '--seed', str(seed), '--guide', guide]
    if depth is not None:
        argv += ['--depth', str(depth)]
    if epochs is not None:
        argv += ['--epochs', str(epochs)]
    command = build_command(parse(get_run_args, argv))
    return vars(parse(get_training_args, command[2:]))

def real_alignment_config(config):
    tree = ast.parse((ROOT / 'prior_utils.py').read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'prior_config')
    namespace = {'os': os}
    exec(compile(ast.Module(body = [node], type_ignores = []), 'prior_config', 'exec'), namespace)
    return namespace['prior_config'](SimpleNamespace(**config))

def metrics(loss, phase):
    result = {'loss': loss, 'cka': 0.8, 'ucka': 0.4, 'knn_overlap': 0.5, 'student_rank': 40.0,
              'teacher_rank': 35.0, 'student_centered_rms': 0.5, 'teacher_centered_rms': 0.6, 'samples': 4096}
    for endpoint in range(1, phase + 1):
        for key, value in list(result.items()):
            if not key.startswith('endpoint') and key != 'samples':
                result[f'endpoint{endpoint}_{key}'] = value
    return result

def fixture(root, name, method = 'CKA', seed = 0, family = 'resnet', guide = None, depth = None, epochs = None,
            top1 = 62.0, guide_seed = 1000, complete = True):
    guide = guide or ('rn50' if family == 'resnet' else 'vit_tiny')
    config = real_config(family, method, seed, guide, depth, epochs)
    config.update({'exp_name': name, 'teacher_seed': guide_seed})
    student = 'rn18' if family == 'resnet' else 'vit_tiny'
    initial_hash = student + '_s' + str(seed)
    baseline = method == 'baseline'
    metadata = {'version': 2, 'student': student, 'guide': guide, 'student_seed': seed, 'guide_seed': guide_seed,
                'guide_depth_override': config.get('teacher_depth'), 'student_initial_hash': initial_hash,
                'student_aligned_hash': initial_hash if baseline else initial_hash + '_aligned',
                'guide_initial_hash': 'random_guide', 'guide_final_hash': 'random_guide', 'guide_unchanged': True,
                'complete': True, 'baseline': baseline, 'classification_labels_used': False, 'stages': {},
                'schedule': 'cumulative', 'align_mix': 1.0, 'rank_gate_enabled': False,
                'alignment_config': real_alignment_config(config)}
    if not baseline:
        metadata['validation_indices'] = list(range(4096))
        for phase in range(1, 5):
            initial, selected = metrics(0.6, phase), metrics(0.4, phase)
            metadata['stages'][f'stage{phase}'] = {'student_endpoint': phase, 'teacher_endpoint': phase,
                'steps': 1000, 'best_step': 750, 'accepted': True, 'relative_parameter_change': 0.05,
                'schedule': 'cumulative', 'align_mix': 1.0, 'rank_gate_enabled': False,
                'initial': initial, 'final': selected, 'best_unmixed': selected,
                'history': [{'step': 0, **initial}, {'step': 750, **selected}, {'step': 1000, **metrics(0.5, phase)}]}
    folder = root / 'logs' / name
    folder.mkdir(parents = True)
    write_json(folder / 'args.json', config)
    write_json(folder / 'prior_alignment.json', metadata)
    if complete:
        length = config['ft_epochs']
        top1s = [top1 - 5.0 * (length - index - 1) / length for index in range(length)]
        result = {'final_top1': top1, 'best_top1': top1, 'top1_accs': top1s, 'top5_accs': [value + 20 for value in top1s],
                  'val_losses': [3.0 - 0.01 * index for index in range(length)], 'train_losses': [1.0, 2.0] * length,
                  'grad_norms': [0.0] * (2 * length), 'weight_norms': [0.0] * (2 * length)}
        write_json(folder / ('ft_results_seed' + str(seed) + '.json'), result)
    return folder

def change(path, **values):
    data = json.loads(path.read_text())
    data.update(values)
    write_json(path, data)

class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / 'logs').mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def runs(self):
        runs = discover(self.root, ['*'], [])
        match_baselines(runs)
        return {run['experiment']: run for run in runs}

    def pair(self, **kwargs):
        fixture(self.root, 'baseline', 'baseline', top1 = 60.0, **{key: value for key, value in kwargs.items() if key in ['family', 'epochs', 'seed']})
        fixture(self.root, 'aligned', top1 = 62.0, **kwargs)
        return self.runs()

    def test_real_presets_and_final_not_best(self):
        self.pair()
        path = self.root / 'logs' / 'aligned' / 'ft_results_seed0.json'
        result = json.loads(path.read_text())
        result['top1_accs'][10] = 80.0
        result['best_top1'] = 80.0
        write_json(path, result)
        run = self.runs()['aligned']
        self.assertEqual(run['pair_status'], 'matched')
        self.assertEqual(run['delta_pp'], 2.0)
        self.assertEqual(run['best_top1'], 80.0)
        self.assertEqual(self.runs()['baseline']['method'], 'baseline')

    def test_percent_is_not_rescaled(self):
        self.pair()
        for name, top1 in [('baseline', 0.1), ('aligned', 0.2)]:
            path = self.root / 'logs' / name / 'ft_results_seed0.json'
            change(path, final_top1 = top1, best_top1 = top1, top1_accs = [top1] * 100)
        self.assertAlmostEqual(self.runs()['aligned']['delta_pp'], 0.1)

    def test_training_args_override_alignment_budget(self):
        fixture(self.root, 'baseline100', 'baseline', family = 'vit', epochs = 100, top1 = 60.0)
        fixture(self.root, 'baseline300', 'baseline', family = 'vit', epochs = 300, top1 = 66.0)
        folder = fixture(self.root, 'aligned', family = 'vit', depth = 12, epochs = 100)
        config = json.loads((folder / 'args.json').read_text())
        write_json(folder / 'training_args.json', config)
        change(folder / 'args.json', ft_epochs = 300)
        run = self.runs()['aligned']
        self.assertEqual(run['paired_baseline'], 'baseline100')
        self.assertEqual(run['epochs'], 100)
        self.assertEqual(run['delta_pp'], 2.0)

    def test_incompatible_recipes(self):
        self.pair()
        path = self.root / 'logs' / 'aligned' / 'training_args.json'
        config = json.loads((path.parent / 'args.json').read_text())
        for key, value in [('ft_lr', 0.01), ('ft_accum', 2), ('use_scheduler', False), ('warmup_epochs', 9), ('mixup_alpha', 0.1)]:
            with self.subTest(key = key):
                write_json(path, {**config, key: value})
                self.assertEqual(self.runs()['aligned']['pair_status'], 'recipe_mismatch')

    def test_same_effective_batch_different_microbatch_is_not_matched(self):
        self.pair()
        path = self.root / 'logs' / 'aligned' / 'args.json'
        change(path, ft_batch = 128, ft_accum = 2)
        self.assertEqual(self.runs()['aligned']['pair_status'], 'recipe_mismatch')

    def test_initialization_hash_is_required_and_checked(self):
        self.pair()
        path = self.root / 'logs' / 'aligned' / 'prior_alignment.json'
        change(path, student_initial_hash = 'different_initialization')
        self.assertEqual(self.runs()['aligned']['pair_status'], 'initialization_mismatch')
        change(path, student_initial_hash = None)
        self.assertFalse(self.runs()['aligned']['eligible'])

    def test_ambiguous_baseline_is_not_chosen_by_accuracy(self):
        self.pair()
        fixture(self.root, 'other_baseline', 'baseline', top1 = 80.0)
        self.assertEqual(self.runs()['aligned']['pair_status'], 'ambiguous_baseline')

    def test_missing_baseline_and_incomplete_runs(self):
        fixture(self.root, 'aligned')
        self.assertEqual(self.runs()['aligned']['pair_status'], 'missing_baseline')
        folder = fixture(self.root, 'pending', complete = False)
        run = load_run(self.root, folder)
        self.assertEqual(run['status'], 'started_or_incomplete')
        self.assertIsNone(run['final_top1'])
        change(folder / 'args.json', align_only = True)
        self.assertEqual(load_run(self.root, folder)['status'], 'alignment_only')

    def test_partial_result_json_is_a_warning_not_a_crash(self):
        self.pair()
        path = self.root / 'logs' / 'aligned' / 'ft_results_seed0.json'
        path.write_text('{')
        run = self.runs()['aligned']
        self.assertEqual(run['status'], 'invalid_record')
        self.assertFalse(run['eligible'])

    def test_invalid_training_args_do_not_fall_back(self):
        self.pair()
        (self.root / 'logs' / 'aligned' / 'training_args.json').write_text('{')
        self.assertFalse(self.runs()['aligned']['eligible'])

    def test_nan_final_is_not_compared(self):
        self.pair()
        path = self.root / 'logs' / 'aligned' / 'ft_results_seed0.json'
        value = json.loads(path.read_text())
        value['final_top1'] = float('nan')
        path.write_text(json.dumps(value))
        self.assertFalse(self.runs()['aligned']['eligible'])

    def test_missing_final_is_not_replaced_with_best(self):
        self.pair()
        change(self.root / 'logs' / 'aligned' / 'ft_results_seed0.json', final_top1 = None)
        self.assertIsNone(self.runs()['aligned']['delta_pp'])

    def test_final_must_match_last_validation(self):
        self.pair()
        change(self.root / 'logs' / 'aligned' / 'ft_results_seed0.json', final_top1 = 63.0)
        self.assertFalse(self.runs()['aligned']['eligible'])

    def test_early_stopped_epoch_curve_not_final_result(self):
        self.pair()
        change(self.root / 'logs' / 'aligned' / 'ft_results_seed0.json', top1_accs = [62.0] * 50)
        self.assertFalse(self.runs()['aligned']['eligible'])

    def test_changed_or_trained_guide_is_blocked(self):
        self.pair()
        change(self.root / 'logs' / 'aligned' / 'prior_alignment.json', guide_final_hash = 'changed')
        self.assertFalse(self.runs()['aligned']['eligible'])

    def test_missing_weights_do_not_prevent_json_comparison(self):
        runs = self.pair()
        self.assertEqual(runs['aligned']['pair_status'], 'matched')
        self.assertTrue(any(item['code'] == 'checkpoint_not_present' for item in runs['aligned']['issues']))

    def test_selected_geometry_not_unmixed_or_rejected_candidate(self):
        self.pair()
        path = self.root / 'logs' / 'aligned' / 'prior_alignment.json'
        meta = json.loads(path.read_text())
        stage = meta['stages']['stage2']
        stage.update({'accepted': False, 'final': copy.deepcopy(stage['initial']), 'best_unmixed': metrics(0.1, 2)})
        write_json(path, meta)
        stages, endpoints, history = stage_tables(list(self.runs().values()))
        row = next(row for row in stages if row['stage'] == 'stage2')
        self.assertAlmostEqual(row['loss_reduction'], 0.0)
        self.assertEqual(row['best_unmixed_loss'], 0.1)

    def test_rank_drop_does_not_become_a_posthoc_gate(self):
        self.pair()
        path = self.root / 'logs' / 'aligned' / 'prior_alignment.json'
        meta = json.loads(path.read_text())
        meta['stages']['stage2']['final']['endpoint1_student_rank'] = 3.0
        write_json(path, meta)
        runs = self.runs()
        stages, endpoints, history = stage_tables(list(runs.values()))
        self.assertEqual(runs['aligned']['pair_status'], 'matched')
        self.assertTrue(any(item['code'] == 'rank_drop_diagnostic' for item in runs['aligned']['issues']))

    def test_unrecorded_norms_are_not_measurements(self):
        run = self.pair()['aligned']
        summary = run['result']['comparison_compaction']
        self.assertFalse(summary['norms_recorded'])
        self.assertNotIn('grad_norms', summary)
        self.assertEqual(summary['train_epoch_means'], [1.5] * 100)

    def test_guide_seeds_are_separate_and_one_seed_has_no_sd(self):
        self.pair()
        fixture(self.root, 'guide1001', guide_seed = 1001)
        rows = condition_summary(list(self.runs().values()))
        self.assertEqual(len(rows), 2)
        self.assertEqual([row['n_student_seeds'] for row in rows], [1, 1])
        self.assertTrue(all(row['sd_delta_pp'] is None for row in rows))

    def test_student_seed_summary_uses_paired_differences(self):
        self.pair()
        fixture(self.root, 'baseline1', 'baseline', seed = 1, top1 = 65.0)
        fixture(self.root, 'aligned1', seed = 1, top1 = 66.0)
        rows = condition_summary(list(self.runs().values()))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['n_student_seeds'], 2)
        self.assertAlmostEqual(rows[0]['mean_delta_pp'], 1.5)

    def test_duplicate_aligned_seed_not_counted_twice(self):
        self.pair()
        fixture(self.root, 'duplicate', top1 = 80.0)
        runs = self.runs()
        self.assertEqual(runs['aligned']['pair_status'], 'duplicate_aligned_run')
        self.assertEqual(condition_summary(list(runs.values())), [])

    def test_depth_contrasts_can_be_negative(self):
        self.pair()
        fixture(self.root, 'deeper', guide = 'rn101', top1 = 61.0)
        fixture(self.root, 'basic', guide = 'rn34', top1 = 59.0)
        rows = depth_comparisons(list(self.runs().values()))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['deeper_minus_shallower_pp'], -1.0)

    def test_vit_widths_not_mixed_in_depth_comparisons(self):
        self.pair(family = 'vit', depth = 12)
        fixture(self.root, 'd24', family = 'vit', depth = 24, top1 = 63.0)
        fixture(self.root, 'large', family = 'vit', guide = 'vit_large', top1 = 64.0)
        rows = depth_comparisons(list(self.runs().values()))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['guide_series'], 'vit_width192')

    def test_slurm_partial_results_never_become_final(self):
        folder = fixture(self.root, 'running', complete = False)
        path = self.root / 'logs' / 'slurm-prior-b-train-123_2.out'
        path.write_text('Array B[2]: running, epochs=100, action=train\n'
                        'Step 3, Validation Loss 2.5, Top-1 Accuracy 25.0, Top-5 Accuracy 50.0, Samples 50000\n'
                        'srun: error: task killed\nWANDB_API_KEY=private\n')
        run = load_run(self.root, folder)
        observations = slurm_observations(self.root, [run])
        self.assertEqual(observations[0]['validation'][0]['position'], 3)
        self.assertTrue(observations[0]['errors'])
        self.assertNotIn('private', json.dumps(observations))
        self.assertIsNone(run['final_top1'])

    def test_separate_slurm_stderr_is_collected(self):
        folder = fixture(self.root, 'running', complete = False)
        path = self.root / 'logs' / 'slurm-prior-b-train-123_2.out'
        path.write_text('Array B[2]: running, epochs=100, action=train\n')
        path.with_suffix('.err').write_text('conda: error: unrecognized arguments: --no-capture-output\n')
        run = load_run(self.root, folder)
        observations = slurm_observations(self.root, [run])
        self.assertEqual(len(observations), 1)
        self.assertTrue(observations[0]['errors'])
        self.assertEqual(observations[0]['stderr_file'], path.with_suffix('.err').name)

    def test_different_precision_is_not_matched(self):
        self.pair()
        folder = self.root / 'logs' / 'aligned'
        change(folder / 'args.json', use_amp = True)
        meta = json.loads((folder / 'prior_alignment.json').read_text())
        meta['alignment_config']['use_amp'] = True
        write_json(folder / 'prior_alignment.json', meta)
        self.assertEqual(self.runs()['aligned']['pair_status'], 'recipe_mismatch')

    def test_config_seed_must_match_result_filename(self):
        self.pair()
        folder = self.root / 'logs' / 'aligned'
        (folder / 'ft_results_seed0.json').rename(folder / 'ft_results_seed1.json')
        self.assertFalse(self.runs()['aligned']['eligible'])
        self.assertEqual(self.runs()['aligned']['status'], 'invalid_record')

    def test_no_op_alignment_is_diagnostic_not_rejected(self):
        self.pair()
        path = self.root / 'logs' / 'aligned' / 'prior_alignment.json'
        meta = json.loads(path.read_text())
        meta['student_aligned_hash'] = meta['student_initial_hash']
        write_json(path, meta)
        run = self.runs()['aligned']
        self.assertEqual(run['pair_status'], 'matched')
        self.assertTrue(any(item['code'] == 'alignment_no_op' for item in run['issues']))

    def test_launch_only_conda_failure_is_in_inventory(self):
        folder = self.root / 'logs' / 'launch_failure'
        folder.mkdir()
        write_json(folder / 'slurm_launch_123.json', {'group': 'B', 'name': folder.name})
        self.assertEqual(self.runs()['launch_failure']['status'], 'launch_only')

    def test_bundle_roundtrip_and_inputs_unchanged(self):
        self.pair()
        (self.root / '.env').write_text('SECRET=not for upload')
        (self.root / 'saved_models').mkdir()
        (self.root / 'saved_models' / 'orig_aligned.pt').write_bytes(b'not a real checkpoint')
        change(self.root / 'logs' / 'aligned' / 'args.json', wandb_api_key = 'private')
        before = {str(path): path.read_bytes() for path in (self.root / 'logs').rglob('*.json')}
        output = self.root / 'report'
        args = get_args(['--root', str(self.root), '--output', str(output), '--bundle', '--no_plots'])
        with contextlib.redirect_stdout(io.StringIO()):
            data = analyze(args)
        with zipfile.ZipFile(data['bundle']) as archive:
            names = archive.namelist()
            self.assertFalse(any(name.endswith('.pt') or name.endswith('.env') for name in names))
            payload = archive.read('logs/aligned/ft_results_seed0.json').decode()
            self.assertNotIn('grad_norms', json.loads(payload))
            self.assertEqual(len(json.loads(payload)['top1_accs']), 100)
            self.assertNotIn('private', archive.read('logs/aligned/args.json').decode())
            target = self.root / 'shared'
            archive.extractall(target)
        paired = discover(target, ['*'], [])
        match_baselines(paired)
        self.assertEqual(next(run for run in paired if run['experiment'] == 'aligned')['delta_pp'], 2.0)
        after = {str(path): path.read_bytes() for path in (self.root / 'logs').rglob('*.json')}
        self.assertEqual(before, after)

    def test_command_line_without_gpu_imports(self):
        self.pair()
        output = self.root / 'cli_output'
        completed = subprocess.run([sys.executable, str(ROOT / 'compare_prior.py'), '--root', str(self.root), '--output', str(output), '--no_plots'],
                                   capture_output = True, text = True, timeout = 30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((output / 'paired_comparisons.csv').exists())

    def test_plots_smoke(self):
        if Figure is None:
            self.skipTest('Matplotlib is unavailable')
        self.pair()
        args = get_args(['--root', str(self.root), '--output', str(self.root / 'plots_test')])
        with contextlib.redirect_stdout(io.StringIO()):
            result = analyze(args)
        self.assertGreaterEqual(len(result['plots']), 6)
        self.assertTrue(all(path.read_bytes().startswith(b'\x89PNG') for path in result['plots']))

    def test_filters_disambiguate(self):
        self.pair()
        fixture(self.root, 'other_baseline', 'baseline')
        runs = discover(self.root, ['*'], ['other_*'])
        match_baselines(runs)
        self.assertEqual(next(run for run in runs if run['experiment'] == 'aligned')['pair_status'], 'matched')

    def test_no_matplotlib_still_writes_tables(self):
        self.pair()
        args = get_args(['--root', str(self.root), '--output', str(self.root / 'no_matplotlib')])
        with patch('compare_prior.Figure', None), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = analyze(args)
        self.assertEqual(result['plots'], [])
        self.assertTrue((Path(args.output) / 'report.md').is_file())

    def test_output_cannot_overwrite_inputs(self):
        self.pair()
        for output in [self.root, self.root / 'logs', self.root / 'logs' / 'aligned', self.root / 'saved_models']:
            with self.assertRaises(ValueError):
                analyze(get_args(['--root', str(self.root), '--output', str(output)]))

    def test_step_validation_not_labelled_epochs(self):
        self.pair()
        for name in ['baseline', 'aligned']:
            change(self.root / 'logs' / name / 'args.json', evaluate = 'steps')
        self.assertEqual(self.runs()['aligned']['pair_status'], 'matched')

    def test_new_code_style(self):
        for name in ['compare_prior.py', 'test_compare_prior.py']:
            source = (ROOT / name).read_text()
            self.assertNotIn('\n\n\n', source)
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                self.assertNotEqual(token.type, tokenize.COMMENT)
                if token.type == tokenize.STRING:
                    self.assertFalse(token.string.lstrip('frbuFRBU').startswith(chr(34)))

if __name__ == '__main__':
    unittest.main(verbosity = 2)
