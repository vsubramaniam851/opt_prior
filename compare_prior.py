import argparse
import csv
import fnmatch
import hashlib
import json
import math
import platform
import re
import statistics
import sys
import zipfile

from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
except ImportError:
    FigureCanvasAgg = None
    Figure = None

TRAINING_KEYS = ['setting', 'imagenet', 'data_dir', 'ft_epochs', 'ft_batch', 'ft_accum', 'ft_optimizer', 'ft_lr',
                 'ft_wd', 'ft_clip', 'label_smoothing', 'mixup_alpha', 'use_scheduler', 'scheduler_type',
                 'warmup_epochs', 'use_amp', 'num_workers', 'evaluate', 'pretrained', 'fc_only', 'tune_linear',
                 'preserve_alignment', 'continue_ft', 'measure_alignment']
ALIGNMENT_KEYS = ['repdist', 'cka_loss', 'knn_loss', 'knn_k', 'knn_temp', 'rep_batch', 'rep_val_size',
                  'rep_split_seed', 'rep_val_interval', 'max_stage_steps', 'progressive_lrs', 'progressive_epochs',
                  'rep_wd', 'rep_scheduler', 'pool_hw', 'prior_schedule', 'align_mix', 'prior_min_rank_ratio',
                  'prior_norm', 'prior_descriptor', 'anchor_to_init', 'anchor_lambda']
METRICS = ['loss', 'cka', 'ucka', 'knn_overlap', 'student_rank', 'teacher_rank', 'student_centered_rms', 'teacher_centered_rms']
SOURCE_FILES = ['args.py', 'run_prior.py', 'theseus.py', 'prior_utils.py', 'rep_sims.py', 'finetune.py',
                'dataloader.py', 'vits.py', 'slurm_scripts/run_prior_array.py', 'compare_prior.py',
                'audit_prior_geometry.py', 'restart_prior_baseline.py']
VITS = {'vit_tiny': (192, 4), 'vit_small': (384, 8), 'vit_base': (768, 12), 'vit_large': (1024, 24)}
SECRET_KEYS = {'api_key', 'wandb_api_key', 'hf_token', 'access_token', 'auth_token', 'password', 'secret', 'token'}

def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys = True, allow_nan = False).encode()).hexdigest()[:12]

def clean(value):
    if isinstance(value, dict):
        return {key: '[REDACTED]' if key.lower() in SECRET_KEYS else clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def text_redact(text):
    return re.sub(r'(?i)\b(WANDB_API_KEY|HF_TOKEN|ACCESS_TOKEN|API_KEY|PASSWORD)\s*[=:]\s*[^\s,;]+', r'\1=[REDACTED]', text)

def read_json(path):
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object: ' + str(path))
    return value

def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent = 2, allow_nan = False) + '\n')

def file_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def average(values):
    values = [value for value in values if finite(value)]
    return statistics.fmean(values) if values else None

def ratio(a, b):
    return a / b if finite(a) and finite(b) and b > 0 else None

def difference(a, b):
    return a - b if finite(a) and finite(b) else None

def issue(run, code, message, blocking = False):
    run['issues'].append({'experiment': run['experiment'], 'code': code, 'message': message, 'blocking': blocking})

def summarize_series(values, limit = 600):
    values = values if isinstance(values, list) else []
    good = [value for value in values if finite(value)]
    width = max(1, math.ceil(len(values) / limit))
    bins = [{'start': start + 1, 'end': min(start + width, len(values)), 'mean': average(values[start:start + width])}
            for start in range(0, len(values), width)]
    return {'count': len(values), 'nonfinite': len(values) - len(good), 'mean': statistics.fmean(good) if good else None, 'bins': bins}

def compact_result(result, config):
    if 'comparison_compaction' in result:
        return result
    output = {key: value for key, value in result.items() if key not in ['train_losses', 'grad_norms', 'weight_norms']}
    summary = {'version': 1, 'train_losses': summarize_series(result.get('train_losses')),
               'norms_recorded': config.get('compute_norms') is True}
    epochs = config.get('ft_epochs')
    losses = result.get('train_losses', [])
    if isinstance(losses, list) and isinstance(epochs, int) and epochs > 0 and losses and len(losses) % epochs == 0:
        width = len(losses) // epochs
        summary['train_epoch_means'] = [average(losses[i:i + width]) for i in range(0, len(losses), width)]
        summary['train_epoch_axis'] = 'Inferred from equal minibatch counts per epoch in the v2 trainer'
    if summary['norms_recorded']:
        for name in ['grad_norms', 'weight_norms']:
            summary[name] = summarize_series(result.get(name))
    output['comparison_compaction'] = summary
    return output

def checkpoint_inventory(root, name):
    paths = [root / 'saved_models' / (prefix + name + '.pt') for prefix in ['orig_', 'ft_', 'ft_final_']]
    paths += sorted((root / 'saved_models' / name).glob('prior_stage*.pt'))
    return [{'path': str(path.relative_to(root)), 'present': path.is_file(),
             'bytes': path.stat().st_size if path.is_file() else None} for path in paths]

def load_run(root, folder):
    run = {'experiment': folder.name, 'issues': [], 'sources': [], 'config': {}, 'metadata': {}, 'result': {},
           'status': 'unknown', 'eligible': False, 'pair_status': 'not_eligible', 'paired_baseline': None,
           'delta_pp': None, 'recipe_id': None, 'condition_id': None, 'student': None, 'method': None,
           'guide': None, 'guide_depth': None, 'guide_width': None, 'guide_series': None, 'seed': None,
           'guide_seed': None, 'epochs': None, 'final_top1': None, 'best_top1': None, 'final_val_loss': None,
           'checkpoints': checkpoint_inventory(root, folder.name), 'snapshots': {}}
    for name in ['args.json', 'training_args.json', 'prior_alignment.json', 'initialization_source.json']:
        path = folder / name
        if not path.is_file():
            continue
        run['sources'].append(path)
        try:
            value = read_json(path)
            run['snapshots'][name] = value
            if name in ['args.json', 'training_args.json']:
                run['config'].update(value)
            elif name == 'prior_alignment.json':
                run['metadata'] = value
        except (OSError, ValueError) as error:
            issue(run, 'unreadable_json', str(error), True)
    for path in sorted(folder.glob('slurm_launch_*.json')):
        run['sources'].append(path)
        try:
            run['snapshots'][path.name] = read_json(path)
        except (OSError, ValueError) as error:
            issue(run, 'unreadable_launch', str(error))
    config, meta = run['config'], run['metadata']
    baseline = meta.get('baseline', config.get('baseline')) is True
    run.update({'student': meta.get('student'), 'seed': config.get('seed'), 'guide_seed': meta.get('guide_seed'),
                'guide': None if baseline else meta.get('guide', config.get('model')),
                'method': 'baseline' if baseline else config.get('repdist'), 'epochs': config.get('ft_epochs')})
    if run['student'] is None:
        replacement = config.get('replacement_type')
        run['student'] = 'rn18' if replacement == 'progressive_align_resnet' else config.get('student_vit') if replacement == 'progressive_align_vits' else None
    guide = run['guide']
    if guide in VITS:
        width, depth = VITS[guide]
        run.update({'guide_width': width, 'guide_depth': meta.get('guide_depth_override') or depth, 'guide_series': f'vit_width{width}'})
    elif guide and re.fullmatch(r'rn(18|34|50|101|152)', guide):
        depth = int(guide[2:])
        run.update({'guide_depth': depth, 'guide_series': 'resnet_basicblock' if depth < 50 else 'resnet_bottleneck'})
    expected = folder / ('ft_results_seed' + str(run['seed']) + '.json')
    other_results = list(folder.glob('ft_results_seed*.json'))
    if expected.is_file():
        run['sources'].append(expected)
        try:
            run['result'] = compact_result(read_json(expected), config)
        except (OSError, ValueError, TypeError) as error:
            issue(run, 'unreadable_result', str(error), True)
    elif other_results:
        issue(run, 'result_seed_mismatch', 'Result filenames do not match the saved training seed', True)
    result = run['result']
    if result:
        run['status'] = 'completed_result'
    elif expected.is_file() or any(item['blocking'] for item in run['issues']):
        run['status'] = 'invalid_record'
    elif (folder / 'training_args.json').exists() or (config and not config.get('align_only', False)):
        run['status'] = 'started_or_incomplete'
    elif meta.get('complete'):
        run['status'] = 'alignment_only'
    elif meta:
        run['status'] = 'partial_alignment'
    elif not config:
        run['status'] = 'launch_only'
    if not result:
        return run
    required = TRAINING_KEYS + ['seed']
    missing = [key for key in required if key not in config]
    if missing or not run['student']:
        issue(run, 'missing_config', 'Cannot establish a matched training recipe: ' + ', '.join(missing), True)
    if config.get('optimization_prior') is not True or meta.get('version') != 2:
        issue(run, 'unsupported_version', 'Expected optimization-prior v2 metadata', True)
    if not meta.get('complete') or meta.get('classification_labels_used') is not False:
        issue(run, 'invalid_alignment_metadata', 'Alignment must be complete and label-free', True)
    if meta.get('guide_unchanged') is not True or not meta.get('guide_initial_hash') or meta.get('guide_initial_hash') != meta.get('guide_final_hash'):
        issue(run, 'guide_changed_or_unknown', 'Recorded guide fingerprints are missing or do not match', True)
    if not meta.get('student_initial_hash') or not meta.get('student_aligned_hash'):
        issue(run, 'missing_initialization_hash', 'Recorded student fingerprints are required for pairing', True)
    if run['seed'] != meta.get('student_seed'):
        issue(run, 'seed_mismatch', 'Training and alignment student seeds differ', True)
    if config.get('pretrained') is not False or config.get('continue_ft') or config.get('preserve_alignment'):
        issue(run, 'different_experiment', 'Expected a random guide and ordinary downstream training from initialization', True)
    if meta.get('baseline') != config.get('baseline'):
        issue(run, 'baseline_metadata_mismatch', 'Baseline flags differ between arguments and metadata', True)
    if baseline and meta.get('student_initial_hash') != meta.get('student_aligned_hash'):
        issue(run, 'changed_baseline_initialization', 'The baseline initialization was modified', True)
    if not baseline and meta.get('student_initial_hash') == meta.get('student_aligned_hash'):
        issue(run, 'alignment_no_op', 'The retained initialization has the same recorded state hash as the random student')
    alignment = meta.get('alignment_config', {})
    mismatches = [key for key, value in alignment.items() if key in config and config[key] != value]
    if not alignment or mismatches:
        issue(run, 'alignment_config_mismatch', 'Missing alignment configuration or differing values: ' + ', '.join(mismatches), True)
    final = result.get('final_top1')
    top1 = result.get('top1_accs')
    if not finite(final) or not 0 <= final <= 100:
        issue(run, 'invalid_final_top1', 'A finite final_top1 in percentage units is required; best_top1 is never substituted', True)
    elif not isinstance(top1, list) or not top1 or not all(finite(value) and 0 <= value <= 100 for value in top1):
        issue(run, 'invalid_accuracy_curve', 'Missing or invalid top1_accs', True)
    elif abs(top1[-1] - final) > 1e-5:
        issue(run, 'final_curve_mismatch', 'final_top1 differs from the last top1_accs entry', True)
    if isinstance(top1, list):
        if config.get('evaluate') == 'epoch' and len(top1) != run['epochs']:
            issue(run, 'epoch_count_mismatch', 'Expected one validation measurement per completed epoch', True)
        for key in ['val_losses', 'top5_accs']:
            values = result.get(key)
            if not isinstance(values, list) or len(values) != len(top1) or not all(finite(value) for value in values):
                issue(run, 'invalid_validation_curve', 'Invalid or inconsistent ' + key, True)
    compaction = result.get('comparison_compaction', {})
    if compaction.get('train_losses', {}).get('nonfinite', 0):
        issue(run, 'nonfinite_training_loss', 'Saved training losses contain invalid values', True)
    if not any(item['path'].endswith('ft_final_' + folder.name + '.pt') and item['present'] for item in run['checkpoints']):
        issue(run, 'checkpoint_not_present', 'Final weights are not present here; JSON comparisons do not require them')
    run['final_top1'] = final if finite(final) else None
    run['best_top1'] = max(top1) if isinstance(top1, list) and top1 and all(finite(value) for value in top1) else None
    losses = result.get('val_losses', [])
    run['final_val_loss'] = losses[-1] if isinstance(losses, list) and losses and finite(losses[-1]) else None
    if finite(result.get('best_top1')) and run['best_top1'] is not None and abs(result['best_top1'] - run['best_top1']) > 1e-5:
        issue(run, 'best_top1_mismatch', 'The saved best_top1 disagrees with the validation curve; final_top1 remains the primary metric')
    recipe = {key: config.get(key) for key in TRAINING_KEYS}
    recipe.update({'student': run['student'], 'eval_steps': config.get('eval_steps') if config.get('evaluate') == 'steps' else None,
                   'flat_epochs': config.get('flat_epochs') if config.get('use_scheduler') and config.get('scheduler_type') == 'cosine_flat' else None})
    run['recipe'] = recipe
    run['recipe_id'] = digest(clean(recipe))
    controls = {key: alignment.get(key, config.get(key)) for key in ALIGNMENT_KEYS}
    controls['prior_descriptor'] = alignment.get('prior_descriptor', config.get('prior_descriptor', 'pooled'))
    controls['validation_indices_hash'] = digest(meta['validation_indices']) if isinstance(meta.get('validation_indices'), list) else None
    run['alignment_controls'] = controls
    run['alignment_id'] = digest(clean(controls))
    condition = {'recipe_id': run['recipe_id'], 'method': run['method'], 'guide': guide, 'guide_depth': run['guide_depth'],
                 'guide_seed': None if baseline else run['guide_seed'], 'alignment_id': None if baseline else run['alignment_id']}
    run['condition_id'] = digest(condition)
    run['eligible'] = not any(item['blocking'] for item in run['issues'])
    if not run['eligible']:
        run['status'] = 'invalid_record'
    return run

def discover(root, includes, excludes):
    logs = root / 'logs'
    if not logs.is_dir():
        raise FileNotFoundError('Expected a logs directory under --root: ' + str(root))
    runs = []
    for folder in sorted(logs.iterdir()):
        if not folder.is_dir() or folder.is_symlink() or not any(fnmatch.fnmatchcase(folder.name, item) for item in includes):
            continue
        if any(fnmatch.fnmatchcase(folder.name, item) for item in excludes):
            continue
        relevant = (folder / 'prior_alignment.json').exists() or any(folder.glob('slurm_launch_*.json'))
        if not relevant and (folder / 'args.json').is_file():
            try:
                relevant = read_json(folder / 'args.json').get('optimization_prior') is True
            except (OSError, ValueError):
                relevant = folder.name.startswith('prior_')
        if relevant:
            runs.append(load_run(root, folder))
    return runs

def match_baselines(runs):
    baselines = [run for run in runs if run['eligible'] and run['method'] == 'baseline']
    duplicates = defaultdict(list)
    for run in runs:
        if run['eligible'] and run['method'] != 'baseline':
            duplicates[(run['condition_id'], run['seed'])].append(run)
    for entries in duplicates.values():
        if len(entries) > 1:
            for run in entries:
                issue(run, 'duplicate_condition_seed', 'Multiple aligned runs have the same condition and student seed; use --exclude to choose explicitly', True)
                run['eligible'] = False
                run['pair_status'] = 'duplicate_aligned_run'
    for run in runs:
        if not run['eligible']:
            continue
        if run['method'] == 'baseline':
            run['pair_status'] = 'baseline'
            continue
        related = [base for base in baselines if base['student'] == run['student'] and base['seed'] == run['seed']]
        compatible = [base for base in related if base['recipe'] == run['recipe']]
        matched = [base for base in compatible if base['metadata']['student_initial_hash'] == run['metadata']['student_initial_hash']]
        if len(matched) == 1:
            base = matched[0]
            run.update({'pair_status': 'matched', 'paired_baseline': base['experiment'],
                        'baseline_top1': base['final_top1'], 'delta_pp': run['final_top1'] - base['final_top1']})
        elif len(matched) > 1:
            run['pair_status'] = 'ambiguous_baseline'
            issue(run, 'ambiguous_baseline', 'Several baselines match; use --exclude rather than selecting the best one')
        elif compatible:
            run['pair_status'] = 'initialization_mismatch'
            issue(run, 'initialization_mismatch', 'Student seed matches, but the recorded initial weight hash does not')
        else:
            run['pair_status'] = 'missing_baseline' if not related else 'recipe_mismatch'
            detail = '; '.join(base['experiment'] + ': ' + ', '.join(key for key in run['recipe'] if base['recipe'].get(key) != run['recipe'][key]) for base in related)
            issue(run, run['pair_status'], detail or 'No completed valid baseline for this student and student seed')

def stage_tables(runs):
    stages, endpoints, history = [], [], []
    for run in runs:
        metadata = run['metadata']
        for stage_name, stage in metadata.get('stages', {}).items():
            if not isinstance(stage, dict):
                issue(run, 'invalid_stage', stage_name + ' is not an object')
                continue
            initial, final = stage.get('initial', {}), stage.get('final', {})
            common = {'experiment': run['experiment'], 'method': run['method'], 'stage': stage_name,
                      'student_endpoint': stage.get('student_endpoint'), 'teacher_endpoint': stage.get('teacher_endpoint')}
            row = {**common, 'accepted': stage.get('accepted'), 'steps': stage.get('steps'), 'best_step': stage.get('best_step'),
                   'align_mix': stage.get('align_mix'), 'rank_gate_enabled': stage.get('rank_gate_enabled'),
                   'relative_parameter_change': stage.get('relative_parameter_change'), 'initial_loss': initial.get('loss'),
                   'selected_loss': final.get('loss'), 'best_unmixed_loss': stage.get('best_unmixed', {}).get('loss'),
                   'loss_reduction': difference(initial.get('loss'), final.get('loss')),
                   'rank_ratio': ratio(final.get('student_rank'), initial.get('student_rank'))}
            stages.append(row)
            if stage.get('accepted') is False:
                issue(run, 'rejected_phase', stage_name + ' retained its phase-start state; best_unmixed is not the retained geometry')
            indices = sorted({int(match.group(1)) for key in initial for match in [re.match(r'endpoint(\d+)_', key)] if match})
            for index in indices:
                endpoint = {**common, 'endpoint': index}
                for key in METRICS:
                    endpoint['initial_' + key] = initial.get(f'endpoint{index}_{key}')
                    endpoint['selected_' + key] = final.get(f'endpoint{index}_{key}')
                endpoint['loss_reduction'] = difference(endpoint['initial_loss'], endpoint['selected_loss'])
                endpoint['rank_ratio'] = ratio(endpoint['selected_student_rank'], endpoint['initial_student_rank'])
                endpoint['centered_rms_ratio'] = ratio(endpoint['selected_student_centered_rms'], endpoint['initial_student_centered_rms'])
                endpoints.append(endpoint)
                if finite(endpoint['rank_ratio']) and endpoint['rank_ratio'] < 0.5:
                    issue(run, 'rank_drop_diagnostic', f'{stage_name} endpoint{index}: participation rank fell below half its phase-start value; diagnostic only')
            for point in stage.get('history', []):
                history.append({**common, **point})
    return stages, endpoints, history

def condition_summary(runs):
    groups = defaultdict(list)
    for run in runs:
        if run['pair_status'] == 'matched':
            groups[run['condition_id']].append(run)
    output = []
    for condition, entries in sorted(groups.items()):
        first = entries[0]
        values = [run['delta_pp'] for run in entries]
        output.append({'condition_id': condition, 'recipe_id': first['recipe_id'], 'student': first['student'],
                       'method': first['method'], 'guide': first['guide'], 'guide_depth': first['guide_depth'],
                       'guide_width': first['guide_width'], 'guide_seed': first['guide_seed'], 'epochs': first['epochs'],
                       'alignment_id': first['alignment_id'], 'student_seeds': sorted(run['seed'] for run in entries),
                       'n_student_seeds': len(entries), 'mean_final_top1': average([run['final_top1'] for run in entries]),
                       'mean_delta_pp': average(values), 'sd_delta_pp': statistics.stdev(values) if len(values) > 1 else None,
                       'min_delta_pp': min(values), 'max_delta_pp': max(values), 'experiments': [run['experiment'] for run in entries]})
    return output

def depth_comparisons(runs):
    groups = defaultdict(list)
    for run in runs:
        if run['eligible'] and run['method'] != 'baseline' and run['guide_series'] and run['guide_depth']:
            key = (run['recipe_id'], run['alignment_id'], run['guide_series'], run['seed'], run['guide_seed'], run['metadata']['student_initial_hash'])
            groups[key].append(run)
    rows = []
    for entries in groups.values():
        entries.sort(key = lambda run: run['guide_depth'])
        for lower, higher in zip(entries, entries[1:]):
            if lower['guide_depth'] == higher['guide_depth']:
                continue
            rows.append({'student': lower['student'], 'method': lower['method'], 'guide_series': lower['guide_series'],
                         'recipe_id': lower['recipe_id'], 'alignment_id': lower['alignment_id'], 'seed': lower['seed'],
                         'guide_seed': lower['guide_seed'], 'epochs': lower['epochs'], 'shallower': lower['experiment'],
                         'deeper': higher['experiment'], 'shallower_depth': lower['guide_depth'], 'deeper_depth': higher['guide_depth'],
                         'deeper_minus_shallower_pp': higher['final_top1'] - lower['final_top1']})
    return rows

def slurm_observations(root, runs):
    wanted = {run['experiment'] for run in runs}
    saved = root / 'slurm_observations.json'
    output = []
    if saved.is_file():
        output.extend(item for item in read_json(saved).get('attempts', []) if item.get('experiment') in wanted)
    pattern = re.compile(r'Step (\d+), Validation Loss ([^,\s]+), Top-1 Accuracy ([^,\s]+), Top-5 Accuracy ([^,\s]+), Samples (\d+)')
    for path in sorted((root / 'logs').glob('slurm-prior-*.out')):
        name, points, errors, tail = None, [], [], deque(maxlen = 35)
        sources = [item for item in [path, path.with_suffix('.err')] if item.is_file()]
        for source in sources:
            with source.open(errors = 'replace') as handle:
                for line in handle:
                    line = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line)
                    match = re.search(r'Array [BCD]\[\d+\]: (.*?), epochs=', line)
                    if match:
                        name = match.group(1)
                    for match in pattern.finditer(line):
                        try:
                            point = dict(zip(['position', 'val_loss', 'top1', 'top5', 'samples'], [int(match[1]), float(match[2]), float(match[3]), float(match[4]), int(match[5])]))
                            if all(finite(value) for value in point.values()):
                                points.append(point)
                        except ValueError:
                            pass
                    if re.search(r'(?i)(Traceback|Error:|\[ERROR\]|out of memory|DUE TO TIME LIMIT|non-zero exit status)', line):
                        errors.append(text_redact(line.strip()[-1500:]))
                    for part in line.split('\r'):
                        if part.strip():
                            tail.append(text_redact(part.strip()[-1500:]))
        if name in wanted:
            output.append({'experiment': name, 'file': path.name, 'stderr_file': path.with_suffix('.err').name if path.with_suffix('.err').is_file() else None,
                           'mtime': max(source.stat().st_mtime for source in sources), 'validation': points,
                           'errors': errors[-10:], 'tail': list(tail)})
    return output

def inventory_row(run, observations):
    names = ['experiment', 'status', 'eligible', 'pair_status', 'student', 'method', 'guide', 'guide_depth', 'guide_width',
             'seed', 'guide_seed', 'epochs', 'recipe_id', 'condition_id', 'final_top1', 'best_top1', 'final_val_loss',
             'paired_baseline', 'baseline_top1', 'delta_pp']
    row = {name: run.get(name) for name in names}
    meta = run['metadata']
    row.update({'schedule': meta.get('schedule'), 'align_mix': meta.get('align_mix'),
                'rank_gate_enabled': meta.get('rank_gate_enabled'), 'guide_unchanged': meta.get('guide_unchanged'),
                'initialization_hash': meta.get('student_initial_hash'), 'aligned_hash': meta.get('student_aligned_hash'),
                'initialization_changed': meta.get('student_aligned_hash') != meta.get('student_initial_hash') if meta.get('student_aligned_hash') else None,
                'phase_count': len(meta.get('stages', {})),
                'accepted_phases': sum(stage.get('accepted') is True for stage in meta.get('stages', {}).values() if isinstance(stage, dict)),
                'effective_batch': run['config'].get('ft_batch', 0) * run['config'].get('ft_accum', 0),
                'precision': 'bf16' if run['config'].get('use_amp') else 'float32'})
    attempts = sorted([item for item in observations if item['experiment'] == run['experiment']], key = lambda item: item['mtime'])
    if attempts:
        latest = attempts[-1]
        row['latest_log'] = latest['file']
        row['log_error_observed'] = bool(latest['errors'])
        if latest['errors']:
            issue(run, 'slurm_error_observed', latest['file'] + ': ' + latest['errors'][-1] + ' (log observation, not a scheduler-state query)')
        if latest['validation']:
            last = latest['validation'][-1]
            row.update({'last_observed_position': last['position'], 'last_observed_top1': last['top1'],
                        'observed_axis': 'epoch' if run['config'].get('evaluate') == 'epoch' else 'validation_step'})
    row['issues'] = '; '.join(dict.fromkeys(item['code'] for item in run['issues']))
    return row

def progress_rows(runs, observations):
    by_name = {run['experiment']: run for run in runs}
    rows = []
    for attempt in observations:
        run = by_name[attempt['experiment']]
        axis = 'epoch' if run['config'].get('evaluate') == 'epoch' else 'validation_step'
        for point in attempt['validation']:
            rows.append({'experiment': run['experiment'], 'attempt': attempt['file'], 'axis': axis,
                         'partial_observation': True, **point})
    return rows

def curve_rows(runs):
    rows = []
    for run in runs:
        result = run['result']
        for key in ['top1_accs', 'top5_accs', 'val_losses']:
            for index, value in enumerate(result.get(key, []) if isinstance(result.get(key), list) else []):
                if finite(value):
                    rows.append({'experiment': run['experiment'], 'metric': key,
                                 'axis': 'epoch' if run['config'].get('evaluate') == 'epoch' else 'evaluation_index',
                                 'position': index + 1, 'value': value})
        for index, value in enumerate(result.get('comparison_compaction', {}).get('train_epoch_means', [])):
            if finite(value):
                rows.append({'experiment': run['experiment'], 'metric': 'train_loss_mean', 'axis': 'inferred_epoch', 'position': index + 1, 'value': value})
    return rows

def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ['no_rows']
    with path.open('w', newline = '') as handle:
        writer = csv.DictWriter(handle, fieldnames = fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(clean(value)) if isinstance(value, (list, dict)) else clean(value) for key, value in row.items()})

def safe_name(name):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)[:100] + '_' + digest(name)[:6]

def short_label(run):
    if run['method'] == 'baseline':
        return 'baseline / s' + str(run['seed'])
    guide = run['guide'] + (' d' + str(run['guide_depth']) if run['guide_width'] else '')
    meta = run['metadata']
    method, seed, guide_seed = run['method'], run['seed'], run['guide_seed']
    schedule, mix = meta.get('schedule'), meta.get('align_mix')
    return f'{method} / {guide} / s{seed} g{guide_seed} / {schedule} a={mix}'

def figure(title, xlabel, ylabel, size = (9, 5)):
    fig = Figure(figsize = size, dpi = 150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha = 0.2)
    return fig, ax

def save_figure(fig, ax, path, legend = False):
    if legend:
        ax.legend(fontsize = 8, loc = 'best')
    fig.tight_layout()
    fig.savefig(path, dpi = 160, bbox_inches = 'tight')
    fig.clear()
    return path

def make_plots(runs, observations, output):
    paths = []
    if Figure is None:
        return paths
    folder = output / 'plots'
    folder.mkdir(exist_ok = True)
    by_name = {run['experiment']: run for run in runs}
    groups = defaultdict(list)
    for run in runs:
        if run['eligible']:
            groups[run['recipe_id']].append(run)
    for recipe, entries in groups.items():
        entries.sort(key = lambda run: (run['method'] != 'baseline', run['method'], run['guide_depth'] or 0, run['seed'], run['guide_seed'] or 0))
        title = entries[0]['student'] + ' / ' + str(entries[0]['epochs']) + ' epochs / recipe ' + recipe
        fig, ax = figure(title, 'Final ImageNet top-1 (%)', '', (12, max(4, 0.42 * len(entries))))
        ax.scatter([run['final_top1'] for run in entries], list(range(len(entries))))
        ax.set_yticks(list(range(len(entries))))
        ax.set_yticklabels([short_label(run) for run in entries], fontsize = 8)
        ax.set_ylim(len(entries) - 0.5, -0.5)
        paths.append(save_figure(fig, ax, folder / ('final_accuracy_' + recipe + '.png')))
        pairs = [run for run in entries if run['pair_status'] == 'matched']
        if pairs:
            fig, ax = figure(title, 'Final top-1 minus matched baseline (percentage points)', '', (12, max(4, 0.42 * len(pairs))))
            ax.scatter([run['delta_pp'] for run in pairs], list(range(len(pairs))))
            ax.axvline(0, linestyle = '--', linewidth = 1)
            ax.set_yticks(list(range(len(pairs))))
            ax.set_yticklabels([short_label(run) for run in pairs], fontsize = 8)
            ax.set_ylim(len(pairs) - 0.5, -0.5)
            paths.append(save_figure(fig, ax, folder / ('paired_deltas_' + recipe + '.png')))
    for run in runs:
        if not run['eligible']:
            continue
        base = by_name.get(run['paired_baseline'])
        x_label = 'Epoch' if run['config'].get('evaluate') == 'epoch' else 'Evaluation index (not epoch)'
        for key, ylabel in [('top1_accs', 'ImageNet validation top-1 (%)'), ('val_losses', 'Validation cross-entropy')]:
            fig, ax = figure(run['experiment'], x_label, ylabel, (11, 5))
            for entry in [run] + ([base] if base else []):
                values = entry['result'].get(key, [])
                ax.plot(range(1, len(values) + 1), values, label = 'matched baseline' if entry is base else short_label(entry))
            paths.append(save_figure(fig, ax, folder / (key + '_' + safe_name(run['experiment']) + '.png'), True))
    depth_groups = defaultdict(list)
    for run in runs:
        if run['pair_status'] == 'matched' and run['guide_series']:
            key = (run['recipe_id'], run['alignment_id'], run['guide_series'])
            depth_groups[key].append(run)
    for key, entries in depth_groups.items():
        trajectories = defaultdict(list)
        for run in entries:
            trajectories[(run['seed'], run['guide_seed'], run['paired_baseline'])].append(run)
        if not any(len({run['guide_depth'] for run in series}) > 1 for series in trajectories.values()):
            continue
        fig, ax = figure(entries[0]['method'] + ' / ' + key[2] + ' / recipe ' + key[0], 'Guide depth', 'Paired final top-1 gain (percentage points)')
        for (seed, guide_seed, _), series in sorted(trajectories.items()):
            series.sort(key = lambda run: run['guide_depth'])
            ax.plot([run['guide_depth'] for run in series], [run['delta_pp'] for run in series], marker = 'o', label = f's{seed}, g{guide_seed}')
        ax.axhline(0, linestyle = '--', linewidth = 1)
        paths.append(save_figure(fig, ax, folder / ('depth_' + digest(key) + '.png'), True))
    for run in runs:
        for name, stage in run['metadata'].get('stages', {}).items():
            if not isinstance(stage, dict):
                continue
            history = [point for point in stage.get('history', []) if finite(point.get('step')) and finite(point.get('loss'))]
            if not history:
                continue
            fig, ax = figure(run['experiment'] + ' / ' + name, 'Updates within this phase', 'Held-out active-endpoint alignment loss', (11, 5))
            ax.plot([point['step'] for point in history], [point['loss'] for point in history], marker = 'o', label = 'evaluated trajectory')
            selected = stage.get('final', {}).get('loss')
            if finite(selected):
                position = stage.get('best_step', 0) if stage.get('accepted') else 0
                ax.scatter([position], [selected], marker = '*', s = 140, label = 'retained model after interpolation / rejection')
            paths.append(save_figure(fig, ax, folder / ('alignment_' + safe_name(run['experiment'] + '_' + name) + '.png'), True))
    for attempt in observations:
        run = by_name[attempt['experiment']]
        points = attempt['validation']
        if run['result'] or not points:
            continue
        xlabel = 'Observed epoch' if run['config'].get('evaluate') == 'epoch' else 'Validation step printed by trainer'
        fig, ax = figure('PARTIAL / ' + run['experiment'] + ' / ' + attempt['file'], xlabel, 'Observed validation top-1 (%)', (11, 5))
        ax.plot([point['position'] for point in points], [point['top1'] for point in points], marker = 'o', label = 'not a completed result')
        paths.append(save_figure(fig, ax, folder / ('partial_top1_' + safe_name(attempt['file']) + '.png'), True))
    return paths

def fmt(value):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:.3f}'
    return str(value).replace('|', '\\|').replace('\n', ' ')

def markdown_table(rows, columns):
    output = ['| ' + ' | '.join(label for key, label in columns) + ' |', '| ' + ' | '.join('---' for column in columns) + ' |']
    output += ['| ' + ' | '.join(fmt(row.get(key)) for key, label in columns) + ' |' for row in rows]
    return '\n'.join(output)

def report_text(runs, inventory, summary, depths, issues, plot_paths):
    counts = Counter(run['status'] for run in runs)
    lines = ['# Optimization-prior comparison', '', 'Generated UTC: ' + datetime.now(timezone.utc).isoformat(), '',
             'Final top-1 is the primary outcome. Accuracy is already in percent; differences are percentage points. '
             'Matching requires the same student, student seed, recorded initial-state hash, data path, and downstream recipe. '
             'Recorded hashes are compared, not recomputed from checkpoints.', '', '## Inventory', '',
             ', '.join(key + '=' + str(value) for key, value in sorted(counts.items())), '',
             markdown_table(inventory, [('experiment', 'Experiment'), ('status', 'State'), ('epochs', 'Epochs'),
                                        ('final_top1', 'Final top-1'), ('delta_pp', 'Paired gain (pp)'), ('pair_status', 'Pairing')]), '',
             '## Seed summaries', '',
             'Guide seeds remain separate. Each row averages paired differences over distinct student seeds. '
             'SD is the sample standard deviation, not a confidence interval; it is blank for one seed. '
             'Repeated guide seeds sharing a student are not counted as independent student replicates.', '']
    partial = [row for row in inventory if row.get('last_observed_top1') is not None and row['final_top1'] is None]
    if partial:
        position = lines.index('## Seed summaries')
        lines[position:position] = ['## Partial log observations', '', markdown_table(partial, [('experiment', 'Experiment'),
                       ('last_observed_position', 'Last position'), ('observed_axis', 'Axis'), ('last_observed_top1', 'Observed top-1'),
                       ('log_error_observed', 'Error in latest log')]), '', 'These are not final results and do not enter paired gains.', '']
    lines += [markdown_table(summary, [('student', 'Student'), ('method', 'Method'), ('guide', 'Guide'), ('guide_depth', 'Depth'),
                                      ('guide_seed', 'Guide seed'), ('epochs', 'Epochs'), ('n_student_seeds', 'Student seeds'),
                                      ('mean_delta_pp', 'Mean gain (pp)'), ('sd_delta_pp', 'SD (pp)'), ('condition_id', 'Condition')]) if summary else 'No valid matched comparisons yet.', '',
              '## Depth contrasts', '', 'These contrasts compare adjacent available depths within the same guide architecture series, '
              'student seed and initial-state hash, guide seed, alignment settings, and downstream recipe. These direct guide-depth contrasts do not require a completed baseline. ViT widths and ResNet basic/bottleneck series are kept separate. '
              'No monotonicity or positive effect is assumed.', '']
    lines += [markdown_table(depths, [('method', 'Method'), ('guide_series', 'Series'), ('seed', 'Student seed'), ('guide_seed', 'Guide seed'),
                                     ('shallower_depth', 'Shallower'), ('deeper_depth', 'Deeper'), ('deeper_minus_shallower_pp', 'Deeper − shallower (pp)')]) if depths else 'No comparable depth pairs yet.', '',
              '## Diagnostics and limitations', '',
              'Alignment CSVs use the retained stage final state, not best_unmixed. Phase objectives change as endpoints are added; '
              'do not subtract phase-1 loss from phase-4 loss. Per-endpoint rank and RMS ratios are diagnostics, not post-hoc exclusion criteria.', '',
              'The v2 trainer writes the final result JSON only after training finishes. Missing results can mean pending, running, failed, or interrupted. '
              'Optional Slurm observations are partial progress, never substituted for final results. Each attempt is retained separately.', '',
              'A 100-epoch recipe and the first 100 epochs of a 300-epoch cosine schedule are not matched conditions. '
              'No confidence intervals, significance tests, compute-matched claims, or across-guide-seed pooled estimates are produced.', '',
              'Training loss epoch means are inferred only when the total minibatch count divides evenly by the requested epoch count. '
              'With compute_norms disabled, the zero-valued norm arrays are placeholders and are not interpreted as measurements.', '',
              'The code snapshot is the current checkout, not proof of the exact code executed by an earlier job. '
              'Dataset file contents, CUDA determinism, hardware, and image-validation sample counts are not verified by the result JSONs.', '',
              '## Warnings', '']
    lines += [markdown_table(issues, [('experiment', 'Experiment'), ('code', 'Code'), ('message', 'Detail')]) if issues else 'None.', '',
              '## Figures', '', str(len(plot_paths)) + ' PNG figures written under plots/. No model weights are loaded.', '']
    return '\n'.join(lines)

def bundle_results(root, output, runs, artifacts, observations, include_code, include_geometry = False):
    target = output / 'prior_results_bundle.zip'
    packages = {}
    for name in ['torch', 'torchvision', 'numpy', 'matplotlib', 'nvidia-dali-cuda120', 'nvidia-dali-cuda110']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    manifest = {'current_analysis_environment': packages, 'format_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
                'checkpoint_files_included': False, 'result_format': 'All validation values retained; minibatch arrays summarized',
                'source_code_is_current_checkout_not_runtime_proof': True,
                'file_hashes': 'Original local files at packaging time, not compacted or redacted archive entries',
                'text_redaction': 'Selected credential names only; inspect the archive before sharing', 'files': [],
                'checkpoints': {run['experiment']: run['checkpoints'] for run in runs}}
    with zipfile.ZipFile(target, 'w', compression = zipfile.ZIP_DEFLATED, compresslevel = 6) as archive:
        for path in artifacts:
            archive.write(path, 'analysis/' + str(path.relative_to(output)))
        for run in runs:
            prefix = 'logs/' + run['experiment'] + '/'
            for name, value in run['snapshots'].items():
                archive.writestr(prefix + name, json.dumps(clean(value), indent = 2, allow_nan = False))
            if run['result']:
                name = 'ft_results_seed' + str(run['seed']) + '.json'
                archive.writestr(prefix + name, json.dumps(clean(run['result']), indent = 2, allow_nan = False))
            for path in run['sources']:
                manifest['files'].append({'path': str(path.relative_to(root)), 'original_bytes': path.stat().st_size, 'original_sha256': file_hash(path)})
        archive.writestr('slurm_observations.json', json.dumps(clean({'attempts': observations}), indent = 2, allow_nan = False))
        for name in SOURCE_FILES:
            path = root / name
            if path.is_file():
                manifest['files'].append({'path': name, 'sha256': file_hash(path), 'included': include_code})
                if include_code:
                    archive.writestr('code/' + name, text_redact(path.read_text()))
        if include_geometry:
            for run in runs:
                path = root / 'geometry_audit' / (run['experiment'] + '.json')
                if path.is_file():
                    geometry = read_json(path)
                    if geometry.get('experiment') != run['experiment']:
                        raise ValueError('Geometry audit experiment mismatch: ' + str(path))
                    archive.writestr('geometry_audit/' + path.name, json.dumps(clean(geometry), indent = 2, allow_nan = False))
                    manifest['files'].append({'path': str(path.relative_to(root)), 'original_sha256': file_hash(path)})
        archive.writestr('manifest.json', json.dumps(clean(manifest), indent = 2, allow_nan = False))
        archive.writestr('README.txt', 'Upload this archive for analysis. No checkpoints, dataset images, environment variables, or W&B directories are included.\n'
                         'All validation curves and available alignment/argument metadata are included. Batchwise loss/norm arrays are summarized.\n'
                         'Paths, experiment names, optional log tails, and optional code remain visible; inspect before sharing.\n'
                         'To reanalyze, extract this archive and pass its directory to compare_prior.py --root.\n')
    return target

def analyze(args):
    root, output = Path(args.root).expanduser().resolve(), Path(args.output).expanduser().resolve()
    if output == root or output == root / 'logs' or root / 'logs' in output.parents or root / 'saved_models' == output or root / 'saved_models' in output.parents:
        raise ValueError('Choose an output directory outside logs/ and saved_models/ and not the repository root')
    output.mkdir(parents = True, exist_ok = True)
    previous = []
    manifest_path = output / 'generated_files.json'
    if manifest_path.is_file():
        try:
            previous = read_json(manifest_path).get('files', [])
        except (OSError, ValueError):
            previous = []
    runs = discover(root, args.include or ['*'], args.exclude)
    if not runs:
        raise ValueError('No optimization-prior run directories matched')
    match_baselines(runs)
    stages, endpoints, history = stage_tables(runs)
    summary, depths = condition_summary(runs), depth_comparisons(runs)
    observations = slurm_observations(root, runs) if args.include_slurm else []
    inventory = [inventory_row(run, observations) for run in runs]
    issues = [item for run in runs for item in run['issues']]
    tables = {'runs.csv': inventory, 'paired_comparisons.csv': [row for row in inventory if row['pair_status'] == 'matched'],
              'condition_summary.csv': summary, 'depth_comparisons.csv': depths, 'alignment_stages.csv': stages,
              'alignment_endpoints.csv': endpoints, 'alignment_history.csv': history, 'learning_curves.csv': curve_rows(runs),
              'progress.csv': progress_rows(runs, observations), 'warnings.csv': issues}
    artifacts = []
    for name, rows in tables.items():
        path = output / name
        write_csv(path, rows)
        artifacts.append(path)
    plot_paths = [] if args.no_plots else make_plots(runs, observations, output)
    if not args.no_plots and Figure is None:
        print('Matplotlib is unavailable; CSV/JSON/report output is complete, but no plots were generated.', file = sys.stderr)
    artifacts += plot_paths
    for name in previous:
        path = output / name
        if str(name).startswith('plots/') and path.suffix == '.png' and path.resolve().parent == output / 'plots' and path.is_file() and path not in plot_paths:
            path.unlink()
    provenance = {'recipes': {run['recipe_id']: run['recipe'] for run in runs if run['recipe_id']},
                  'alignment_settings': {run['alignment_id']: run['alignment_controls'] for run in runs if run.get('alignment_id')},
                  'runs': inventory, 'summaries': summary, 'depth_comparisons': depths, 'issues': issues,
                  'checkpoints': {run['experiment']: run['checkpoints'] for run in runs}}
    for name, value in [('comparison.json', provenance), ('slurm_observations.json', {'attempts': observations})]:
        path = output / name
        write_json(path, value)
        artifacts.append(path)
    report = output / 'report.md'
    report.write_text(report_text(runs, inventory, summary, depths, issues, plot_paths))
    artifacts.append(report)
    artifact_list = output / 'generated_files.json'
    write_json(artifact_list, {'files': [str(path.relative_to(output)) for path in artifacts]})
    artifacts.append(artifact_list)
    bundle = bundle_results(root, output, runs, artifacts, observations, args.include_code, getattr(args, 'include_geometry', False)) if args.bundle else None
    print(markdown_table(inventory, [('experiment', 'Experiment'), ('status', 'State'), ('final_top1', 'Final top-1'),
                                     ('delta_pp', 'Gain (pp)'), ('pair_status', 'Pairing')]))
    print('\nReport: ' + str(report))
    if bundle:
        print('Share: ' + str(bundle))
    return {'runs': runs, 'tables': tables, 'bundle': bundle, 'plots': plot_paths}

def get_args(argv = None):
    parser = argparse.ArgumentParser(description = 'Compare optimization-prior v2 runs without loading models or images')
    parser.add_argument('--root', default = '.')
    parser.add_argument('--output', default = 'prior_comparison')
    parser.add_argument('--include', action = 'append', default = None)
    parser.add_argument('--exclude', action = 'append', default = [])
    parser.add_argument('--no_plots', action = 'store_true')
    parser.add_argument('--include_slurm', action = 'store_true')
    parser.add_argument('--bundle', action = 'store_true')
    parser.add_argument('--include_code', action = 'store_true')
    parser.add_argument('--include_geometry', action = 'store_true')
    return parser.parse_args(argv)

if __name__ == '__main__':
    try:
        analyze(get_args())
    except (OSError, ValueError) as error:
        print(str(error), file = sys.stderr)
        sys.exit(1)
