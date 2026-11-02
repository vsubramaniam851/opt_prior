import argparse
import hashlib
import json
import math
import re
import torch
import torchvision

from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import DataLoader
from audit_prior_geometry import spectrum
from prior_utils import make_prior_model, prior_endpoints, prior_fingerprint, prior_forward, prior_mode, prior_view_geometry
from prior_views import PriorViewDataset, view_images, view_loss, view_spec, view_transform

def audit_view_rows(features, config):
    rows = []
    guide = features['guide']
    for state in ['random_student', 'aligned_student']:
        for endpoint in range(len(guide[0])):
            for descriptor, label in enumerate(['spatial_residual', 'global_diagnostic_only']):
                x = [features[state][v][endpoint][descriptor] for v in range(2)]
                y = [guide[v][endpoint][descriptor] for v in range(2)]
                metrics = prior_view_geometry(x, y, config)
                metrics['loss'] = view_loss(x, y, config).item()
                for kind, values in [('student', x), ('guide', y)]:
                    spectra = [spectrum(z) for z in values]
                    for key in spectra[0]:
                        metrics[kind + '_' + key] = sum(row[key] for row in spectra) / 2
                rows.append({'state': state, 'endpoint': endpoint + 1, 'descriptor': label,
                             'active_target': endpoint + 1 >= getattr(config, 'prior_start_stage', 1) and descriptor == 0, **metrics})
    return rows

def run_audit(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.experiment):
        raise ValueError('Use an experiment name, not a path')
    if args.workers < 0 or args.threads < 1 or args.samples < 4:
        raise ValueError('Invalid audit sample or worker count')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    torch.set_num_threads(args.threads)
    root = Path(args.root).expanduser().resolve()
    metadata = json.loads((root / 'logs' / args.experiment / 'prior_alignment.json').read_text())
    if metadata.get('version') != 2 or not metadata.get('complete') or not metadata.get('guide_unchanged') or metadata.get('baseline'):
        raise ValueError('Audit a completed, unchanged-guide two-view alignment')
    config = SimpleNamespace(**metadata['alignment_config'])
    if getattr(config, 'prior_view_mode', 'single') not in ['separate', 'shared'] or config.prior_descriptor != 'spatial_only':
        raise ValueError('This audit requires a spatial_only two-view initialization')
    if metadata.get('view_spec') != view_spec(config):
        raise ValueError('Recorded view recipe does not match this audit implementation')
    count = min(args.samples, len(metadata['validation_indices']))
    indices = metadata['validation_indices'][:count]
    if count < max(4, config.knn_k + 1) or len(set(indices)) != count:
        raise ValueError('Too few or repeated holdout image indices')
    data = Path(args.data_dir or config.data_dir).expanduser().resolve()
    dataset = torchvision.datasets.ImageFolder(data / 'train')
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError('Recorded holdout does not match the dataset')
    chunks = [indices[start:start + config.rep_batch] for start in range(0, count, config.rep_batch)]
    if len(chunks) > 1 and len(chunks[-1]) < max(4, config.knn_k + 1):
        chunks[-2].extend(chunks.pop())
    models = {'guide': make_prior_model(metadata['guide'], metadata['guide_seed'], metadata.get('guide_depth_override')),
              'random_student': make_prior_model(metadata['student'], metadata['student_seed']),
              'aligned_student': make_prior_model(metadata['student'], metadata['student_seed'])}
    checkpoint = root / 'saved_models' / ('orig_' + args.experiment + '.pt')
    models['aligned_student'].load_state_dict(torch.load(checkpoint, map_location = 'cpu', weights_only = True))
    hashes = {name: prior_fingerprint(model) for name, model in models.items()}
    expected = {'guide': metadata['guide_initial_hash'], 'random_student': metadata['student_initial_hash'],
                'aligned_student': metadata['student_aligned_hash']}
    if hashes != expected:
        raise ValueError('Checkpoint or random-model fingerprint mismatch; use the original training environment')
    device = torch.device(args.device)
    models = {name: model.requires_grad_(False).to(device) for name, model in models.items()}
    student_indices, guide_indices = prior_endpoints(models['guide'], models['random_student'], 3, 'cumulative')
    batches = []
    means = []
    for bank in [0, 1]:
        seed = config.prior_view_seed + bank * 1000003
        views = PriorViewDataset(dataset, view_transform(config), seed)
        loader = DataLoader(views, batch_sampler = chunks, num_workers = args.workers, pin_memory = args.device == 'cuda',
                            generator = torch.Generator().manual_seed(30000))
        sums = {}
        for batch_index, (images, _) in enumerate(loader):
            images = view_images(images, device)
            features = {}
            with torch.no_grad(), torch.autocast(device_type = device.type, dtype = torch.bfloat16,
                                                enabled = config.use_amp and device.type == 'cuda'):
                for state, model in models.items():
                    with prior_mode(model, config.prior_norm):
                        endpoints = guide_indices if state == 'guide' else student_indices
                        features[state] = [prior_forward(model, image, endpoints, config.pool_hw, 'spatial') for image in images]
            rows = audit_view_rows(features, config)
            n = images[0].size(0)
            for row in rows:
                key = (row['state'], row['endpoint'], row['descriptor'])
                identifiers = ['state', 'endpoint', 'descriptor', 'active_target']
                if key not in sums:
                    sums[key] = {key: row[key] for key in identifiers}
                for name, value in row.items():
                    if name not in identifiers:
                        if not math.isfinite(value):
                            raise FloatingPointError('Non-finite view-audit metric: ' + name)
                        sums[key][name] = sums[key].get(name, 0.0) + n * value
                batches.append({'bank': bank, 'view_seed': seed, 'batch': batch_index, 'samples': n, **row})
            print(f'View audit bank {bank}: {batch_index + 1}/{len(chunks)} batches', flush = True)
            del features, images
        for row in sums.values():
            means.append({'bank': bank, 'view_seed': seed, **{key: value if key in identifiers else value / count for key, value in row.items()}})
    if hashes != {name: prior_fingerprint(model) for name, model in models.items()}:
        raise RuntimeError('A model state changed during the read-only view audit')
    report = {'experiment': args.experiment, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
              'view_spec': metadata['view_spec'], 'samples_per_bank': count, 'batch_sizes': [len(x) for x in chunks],
              'holdout_indices_hash': hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
              'state_hashes': hashes, 'states_unchanged': True, 'classification_labels_used': False, 'classifiers_used': False,
              'data_dir': str(data), 'optimizer_steps': 0, 'rows': means, 'batch_rows': batches,
              'notes': ['Bank 0 reuses validation views; bank 1 uses different views of the same held-out images.',
                        'Bank 1 was not used by this run for checkpoint selection, but is not an independent image test set.',
                        'Rows average per-view spatial or global metrics separately; global loss is diagnostic only.',
                        'Kernel consistency does not establish coordinate-level feature invariance or downstream accuracy.',
                        'Means weight minibatches by image count; no full-dataset kernel or confidence interval is reported.']}
    output = Path(args.output) if args.output else root / 'view_audit' / (args.experiment + '.json')
    output.parent.mkdir(parents = True, exist_ok = True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent = 2, allow_nan = False))
    temporary.replace(output)
    print(str(output), flush = True)
    return report

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default = '.')
    parser.add_argument('--experiment', required = True)
    parser.add_argument('--data_dir', default = None)
    parser.add_argument('--output', default = None)
    parser.add_argument('--samples', type = int, default = 1024)
    parser.add_argument('--workers', type = int, default = 4)
    parser.add_argument('--threads', type = int, default = 4)
    parser.add_argument('--device', choices = ['cpu', 'cuda'], default = 'cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

if __name__ == '__main__':
    run_audit(get_args())
