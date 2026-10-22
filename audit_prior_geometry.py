import argparse
import gc
import hashlib
import json
import math
import re
import torch
import torch.nn.functional as F
import torchvision

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from prior_utils import make_prior_model, prior_endpoints, prior_features, prior_fingerprint, prior_forward
from prior_utils import prior_geometry, prior_mode
from rep_sims import CKA

def spectrum(features):
    x = features.double() - features.double().mean(dim = 0, keepdim = True)
    norm = x.norm()
    if norm.item() == 0:
        return {'participation_rank': 0.0, 'top1_variance_fraction': 0.0, 'top5_variance_fraction': 0.0}
    x = x / norm
    values = torch.linalg.eigvalsh(x @ x.T).clamp_min(0)
    total = values.sum().clamp_min(1e-15)
    return {'participation_rank': (total.square() / values.square().sum().clamp_min(1e-15)).item(),
            'top1_variance_fraction': (values[-1] / total).item(),
            'top5_variance_fraction': (values[-5:].sum() / total).item()}

def image_descriptors(images):
    return {'image_color_statistics': torch.cat([images.mean(dim = (2, 3)), images.std(dim = (2, 3), unbiased = False)], dim = 1),
            'image_coarse_pixels': F.adaptive_avg_pool2d(images, 4).flatten(1)}

def describe_model(model, images, indices, normalization, pool_hw):
    with prior_mode(model, normalization), torch.no_grad():
        outputs = prior_forward(model, images, indices)
        result = {mode: [prior_features(z, pool_hw, mode) for z in outputs] for mode in ['pooled', 'spatial']}
        if not hasattr(model, 'layer1'):
            base = model.model if hasattr(model, 'model') else model
            result['normalized_cls'] = [[base.encoder.ln(z[:, 0])] for z in outputs]
    return result

def collect_rows(features, images, k):
    nuisance = image_descriptors(images)
    metric = CKA(images.device, unbiased = True)
    rows = []
    teacher = features['guide']
    for state, modes in features.items():
        for mode, endpoints in modes.items():
            for endpoint, descriptors in enumerate(endpoints):
                for descriptor, x in enumerate(descriptors):
                    y = teacher[mode][endpoint][descriptor]
                    geometry = prior_geometry([x], [y], k)
                    row = {'state': state, 'descriptor_mode': mode, 'endpoint': endpoint + 1, 'descriptor': descriptor + 1,
                           **geometry, **spectrum(x)}
                    for name, values in nuisance.items():
                        row[name + '_ucka'] = metric.linear_CKA(x, values).item()
                    row['permuted_guide_ucka'] = metric.linear_CKA(x, y.roll(1, dims = 0)).item()
                    if state == 'aligned_student':
                        initial = features['random_student'][mode][endpoint][descriptor]
                        row['initial_student_ucka'] = metric.linear_CKA(x, initial).item()
                    rows.append(row)
    return rows

def audit(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.experiment):
        raise ValueError('Use an experiment name, not a path')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    if args.threads < 1 or args.workers < 0:
        raise ValueError('Invalid thread or worker count')
    torch.set_num_threads(args.threads)
    root = Path(args.root).expanduser().resolve()
    folder = root / 'logs' / args.experiment
    metadata = json.loads((folder / 'prior_alignment.json').read_text())
    if metadata.get('version') != 2 or not metadata.get('complete') or not metadata.get('guide_unchanged') or metadata.get('baseline'):
        raise ValueError('Audit a completed, unchanged-guide, non-baseline v2 alignment')
    config = metadata['alignment_config']
    batch = config['rep_batch']
    count = min(args.samples, len(metadata['validation_indices']))
    if count < max(4, config['knn_k'] + 1):
        raise ValueError('Too few images for the recorded CKA and KNN settings')
    indices = metadata['validation_indices'][:count]
    if len(set(indices)) != len(indices):
        raise ValueError('The recorded holdout has duplicate image indices')
    data = Path(args.data_dir or config['data_dir']).expanduser().resolve()
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset = torchvision.datasets.ImageFolder(data / 'train', transform = transform)
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError('The recorded holdout does not fit this training dataset')
    chunks = [indices[start:start + batch] for start in range(0, count, batch)]
    if len(chunks) > 1 and len(chunks[-1]) < max(4, config['knn_k'] + 1):
        chunks[-2].extend(chunks.pop())
    loader = DataLoader(dataset, batch_sampler = chunks, num_workers = args.workers,
                        pin_memory = args.device == 'cuda', generator = torch.Generator().manual_seed(30000))
    device = torch.device(args.device)
    models = {'guide': make_prior_model(metadata['guide'], metadata['guide_seed'], metadata['guide_depth_override']),
              'random_student': make_prior_model(metadata['student'], metadata['student_seed']),
              'aligned_student': make_prior_model(metadata['student'], metadata['student_seed'])}
    checkpoint = root / 'saved_models' / ('orig_' + args.experiment + '.pt')
    models['aligned_student'].load_state_dict(torch.load(checkpoint, map_location = 'cpu', weights_only = True))
    expected = {'guide': metadata['guide_initial_hash'], 'random_student': metadata['student_initial_hash'],
                'aligned_student': metadata['student_aligned_hash']}
    for state, model in models.items():
        if prior_fingerprint(model) != expected[state]:
            raise ValueError(f'{state} fingerprint mismatch; use the training environment and original initialization checkpoint')
    if args.control_guide:
        models['same_depth_guide'] = make_prior_model(metadata['student'], metadata['guide_seed'])
    hashes = {name: prior_fingerprint(model) for name, model in models.items()}
    models = {name: model.requires_grad_(False).to(device) for name, model in models.items()}
    student_indices, guide_indices = prior_endpoints(models['guide'], models['random_student'], 3, 'cumulative')
    sums = {}
    total = 0
    for batch_index, (images, _) in enumerate(loader):
        images = images.to(device, non_blocking = True)
        with torch.no_grad():
            features = {state: describe_model(model, images, guide_indices if state == 'guide' else student_indices,
                                               metadata['normalization'], config['pool_hw']) for state, model in models.items()}
            rows = collect_rows(features, images, config['knn_k'])
        for row in rows:
            key = (row['state'], row['descriptor_mode'], row['endpoint'], row['descriptor'])
            if key not in sums:
                sums[key] = {name: row[name] for name in ['state', 'descriptor_mode', 'endpoint', 'descriptor']}
            for name, value in row.items():
                if name not in ['state', 'descriptor_mode', 'endpoint', 'descriptor']:
                    if not math.isfinite(value):
                        raise FloatingPointError('Non-finite geometry audit metric: ' + name)
                    sums[key][name] = sums[key].get(name, 0.0) + value * images.size(0)
        total += images.size(0)
        print(f'Geometry audit: {batch_index + 1}/{len(chunks)} batches, {total}/{count} images', flush = True)
        del features, rows, images
    final_hashes = {name: prior_fingerprint(model) for name, model in models.items()}
    if hashes != final_hashes:
        raise RuntimeError('A model state changed during the read-only geometry audit')
    rows = []
    for row in sums.values():
        rows.append({name: value if name in ['state', 'descriptor_mode', 'endpoint', 'descriptor'] else value / total
                     for name, value in row.items()})
    report = {'experiment': args.experiment, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
              'data_dir': str(data), 'device': str(device), 'normalization': metadata['normalization'],
              'images': 'held-out images from the training split; labels discarded; official validation unused',
              'samples': total, 'batch_sizes': [len(chunk) for chunk in chunks], 'holdout_indices': indices,
              'holdout_indices_hash': hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
              'state_hashes': hashes, 'states_unchanged': True, 'classifiers_used': False,
              'guide_endpoints': [index + 1 for index in guide_indices],
              'notes': ['Metrics are sample-weighted means of fixed-minibatch statistics, not one full-dataset kernel.',
                        'Descriptors are recorded separately; low rank here is not proof of whole-network collapse.',
                        'normalized_cls uses the existing final LayerNorm at every endpoint; only the final endpoint is the actual readout.',
                        'Color-statistics or coarse-pixel agreement is a diagnostic association, not proof of a causal shortcut.',
                        'No ImageNet classification loss, labels, optimizer or parameter updates are used.'],
              'rows': rows}
    output = Path(args.output) if args.output else root / 'geometry_audit' / (args.experiment + '.json')
    output.parent.mkdir(parents = True, exist_ok = True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent = 2, allow_nan = False))
    temporary.replace(output)
    del models
    gc.collect()
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
    parser.add_argument('--control_guide', action = 'store_true')
    return parser.parse_args()

if __name__ == '__main__':
    audit(get_args())
