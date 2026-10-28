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
from torchvision import transforms
from prior_utils import make_prior_model, prior_batch, prior_endpoints, prior_features, prior_fingerprint, prior_forward
from prior_utils import prior_geometry, prior_metric, prior_mode, prior_named_parameters, prior_parameter_diagnostics
from prior_utils import prior_schedule, prior_state, prior_trainable_parameters

def hybrid_states(initial, aligned, model):
    bn_names = {name for name, _, kind in prior_named_parameters(model) if kind.startswith('bn_')}
    return {'random_student': initial, 'aligned_student': aligned,
            'aligned_convs_initial_bn': {name: initial[name] if name in bn_names else value for name, value in aligned.items()},
            'initial_convs_aligned_bn': {name: aligned[name] if name in bn_names else value for name, value in initial.items()}}

def capture_endpoints(model, images, indices, args):
    captured = {}
    hooks = []
    def record(name):
        def hook(module, inputs, output):
            captured[name] = output.detach().clone()
        return hook
    def record_input(name):
        def hook(module, inputs):
            captured[name] = inputs[0].detach().clone()
        return hook
    for index in indices:
        last = getattr(model, f'layer{index + 1}')[-1]
        final_conv = last.conv3 if hasattr(last, 'conv3') else last.conv2
        hooks.append(final_conv.register_forward_hook(record(('pre_bn_branch', index))))
        hooks.append(last.relu.register_forward_pre_hook(record_input(('pre_relu_sum', index))))
    try:
        with prior_mode(model, args.prior_norm), torch.no_grad():
            post = prior_forward(model, images, indices)
        result = {'post_relu_stage': [prior_features(value, args.pool_hw, args.prior_descriptor) for value in post]}
        for location in ['pre_bn_branch', 'pre_relu_sum']:
            result[location] = [prior_features(captured[(location, index)], args.pool_hw, args.prior_descriptor) for index in indices]
        return result
    finally:
        for hook in hooks:
            hook.remove()

def gradient_probe(args, teacher, student, images, initial):
    stage = prior_schedule(teacher, student, args.prior_schedule, args.prior_start_stage, args.prior_update_scope)[-1]
    params = prior_trainable_parameters(student, stage[2], args.prior_bn_affine)
    ids = {id(param) for param in params}
    indices, targets = prior_endpoints(teacher, student, stage[0], args.prior_schedule, args.prior_start_stage)
    with prior_mode(teacher, args.prior_norm), prior_mode(student, args.prior_norm):
        loss, _, features, guide = prior_batch(args, teacher, student, images, indices, targets)
        losses = {'total': loss}
        for branch, label in enumerate(['spatial', 'global']):
            values = [prior_metric([x[branch]], [y[branch]], args) for x, y in zip(features, guide)]
            losses[label] = torch.stack(values).mean() / 2
        gradients = {}
        rows = {}
        for label, value in losses.items():
            gradients[label] = torch.autograd.grad(value, params, retain_graph = label != 'global', allow_unused = True)
            student.zero_grad(set_to_none = True)
            for param, grad in zip(params, gradients[label]):
                param.grad = grad
            rows[label] = {'weighted_loss': value.item(), 'groups': prior_parameter_diagnostics(student, initial)}
        conv_ids = {id(param) for _, param, kind in prior_named_parameters(student) if kind == 'conv'}
        pairs = [(a, b) for param, a, b in zip(params, gradients['spatial'], gradients['global'])
                 if id(param) in conv_ids and a is not None and b is not None]
        dot = sum((a.detach().float() * b.detach().float()).sum().item() for a, b in pairs)
        a2 = sum(a.detach().float().square().sum().item() for a, _ in pairs)
        b2 = sum(b.detach().float().square().sum().item() for _, b in pairs)
        cosine = dot / math.sqrt(a2 * b2) if a2 > 0 and b2 > 0 else None
        layers = []
        for name, param, kind in prior_named_parameters(student):
            if kind == 'conv' and id(param) in ids:
                index = next(i for i, value in enumerate(params) if value is param)
                grad = gradients['total'][index]
                layers.append({'name': name, 'connected': grad is not None,
                               'gradient_norm': grad.detach().float().norm().item() if grad is not None else None})
    student.zero_grad(set_to_none = True)
    return {'branches': rows, 'spatial_global_conv_gradient_cosine': cosine, 'convolution_gradients': layers}

def convolution_changes(initial, aligned, model):
    rows = []
    for name, param, kind in prior_named_parameters(model):
        if kind != 'conv' or param.dim() < 2:
            continue
        before = initial[name].float().flatten(1)
        after = aligned[name].float().flatten(1)
        norm0 = before.norm(dim = 1)
        norm1 = after.norm(dim = 1)
        ratios = norm1 / norm0.clamp_min(1e-12)
        cosine = (before * after).sum(dim = 1) / (norm0 * norm1).clamp_min(1e-12)
        rows.append({'name': name, 'filter_norm_ratio_min': ratios.min().item(),
                     'filter_norm_ratio_median': ratios.median().item(), 'filter_norm_ratio_max': ratios.max().item(),
                     'mean_filter_cosine': cosine.mean().item(),
                     'relative_change': ((after - before).norm() / before.norm().clamp_min(1e-12)).item()})
    return rows

def run_audit(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.experiment):
        raise ValueError('Use an experiment name, not a path')
    if args.threads < 1 or args.workers < 0 or args.samples < 1:
        raise ValueError('Invalid thread, worker or sample count')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    torch.set_num_threads(args.threads)
    root = Path(args.root).expanduser().resolve()
    metadata = json.loads((root / 'logs' / args.experiment / 'prior_alignment.json').read_text())
    if not metadata.get('complete') or not metadata.get('guide_unchanged') or metadata.get('student') != 'rn18' or metadata.get('baseline'):
        raise ValueError('Use a completed ResNet-18 alignment with an unchanged guide')
    config = dict(metadata['alignment_config'])
    if config.get('prior_descriptor') != 'spatial':
        raise ValueError('This diagnostic currently expects the two-branch spatial descriptor')
    config.setdefault('prior_bn_affine', 'train')
    config.setdefault('prior_start_stage', 1)
    config.setdefault('prior_update_scope', 'prefix')
    settings = SimpleNamespace(**config)
    device = torch.device(args.device)
    teacher = make_prior_model(metadata['guide'], metadata['guide_seed']).requires_grad_(False)
    student = make_prior_model(metadata['student'], metadata['student_seed'])
    initial = prior_state(student)
    hashes = {'guide': prior_fingerprint(teacher), 'random_student': prior_fingerprint(student)}
    checkpoint = root / 'saved_models' / ('orig_' + args.experiment + '.pt')
    student.load_state_dict(torch.load(checkpoint, map_location = 'cpu', weights_only = True))
    aligned = prior_state(student)
    hashes['aligned_student'] = prior_fingerprint(student)
    if hashes != {'guide': metadata['guide_initial_hash'], 'random_student': metadata['student_initial_hash'], 'aligned_student': metadata['student_aligned_hash']}:
        raise ValueError('Fingerprint mismatch; use the original training environment and checkpoints')
    buffer_names = {name for name, _ in student.named_buffers()}
    buffers_unchanged = all(torch.equal(initial[name], aligned[name]) for name in buffer_names)
    if not buffers_unchanged:
        raise ValueError('This audit requires unchanged initialization buffers')
    states = hybrid_states(initial, aligned, student)
    changes = prior_parameter_diagnostics(student, initial)
    teacher.to(device)
    student.to(device)
    indices = metadata['validation_indices'][:min(args.samples, len(metadata['validation_indices']))]
    minimum = max(4, settings.knn_k + 1)
    if len(indices) < minimum or len(indices) != len(set(indices)):
        raise ValueError('Too few images or duplicate holdout indices')
    data = Path(args.data_dir or settings.data_dir).expanduser().resolve()
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset = torchvision.datasets.ImageFolder(data / 'train', transform = transform)
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError('The recorded holdout does not match this dataset')
    chunks = [indices[i:i + settings.rep_batch] for i in range(0, len(indices), settings.rep_batch)]
    if len(chunks) > 1 and len(chunks[-1]) < minimum:
        chunks[-2].extend(chunks.pop())
    loader = DataLoader(dataset, batch_sampler = chunks, num_workers = args.workers,
                        generator = torch.Generator().manual_seed(30000), pin_memory = args.device == 'cuda')
    si, ti = prior_endpoints(teacher, student, 3, settings.prior_schedule, settings.prior_start_stage)
    rows = []
    probes = []
    for batch_index, (images, _) in enumerate(loader):
        images = images.to(device)
        targets = capture_endpoints(teacher, images, ti, settings)
        for state, weights in states.items():
            student.load_state_dict(weights)
            student.requires_grad_(False)
            features = capture_endpoints(student, images, si, settings)
            for location, endpoints in features.items():
                for index, x, y in zip(si, endpoints, targets[location]):
                    for branch in range(2):
                        geometry = prior_geometry([x[branch]], [y[branch]], settings.knn_k)
                        loss = prior_metric([x[branch]], [y[branch]], settings).item()
                        rows.append({'batch': batch_index, 'samples': images.size(0), 'state': state, 'location': location,
                                     'endpoint': index + 1, 'descriptor': branch + 1, 'loss': loss, **geometry})
            if state in ['random_student', 'aligned_student']:
                probes.append({'batch': batch_index, 'state': state, 'samples': images.size(0),
                               **gradient_probe(settings, teacher, student, images, initial)})
        print(f'Mechanism audit: {batch_index + 1}/{len(chunks)} batches', flush = True)
    student.load_state_dict(aligned)
    student.zero_grad(set_to_none = True)
    if prior_fingerprint(teacher) != hashes['guide'] or prior_fingerprint(student) != hashes['aligned_student']:
        raise RuntimeError('The read-only mechanism audit changed a model state')
    result = {'experiment': args.experiment, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
              'samples': len(indices), 'batch_sizes': [len(chunk) for chunk in chunks],
              'holdout_indices_hash': hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
              'classification_labels_used': False, 'classifiers_used': False, 'optimizer_steps': 0,
              'state_hashes': hashes, 'states_unchanged': True, 'initialization_buffers_unchanged': buffers_unchanged,
              'parameter_changes_by_type': changes, 'convolution_changes': convolution_changes(initial, aligned, student),
              'measurements': rows, 'gradient_probes': probes,
              'notes': ['Gradients are measured at the initialization, not reconstructed from the original alignment trajectory.',
                        'Hybrid resets are diagnostic and are never saved as training checkpoints.',
                        'Hybrid effects are nonlinear and are not an additive attribution of the alignment gain.',
                        'The last convolution is before its own BN but still depends on upstream normalization.',
                        'The pre-ReLU sum includes the skip connection; the pre-BN branch does not.',
                        'Each spatial/global gradient is weighted by one half, as in the actual loss.',
                        'Gradient squared fractions describe raw derivatives, not Adam update contributions.',
                        'These images belong to the existing alignment holdout; they are not a new independent test.']}
    output = Path(args.output) if args.output else root / 'mechanism_audit' / (args.experiment + '.json')
    output.parent.mkdir(parents = True, exist_ok = True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent = 2, allow_nan = False))
    temporary.replace(output)
    print(str(output), flush = True)
    return result

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default = '.')
    parser.add_argument('--experiment', required = True)
    parser.add_argument('--data_dir', default = None)
    parser.add_argument('--output', default = None)
    parser.add_argument('--samples', type = int, default = 256)
    parser.add_argument('--workers', type = int, default = 4)
    parser.add_argument('--threads', type = int, default = 4)
    parser.add_argument('--device', choices = ['cpu', 'cuda'], default = 'cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

if __name__ == '__main__':
    run_audit(get_args())
