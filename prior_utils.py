import contextlib
import hashlib
import json
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from rep_sims import CKA, DifferentiableKNN
from vits import ViT

def seed_prior(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def validate_prior_args(args):
    if not args.optimization_prior:
        return
    if args.pretrained or args.task_loss or args.distillation or args.preserve_alignment:
        raise ValueError('The optimization prior requires --untrained and no task loss or distillation')
    if args.setting != 'imagenet' or args.imagenet != 'imagenet-1k':
        raise ValueError('The optimization prior preset expects ImageNet-1K')
    if args.replacement_type not in ['progressive_align_resnet', 'progressive_align_vits']:
        raise ValueError('Use progressive_align_resnet or progressive_align_vits')
    if args.joint_rn or args.joint_vit or args.indices or args.train_context_blocks:
        raise ValueError('Use --prior_schedule cumulative or frozen instead of legacy joint/context flags')
    if args.tune_linear or args.fc_only or args.freeze_bn_stats_align:
        raise ValueError('Use full downstream training and select alignment normalization with --prior_norm')
    if args.reload_progressive or args.continue_ft or args.bit8_adam or args.measure_alignment:
        raise ValueError('This preset supports --reload_ft, but not legacy continuation or alignment diagnostics')
    if args.prior_schedule not in ['cumulative', 'frozen']:
        raise ValueError('prior_schedule must be cumulative or frozen')
    if args.repdist not in ['CKA', 'UCKA', 'KNN']:
        raise ValueError('The optimization prior requires CKA, UCKA or KNN')
    if args.repdist == 'KNN' and args.knn_loss != 'soft':
        raise ValueError('Use --knn_loss soft for the optimization prior')
    if args.repdist != 'KNN' and args.cka_loss not in ['normal', 'debiased']:
        raise ValueError('Use --cka_loss debiased or normal, not a logarithm of signed CKA')
    if args.rep_batch < max(4, args.knn_k + 1) or args.rep_val_size < args.rep_batch:
        raise ValueError('Use rep_val_size >= rep_batch > knn_k and rep_batch >= 4')
    if args.knn_k < 1 or args.knn_temp <= 0 or args.rep_val_interval < 1 or args.num_workers < 0:
        raise ValueError('Invalid temperature, validation interval or worker count')
    if not 0 <= args.align_mix <= 1 or not 0 <= args.prior_min_rank_ratio <= 1:
        raise ValueError('align_mix and prior_min_rank_ratio must lie in [0, 1]')
    if not args.progressive_lrs or not args.progressive_epochs:
        raise ValueError('Specify at least one progressive learning rate and epoch count')
    if min(args.progressive_lrs) <= 0 or min(args.progressive_epochs) < 1:
        raise ValueError('Progressive learning rates and epoch counts must be positive')
    if args.max_stage_steps is not None and args.max_stage_steps < 1:
        raise ValueError('max_stage_steps must be positive')
    if not args.align_only and args.ft_epochs < 1:
        raise ValueError('Use --align_only instead of zero downstream epochs')
    if args.ft_batch < 1 or args.ft_accum < 1 or args.ft_lr <= 0:
        raise ValueError('Downstream batch size, accumulation and learning rate must be positive')
    if args.use_amp and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise ValueError('This AMP preset requires bfloat16 support; omit --use_amp on this GPU')

def prior_config(args):
    names = ['replacement_type', 'student_vit', 'model', 'seed', 'teacher_seed', 'teacher_depth', 'baseline', 'data_dir',
             'repdist', 'cka_loss', 'knn_loss', 'knn_k', 'knn_temp', 'rep_batch', 'rep_val_size', 'rep_split_seed',
             'rep_val_interval', 'max_stage_steps', 'progressive_lrs', 'progressive_epochs', 'rep_wd', 'rep_scheduler',
             'pool_hw', 'prior_schedule', 'align_mix', 'prior_min_rank_ratio', 'prior_norm', 'use_amp', 'num_workers',
             'anchor_to_init', 'anchor_lambda']
    config = {name: getattr(args, name, None) for name in names}
    if config['data_dir'] is not None:
        config['data_dir'] = os.path.realpath(os.path.expanduser(config['data_dir']))
    return config

def check_prior_run(args):
    if not args.optimization_prior:
        return
    checkpoint = f'saved_models/orig_{args.exp_name}.pt'
    metadata_path = f'logs/{args.exp_name}/prior_alignment.json'
    if not args.reload_ft:
        paths = [checkpoint, metadata_path, f'logs/{args.exp_name}/args.json']
        if any(os.path.exists(path) for path in paths):
            raise FileExistsError('This experiment already exists; use --reload_ft for training or choose a new --exp_name')
        return
    if not os.path.isfile(checkpoint) or not os.path.isfile(metadata_path):
        raise FileNotFoundError('Run alignment first with the same experiment name and configuration')
    with open(metadata_path) as f:
        metadata = json.load(f)
    if metadata.get('version') != 2 or not metadata.get('complete'):
        raise ValueError('The initialization is not a completed version 2 run')
    config = prior_config(args)
    mismatched = [name for name, value in config.items() if metadata['alignment_config'].get(name) != value]
    if mismatched:
        raise ValueError(f'Alignment configuration differs from the checkpoint: {mismatched}')
    if os.path.exists(f'saved_models/ft_final_{args.exp_name}.pt'):
        raise FileExistsError('Downstream training already completed for this experiment')

def check_prior_checkpoint(args, student):
    with open(f'logs/{args.exp_name}/prior_alignment.json') as f:
        metadata = json.load(f)
    if prior_fingerprint(student) != metadata['student_aligned_hash']:
        raise ValueError('The initialization checkpoint does not match its alignment metadata')

def make_prior_model(name, seed, depth = None):
    with torch.random.fork_rng(devices = []):
        torch.random.default_generator.manual_seed(seed)
        if name in ['rn18', 'rn34', 'rn50', 'rn101', 'rn152']:
            if depth is not None:
                raise ValueError('teacher_depth is only used for ViTs')
            return getattr(torchvision.models, 'resnet' + name[2:])(weights = None)
        layouts = {'vit_tiny': (192, 4, 3), 'vit_small': (384, 8, 6),
                   'vit_base': (768, 12, 12), 'vit_large': (1024, 24, 16)}
        if name not in layouts:
            raise ValueError(f'Unsupported prior architecture: {name}')
        width, layers, heads = layouts[name]
        if depth is not None:
            if depth < 1:
                raise ValueError('teacher_depth must be positive')
            layers = depth
        return ViT(hidden_dim = width, num_layers = layers, num_heads = heads, mlp_dim = width * 4)

def prior_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

def prior_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

@contextlib.contextmanager
def prior_mode(model, normalization = 'batch'):
    modes = [(module, module.training) for module in model.modules()]
    batchnorms = [(module, module.track_running_stats) for module in model.modules()
                 if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    model.eval()
    if normalization == 'batch':
        for module, _ in batchnorms:
            module.track_running_stats = False
            module.training = True
    try:
        yield
    finally:
        for module, tracking in batchnorms:
            module.track_running_stats = tracking
        for module, training in modes:
            module.training = training

def prior_parts(model):
    if hasattr(model, 'layer1'):
        stem = [model.conv1, model.bn1]
        return stem, [getattr(model, f'layer{i}') for i in range(1, 5)], []
    model = model.model if hasattr(model, 'model') else model
    return [model.conv_proj], list(model.encoder.layers), [model.class_token, model.encoder.pos_embedding]

def prior_schedule(teacher, student, schedule = 'cumulative'):
    if schedule not in ['cumulative', 'frozen']:
        raise ValueError('Unknown progressive alignment schedule')
    stem, blocks, global_params = prior_parts(student)
    _, teacher_blocks, _ = prior_parts(teacher)
    if len(teacher_blocks) < len(blocks):
        raise ValueError('The guide must have at least as many segments as the student')
    stages = []
    for index, block in enumerate(blocks):
        target = math.ceil((index + 1) * len(teacher_blocks) / len(blocks)) - 1
        modules = stem + blocks[:index + 1] if schedule == 'cumulative' else (stem + [block] if index == 0 else [block])
        params = [p for module in modules for p in module.parameters()]
        if schedule == 'cumulative' or index == 0:
            params += global_params
        stages.append((index, target, params))
    return stages

def prior_endpoints(teacher, student, index, schedule):
    _, blocks, _ = prior_parts(student)
    _, teacher_blocks, _ = prior_parts(teacher)
    indices = list(range(index + 1)) if schedule == 'cumulative' else [index]
    targets = [math.ceil((i + 1) * len(teacher_blocks) / len(blocks)) - 1 for i in indices]
    return indices, targets

def prior_forward(model, images, indices, pool_hw = None):
    single = isinstance(indices, int)
    indices = [indices] if single else list(indices)
    _, blocks, _ = prior_parts(model)
    if not indices or indices != sorted(set(indices)) or indices[0] < 0 or indices[-1] >= len(blocks):
        raise ValueError('Endpoints must be sorted, unique and within the model')
    outputs = []
    if hasattr(model, 'layer1'):
        x = model.maxpool(model.relu(model.bn1(model.conv1(images))))
    else:
        model = model.model if hasattr(model, 'model') else model
        x = model._process_input(images)
        x = torch.cat([model.class_token.expand(x.size(0), -1, -1), x], dim = 1)
        x = model.encoder.dropout(x + model.encoder.pos_embedding)
    for index, block in enumerate(blocks[:indices[-1] + 1]):
        x = block(x)
        if index in indices:
            outputs.append(prior_features(x, pool_hw) if pool_hw is not None else x)
    return outputs[0] if single else outputs

def prior_features(z, pool_hw = 4):
    z = z.float()
    if z.dim() == 4:
        width = min(pool_hw, z.size(-2), z.size(-1)) if pool_hw > 0 else min(4, z.size(-2), z.size(-1))
        spatial = F.adaptive_avg_pool2d(z, width).flatten(1)
        return [spatial, z.mean(dim = (2, 3))]
    if z.dim() == 3:
        return [z[:, 0], z[:, 1:].mean(dim = 1)]
    return [z.flatten(1)]

def prior_metric(features, targets, args):
    if args.repdist == 'KNN':
        metric = DifferentiableKNN(features[0].device, args.knn_k, temperature = args.knn_temp)
        values = [metric.diff_knn_distribution_loss(x, y) for x, y in zip(features, targets)]
    else:
        metric = CKA(features[0].device, unbiased = args.repdist == 'UCKA' or args.cka_loss == 'debiased')
        values = [1.0 - metric.linear_CKA(x, y) for x, y in zip(features, targets)]
    return torch.stack(values).mean()

def prior_geometry(features, targets, k):
    values = {'cka': [], 'ucka': [], 'knn_overlap': [], 'student_rank': [], 'teacher_rank': [],
              'student_centered_rms': [], 'teacher_centered_rms': []}
    for x, y in zip(features, targets):
        x = x.float() - x.float().mean(dim = 0, keepdim = True)
        y = y.float() - y.float().mean(dim = 0, keepdim = True)
        values['cka'].append(CKA(x.device).linear_CKA(x, y).item())
        values['ucka'].append(CKA(x.device, unbiased = True).linear_CKA(x, y).item())
        for name, z in [('student_rank', x), ('teacher_rank', y)]:
            rms = z.square().mean().sqrt()
            values[name.replace('_rank', '_centered_rms')].append(rms.item())
            z = z / rms.clamp_min(1e-12)
            gram = z @ z.T / z.size(1)
            values[name].append((gram.trace().square() / gram.square().sum().clamp_min(1e-12)).item())
        x = F.normalize(x, dim = 1)
        y = F.normalize(y, dim = 1)
        mask = torch.eye(x.size(0), device = x.device, dtype = torch.bool)
        a = (x @ x.T).masked_fill(mask, -torch.inf).topk(k, dim = 1).indices
        b = (y @ y.T).masked_fill(mask, -torch.inf).topk(k, dim = 1).indices
        values['knn_overlap'].append((a.unsqueeze(2) == b.unsqueeze(1)).any(dim = 2).float().mean().item())
    return {name: float(np.mean(value)) for name, value in values.items()}

def make_prior_loaders(args):
    root = os.path.join(args.data_dir, 'train')
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_set = torchvision.datasets.ImageFolder(root, transform = train_transform)
    val_set = torchvision.datasets.ImageFolder(root, transform = val_transform)
    if args.rep_val_size > len(train_set) - args.rep_batch:
        raise ValueError('The alignment holdout leaves too few training images')
    order = torch.randperm(len(train_set), generator = torch.Generator().manual_seed(args.rep_split_seed)).tolist()
    val_indices = order[:args.rep_val_size]
    train_indices = order[args.rep_val_size:]
    workers = args.num_workers
    train_loader = DataLoader(Subset(train_set, train_indices), batch_size = args.rep_batch, shuffle = True,
                              num_workers = workers, pin_memory = True, drop_last = True,
                              generator = torch.Generator().manual_seed(args.seed + 10000), persistent_workers = workers > 0)
    chunks = [val_indices[i:i + args.rep_batch] for i in range(0, len(val_indices), args.rep_batch)]
    if len(chunks[-1]) < max(4, args.knn_k + 1):
        chunks[-2].extend(chunks.pop())
    val_loader = DataLoader(val_set, batch_sampler = chunks, num_workers = workers, pin_memory = True,
                            generator = torch.Generator().manual_seed(args.rep_split_seed + 30000), persistent_workers = workers > 0)
    return train_loader, val_loader, val_indices

def prior_log(values):
    if wandb is not None and wandb.run is not None:
        wandb.log(values)

def prior_batch(args, teacher, student, images, student_indices, teacher_indices):
    with torch.autocast(device_type = images.device.type, dtype = torch.bfloat16, enabled = args.use_amp and images.device.type == 'cuda'):
        with torch.no_grad():
            targets = prior_forward(teacher, images, teacher_indices, args.pool_hw)
        features = prior_forward(student, images, student_indices, args.pool_hw)
    losses = [prior_metric(x, y, args) for x, y in zip(features, targets)]
    return torch.stack(losses).mean(), losses, features, targets

def prior_evaluate(args, teacher, student, loader, student_index, teacher_index, device):
    student_indices, teacher_indices = prior_endpoints(teacher, student, student_index, args.prior_schedule)
    if teacher_indices[-1] != teacher_index:
        raise ValueError('Guide endpoint does not match the progressive schedule')
    sums = {}
    count = 0
    with torch.random.fork_rng(devices = []), prior_mode(teacher, args.prior_norm), prior_mode(student, args.prior_norm), torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking = True)
            loss, losses, features, targets = prior_batch(args, teacher, student, images, student_indices, teacher_indices)
            metrics = {'loss': loss.item()}
            for index, endpoint_loss, x, y in zip(student_indices, losses, features, targets):
                geometry = prior_geometry(x, y, args.knn_k)
                geometry['loss'] = endpoint_loss.item()
                for name, value in geometry.items():
                    metrics[f'endpoint{index + 1}_{name}'] = value
                    if name != 'loss':
                        metrics[name] = metrics.get(name, 0.0) + value / len(student_indices)
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + value * images.size(0)
            count += images.size(0)
    if not count:
        raise ValueError('The alignment validation loader is empty')
    metrics = {name: value / count for name, value in sums.items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError('Non-finite alignment validation geometry')
    metrics['samples'] = count
    return metrics

def prior_rank_ok(args, metrics, initial):
    if args.prior_min_rank_ratio == 0:
        return True
    names = [name for name in initial if name.startswith('endpoint') and name.endswith('_student_rank')]
    return all(metrics[name] >= args.prior_min_rank_ratio * initial[name] for name in names)

def align_prior_stage(args, teacher, student, train_loader, val_loader, stage, device):
    index, target_index, params = stage
    student_indices, teacher_indices = prior_endpoints(teacher, student, index, args.prior_schedule)
    student.requires_grad_(False)
    teacher.requires_grad_(False)
    student.zero_grad(set_to_none = True)
    for p in params:
        p.requires_grad_(True)
    initial = prior_state(student)
    initial_metrics = prior_evaluate(args, teacher, student, val_loader, index, target_index, device)
    best_metrics = dict(initial_metrics)
    best_state = initial
    best_step = 0
    lr = args.progressive_lrs[min(index, len(args.progressive_lrs) - 1)]
    epochs = args.progressive_epochs[min(index, len(args.progressive_epochs) - 1)]
    steps = args.max_stage_steps if args.max_stage_steps is not None else epochs * len(train_loader)
    if steps < 1:
        raise ValueError('The alignment training loader is empty')
    optimizer = torch.optim.AdamW(params, lr = lr, weight_decay = args.rep_wd)
    history = [{'step': 0, **initial_metrics}]
    prefix = f'prior/stage{index + 1}'
    prior_log({f'{prefix}/val_{name}': value for name, value in initial_metrics.items()})
    initial_loss = initial_metrics['loss']
    print(f'{prefix}: {args.prior_schedule}, endpoints {[(i + 1, j + 1) for i, j in zip(student_indices, teacher_indices)]}, initial loss {initial_loss:.6f}')
    iterator = iter(train_loader)
    with prior_mode(teacher, args.prior_norm), prior_mode(student, args.prior_norm):
        for step in range(steps):
            try:
                images, _ = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                images, _ = next(iterator)
            images = images.to(device, non_blocking = True)
            if args.rep_scheduler:
                warmup = max(1, int(steps * 0.1))
                factor = 0.1 + 0.9 * step / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
                for group in optimizer.param_groups:
                    group['lr'] = lr * factor
            loss, endpoint_losses, _, _ = prior_batch(args, teacher, student, images, student_indices, teacher_indices)
            if args.anchor_to_init:
                anchor = torch.stack([(p - initial[name].to(device)).square().mean() for name, p in student.named_parameters() if p.requires_grad]).mean()
                loss = loss + args.anchor_lambda * anchor
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite alignment loss at stage {index + 1}, step {step + 1}')
            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite = True)
            optimizer.step()
            if (step + 1) % 20 == 0:
                prior_log({f'{prefix}/train_loss': loss.item(), f'{prefix}/grad_norm': grad_norm.item(), f'{prefix}/lr': optimizer.param_groups[0]['lr'],
                           **{f'{prefix}/train_endpoint{i + 1}_loss': value.item() for i, value in zip(student_indices, endpoint_losses)}})
            if (step + 1) % args.rep_val_interval == 0 or step + 1 == steps:
                metrics = prior_evaluate(args, teacher, student, val_loader, index, target_index, device)
                rank_ok = prior_rank_ok(args, metrics, initial_metrics)
                if rank_ok and metrics['loss'] < best_metrics['loss']:
                    best_metrics = metrics
                    best_state = prior_state(student)
                    best_step = step + 1
                history.append({'step': step + 1, 'rank_ok': rank_ok, **metrics})
                prior_log({f'{prefix}/val_{name}': value for name, value in metrics.items()})
                val_loss = metrics['loss']
                print(f'{prefix}: step {step + 1}/{steps}, val loss {val_loss:.6f}, best step {best_step}')
    student.load_state_dict(best_state)
    if args.align_mix < 1.0:
        mixed = {name: initial[name] + args.align_mix * (value - initial[name]) if value.is_floating_point() else value
                 for name, value in best_state.items()}
        student.load_state_dict(mixed)
    final_metrics = prior_evaluate(args, teacher, student, val_loader, index, target_index, device)
    accepted = final_metrics['loss'] < initial_metrics['loss'] and prior_rank_ok(args, final_metrics, initial_metrics)
    if not accepted:
        student.load_state_dict(initial)
        final_metrics = dict(initial_metrics)
    delta = sum((value.detach().cpu() - initial[name]).float().square().sum().item()
                for name, value in student.named_parameters())
    norm = sum(initial[name].float().square().sum().item() for name, p in student.named_parameters() if p.requires_grad)
    student.zero_grad(set_to_none = True)
    summary = {'student_endpoint': index + 1, 'teacher_endpoint': target_index + 1, 'steps': steps,
               'schedule': args.prior_schedule, 'active_endpoints': [[i + 1, j + 1] for i, j in zip(student_indices, teacher_indices)],
               'trainable_parameters': sum(p.numel() for p in params), 'rank_gate_enabled': args.prior_min_rank_ratio > 0,
               'align_mix': args.align_mix,
               'best_step': best_step, 'accepted': accepted, 'relative_parameter_change': math.sqrt(delta / max(norm, 1e-12)),
               'initial': initial_metrics, 'best_unmixed': best_metrics, 'final': final_metrics, 'history': history}
    prior_log({f'{prefix}/accepted': int(accepted), f'{prefix}/relative_parameter_change': summary['relative_parameter_change'],
               **{f'{prefix}/selected_{name}': value for name, value in final_metrics.items()}})
    return summary

def initialize_optimization_prior(args, teacher, device, initialize_only = False):
    student_name = 'rn18' if args.replacement_type == 'progressive_align_resnet' else args.student_vit
    student = make_prior_model(student_name, args.seed).to(device)
    if hasattr(teacher, 'layer1') != hasattr(student, 'layer1'):
        raise ValueError('The student and guide must belong to the same architecture family')
    os.makedirs(f'logs/{args.exp_name}', exist_ok = True)
    os.makedirs(f'saved_models/{args.exp_name}', exist_ok = True)
    metadata = {'student': student_name, 'guide': args.model, 'student_seed': args.seed, 'guide_seed': args.teacher_seed,
                'guide_depth_override': args.teacher_depth, 'student_initial_hash': prior_fingerprint(student),
                'guide_initial_hash': prior_fingerprint(teacher), 'normalization': args.prior_norm,
                'version': 2, 'schedule': args.prior_schedule, 'align_mix': args.align_mix,
                'rank_gate_enabled': args.prior_min_rank_ratio > 0, 'prior_min_rank_ratio': args.prior_min_rank_ratio,
                'selection_metric': 'held-out mean active-endpoint representation loss',
                'classification_labels_used': False, 'alignment_config': prior_config(args),
                'complete': False, 'stages': {}, 'baseline': args.baseline}
    if initialize_only:
        if args.baseline and not args.reload_ft:
            metadata.update({'complete': True, 'student_aligned_hash': metadata['student_initial_hash'],
                             'guide_final_hash': metadata['guide_initial_hash'], 'guide_unchanged': True})
            with open(f'logs/{args.exp_name}/prior_alignment.json', 'w') as f:
                json.dump(metadata, f, indent = 2)
        return student
    seed_prior(args.seed + 10000)
    train_loader, val_loader, indices = make_prior_loaders(args)
    metadata['validation_indices'] = indices
    metadata['validation_source'] = 'held-out ImageNet training images, excluded from alignment updates'
    for stage in prior_schedule(teacher, student, args.prior_schedule):
        index = stage[0]
        metadata['stages'][f'stage{index + 1}'] = align_prior_stage(args, teacher, student, train_loader, val_loader, stage, device)
        torch.save(student.state_dict(), f'saved_models/{args.exp_name}/prior_stage{index + 1}.pt')
        with open(f'logs/{args.exp_name}/prior_alignment.json', 'w') as f:
            json.dump(metadata, f, indent = 2)
    metadata['guide_final_hash'] = prior_fingerprint(teacher)
    metadata['guide_unchanged'] = metadata['guide_initial_hash'] == metadata['guide_final_hash']
    if not metadata['guide_unchanged']:
        raise RuntimeError('The supposedly frozen guide state changed during alignment')
    metadata['student_aligned_hash'] = prior_fingerprint(student)
    metadata['complete'] = True
    with open(f'logs/{args.exp_name}/prior_alignment.json', 'w') as f:
        json.dump(metadata, f, indent = 2)
    student.requires_grad_(True)
    student.zero_grad(set_to_none = True)
    return student
