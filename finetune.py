import os
import math
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import contextlib
import bitsandbytes.optim as bnb_optim
import torchvision
import copy

from tqdm import tqdm
from rep_sims import CKA

from dataloader import get_imagenet_dataloaders, cifar_dataset, get_text_dataloaders, get_text_embedding_dataloaders, mixup_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_model_cka(model1, model2, dataloader, device, img, preserve_alignment = True):
    cka_computer = CKA(device)
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    
    activations1 = {}
    activations2 = {}
    hooks = []
    for name in layer_names:
        m1_layer = dict(model1.named_modules()).get(name)
        m2_layer = dict(model2.named_modules()).get(name)
        if m1_layer is None or m2_layer is None:
            continue
        hooks.append(m1_layer.register_forward_hook(
            lambda m, i, o, n=name: activations1.__setitem__(n, o)))
        hooks.append(m2_layer.register_forward_hook(
            lambda m, i, o, n=name: activations2.__setitem__(n, o)))
    
    model1.eval()
    model2.eval()
    
    cka_scores = {name: [] for name in layer_names}
    
    activations1.clear()
    activations2.clear()
    
    with torch.no_grad():
        model2(img)
    
    if preserve_alignment:
        model1(img)
    else:
        with torch.no_grad():
            model1(img)
    
    for name in layer_names:
        if name in activations1 and name in activations2:
            f1 = activations1[name].reshape(img.size(0), -1).float()
            f2 = activations2[name].reshape(img.size(0), -1).float()
            cka_val = cka_computer.linear_CKA(f1, f2)
            cka_scores[name].append(cka_val)

    for h in hooks:
        h.remove()
    
    sim_dict =  {name: torch.mean(torch.stack(scores)) if scores else 0.0 for name, scores in cka_scores.items()}
    return torch.mean(torch.stack(list(sim_dict.values())))

def adjust_learning_rate(lr, optimizer, epoch):
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_weight_decay(weight_decay, optimizer, epoch):
    if epoch == 25 or epoch == 50:
        weight_decay = weight_decay * 0.5
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = weight_decay
        
def compute_grad_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm

def compute_weight_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm

def overfit_batch(replaced_model, train_loader, device):
    # before full training
    data = next(iter(train_loader))
    imgs = data[0]['images'].permute(0,3,1,2).to(device); labels = data[0]['labels'].squeeze().long().to(device)
    for p in replaced_model.parameters(): p.requires_grad=True
    opt = optim.SGD(replaced_model.parameters(), lr=1.0, momentum=0.9)
    for i in range(2000):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = replaced_model(imgs)
            loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward(); opt.step()
        if i % 20 == 0: print(i, loss.item())

def overfit_one_batch_strict(model, train_loader, device, steps=3000, lr=5e-6):
    model.to(device)
    model.train()

    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.p = 0.0
            m.train(False)  # disable dropout
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.train(False)           # freeze running stats
            m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)

    batch = next(iter(train_loader))
    imgs = batch[0]['images'].permute(0, 3, 1, 2).contiguous().to(device)
    labels = batch[0]['labels'].squeeze().long().to(device)

    criterion = nn.CrossEntropyLoss()
    optim_params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(optim_params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    model.zero_grad(set_to_none=True)
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(imgs) 
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()
        if (t % 20) == 0:
            print(t, float(loss))

def validate(step, lin_conv_model, val_loader, loss_fn, use_amp = False):
    lin_conv_model.eval()
    val_loss = 0.0
    top1_total = 0.0
    top5_total = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc = 'Iterating over validation set...', dynamic_ncols = True):
            img = batch[0]['images'].permute(0, 3, 1, 2).to(device, non_blocking = True)
            label = batch[0]['labels'].reshape(-1).long().to(device, non_blocking = True)
            with torch.autocast(device_type = device.type, dtype = torch.bfloat16, enabled = use_amp and device.type == 'cuda'):
                preds = lin_conv_model(img)
                loss = loss_fn(preds, label)
            acc1, acc5 = accuracy(preds, label, topk = (1, 5))
            val_loss += loss.item() * img.size(0)
            top1_total += acc1.item() * img.size(0)
            top5_total += acc5.item() * img.size(0)
            total_samples += img.size(0)
    if total_samples == 0:
        raise ValueError('The validation loader is empty')
    avg_val_loss = val_loss / total_samples
    top1_acc = top1_total / total_samples
    top5_acc = top5_total / total_samples
    print(f'Step {step + 1}, Validation Loss {avg_val_loss}, Top-1 Accuracy {top1_acc}, Top-5 Accuracy {top5_acc}, Samples {total_samples}')
    wandb.log({'top1_acc': top1_acc, 'top5_acc': top5_acc, 'val_loss': avg_val_loss, 'val_samples': total_samples})
    return avg_val_loss, top1_acc, top5_acc

def recalibrate_bn(model, loader, num_batches=200, use_amp = False):
    if use_amp:
        autocast = torch.autocast('cuda', dtype = torch.bfloat16)
    else:
        autocast = contextlib.nullcontext()
    model.train()
    saved = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            saved.append((m, m.momentum))
            m.momentum = None  # cumulative moving average
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc = 'Recalibrating batch statistics...', dynamic_ncols = True)):
            if i >= num_batches: break
            x = batch[0]['images'].permute(0,3,1,2).to(next(model.parameters()).device)
            with autocast:
                model(x)
    for m, mom in saved:
        m.momentum = mom

def compute_layer_grad_stats(model):
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None and 'conv' in name and 'weight' in name:
            grad_norm = param.grad.norm().item()
            weight_norm = param.data.norm().item()
            relative_grad = grad_norm / (weight_norm + 1e-8)
            grad_stats[name] = {
                'grad_norm': grad_norm,
                'weight_norm': weight_norm,
                'relative_grad': relative_grad
            }
    return grad_stats

def normalize_conv_weights(aligned_model, reference_model):
    aligned_sd = aligned_model.state_dict()
    ref_sd = reference_model.state_dict()
    
    for name in aligned_sd:
        if 'conv' in name and 'weight' in name:
            aligned_std = aligned_sd[name].std().item()
            ref_std = ref_sd[name].std().item()
            if aligned_std > 1e-8:
                scale = ref_std / aligned_std
                aligned_sd[name] = aligned_sd[name] * scale
                print(f'{name}: aligned_std={aligned_std:.4f}, ref_std={ref_std:.4f}, scale={scale:.4f}')
    
    aligned_model.load_state_dict(aligned_sd)
    return aligned_model

def freeze_bn_eval(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

def imagenet_finetune(args, replaced_model, orig):
    if args.use_amp and device.type == 'cuda':
        autocast = torch.autocast('cuda', dtype = torch.bfloat16)
    else:
        autocast = contextlib.nullcontext()
    data_seed = args.seed + 20000 if args.optimization_prior else args.seed
    train_loader, val_loader = get_imagenet_dataloaders(args.imagenet, batch_size = args.ft_batch, data_dir = args.data_dir, seed = data_seed)
    orig_top1_acc = None
    if args.imagenet == 'imagenet-21k':
        run_eval = False
        replaced_model.fc = nn.Linear(replaced_model.fc.in_features, 21000).to(device)
    else:
        run_eval = True
    # if args.replacement_type == 'progressive_align_resnet' and not args.pretrained and not args.baseline:
    #     reference_model = torchvision.models.resnet18(pretrained=False)
    #     replaced_model = normalize_conv_weights(replaced_model, reference_model)
    if run_eval and not args.optimization_prior:
        orig_top1_acc, orig_top5_acc = eval_loop(orig, val_loader, device)
        print('ORIGINAL ACCURACY', orig_top1_acc)
    if not args.optimization_prior and any(isinstance(m, nn.BatchNorm2d) for m in replaced_model.modules()) and args.replacement_type in ['progressive', 'progressive_align_resnet']:
        replaced_model.train()
        recalibrate_bn(replaced_model, train_loader, use_amp = args.use_amp, num_batches = 5003)
    if args.measure_alignment:
        random_baseline = torchvision.models.resnet18(pretrained = False).to(device)
        random_baseline.eval()
        initial_aligned_state = copy.deepcopy(replaced_model.state_dict())
    if args.tune_linear:
        if args.model == 'rn18':
            for p in replaced_model.modules():
                if isinstance(p, nn.Linear): 
                    if p.out_features != 1000:
                        p.requires_grad = True
                elif isinstance(p, nn.BatchNorm2d):
                    p.requires_grad = True
        else:
            for n, p in replaced_model.named_parameters():
                p.requires_grad = any(k in n for k in ('head', 'fc', 'classifier')) or ('norm' in n or 'bn' in n)
    elif args.fc_only:
        print('ONLY TUNING TOP LINEAR LAYER...')
        for p in replaced_model.parameters():
            p.requires_grad = False
        replaced_model.fc.requires_grad_(True)

    else:
        for p in replaced_model.parameters(): 
            p.requires_grad = True
    print('Number of tunable parameters...', sum(p.numel() for p in replaced_model.parameters() if p.requires_grad))
    loss_fn = nn.CrossEntropyLoss(label_smoothing = args.label_smoothing)
    if args.ft_optimizer == 'bit8_adam':
        optimizer = bnb_optim.AdamW8bit(
            [
                {'params': [p for n,p in replaced_model.named_parameters() if p.ndim > 1 and not any(t in n for t in ('pos_embedding', 'class_token'))], 
                'weight_decay': args.ft_wd},
                {'params': [p for n,p in replaced_model.named_parameters() if p.ndim <= 1 or any(t in n for t in ('pos_embedding', 'class_token'))],
                'weight_decay': 0.0},

            ],
            lr = args.ft_lr, betas = (0.9, 0.95), eps = 1e-6 , weight_decay = args.ft_wd
        )
    elif args.ft_optimizer == 'bit8_lamb':
        optimizer = bnb_optim.LAMB8bit(
            [
                {'params': [p for n,p in replaced_model.named_parameters()
                            if p.ndim > 1 and not any(t in n for t in ('pos_embedding', 'class_token'))],
                'weight_decay': args.ft_wd},
                {'params': [p for n,p in replaced_model.named_parameters()
                            if p.ndim <= 1 or any(t in n for t in ('pos_embedding', 'class_token'))],
                'weight_decay': 0.0},
            ],
            lr=args.ft_lr, betas=(0.9, 0.98), eps=1e-6
        )
    elif args.ft_optimizer == 'lamb':
        optimizer = bnb_optim.LAMB32bit(
            [
                {'params': [p for n,p in replaced_model.named_parameters()
                            if p.ndim > 1 and not any(t in n for t in ('pos_embedding', 'class_token'))],
                'weight_decay': args.ft_wd},
                {'params': [p for n,p in replaced_model.named_parameters()
                            if p.ndim <= 1 or any(t in n for t in ('pos_embedding', 'class_token'))],
                'weight_decay': 0.0},
            ],
            lr=args.ft_lr, betas=(0.9, 0.98), eps=1e-6
        )
    elif args.ft_optimizer == 'sgd':
        optimizer = optim.SGD(
            [
                {'params':  [p for n,p in replaced_model.named_parameters() if p.ndim > 1 and not any(t in n for t in ('pos_embedding', 'class_token'))],
                 'weight_decay': args.ft_wd},
                {'params': [p for n,p in replaced_model.named_parameters() if p.ndim <= 1 or any(t in n for t in ('pos_embedding', 'class_token'))], 
                 'weight_decay': 0.0}
            ],
            lr=args.ft_lr, momentum=0.9, nesterov=True
        )
    else:
        optimizer = optim.AdamW(
        [
           {'params': [p for n, p in replaced_model.named_parameters() if p.ndim > 1 and not any(t in n for t in ('pos_embedding', 'class_token'))],
            'weight_decay': args.ft_wd},
           {'params': [p for n, p in replaced_model.named_parameters() if p.ndim <= 1 or any(t in n for t in ('pos_embedding', 'class_token'))],
           'weight_decay': 0.0},
        ],
        lr = args.ft_lr
        )
    accum_steps = args.ft_accum
    if accum_steps < 1 or len(train_loader) < 1:
        raise ValueError('ft_accum and the number of training batches must be positive')
    optimizer.zero_grad(set_to_none = True)
    if args.use_scheduler:
        total_steps = math.ceil(len(train_loader) / accum_steps) * args.ft_epochs
        if args.scheduler_type == 'cosine': 
            print('Building cosine scheduler...')
            num_batches = len(train_loader)
            steps_per_epoch = math.ceil(num_batches / accum_steps)
            total_steps  = args.ft_epochs * steps_per_epoch
            warmup_steps = min(args.warmup_epochs * steps_per_epoch, max(0, total_steps - 1))
            cosine_steps = max(1, total_steps - warmup_steps)
            sched_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, total_iters = max(1, warmup_steps))
            sched_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cosine_steps, eta_min = 1e-5)  
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers = [sched_warmup, sched_cosine], milestones = [warmup_steps]) if warmup_steps else sched_cosine
            if not warmup_steps:
                for group in optimizer.param_groups:
                    group['lr'] = args.ft_lr
        elif args.scheduler_type == 'cosine_flat': 
            print('Building cosine flat scheduler...')
            num_batches = len(train_loader)
            steps_per_epoch = math.ceil(num_batches / accum_steps)
            total_steps  = args.ft_epochs * steps_per_epoch
            warmup_steps = min(args.warmup_epochs * steps_per_epoch, max(0, total_steps - 1))
            flat_steps = args.flat_epochs * steps_per_epoch
            cosine_steps  = total_steps - warmup_steps - flat_steps
            sched_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor = 1e-3, end_factor = 1.0, total_iters = max(1, warmup_steps))
            sched_flat  = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=flat_steps)
            sched_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cosine_steps, eta_min = 1e-6)  
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers = [sched_warmup, sched_flat, sched_cosine], 
                                                        milestones = [warmup_steps, warmup_steps + flat_steps])
        elif args.scheduler_type == 'onecycle':
            print('Building onecycle scheduler...')
            max_lr = args.ft_lr
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = max_lr, total_steps = total_steps, pct_start = 0.1, anneal_strategy = 'cos', 
                                                            div_factor=10, final_div_factor=1000)
        elif args.scheduler_type == 'lambda':
            print('Building lambda scheduler...')
            warmup_steps = int(0.05 * total_steps)
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps              
                q = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * q))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 10)
    else:
        scheduler = None
    replaced_model = replaced_model.to(device)
    val_losses = []
    train_losses = []
    top1_accs = []
    top5_accs = []
    weight_norms = []
    grad_norms = []
    log_json = {}
    for epoch in range(args.ft_epochs):
        if scheduler == None and not args.fc_only:
            adjust_learning_rate(args.ft_lr, optimizer, epoch)
        train_loss = 0.0
        train_samples = 0
        for i, batch in enumerate(tqdm(train_loader, desc = 'Iterating over training set...', dynamic_ncols = True)):
            if args.evaluate == 'steps' and run_eval:
                if (epoch * len(train_loader) + i) % args.eval_steps == 0:
                    step = i + len(train_loader) * epoch
                    avg_val_loss, top1_acc, top5_acc = validate(step, replaced_model, val_loader, loss_fn, use_amp=args.use_amp)
                    if not val_losses or avg_val_loss <= min(val_losses):
                        torch.save(replaced_model.state_dict(), f'saved_models/ft_{args.exp_name}.pt')  
                    val_losses.append(avg_val_loss)
                    top1_accs.append(top1_acc)
                    top5_accs.append(top5_acc)
                    if scheduler and args.scheduler_type == 'plateau':
                        scheduler.step(avg_val_loss)
                    replaced_model.train()
            img, label = batch[0]['images'].to(device, non_blocking=True), batch[0]['labels'].reshape(-1).long().to(device, non_blocking=True)
            replaced_model.train()
            # freeze_bn_eval(replaced_model)
            img = img.permute(0, 3, 1, 2)
            if args.mixup_alpha > 0:
                img, y_a, y_b, lam = mixup_batch(img, label, args.mixup_alpha)
                with autocast:
                    preds = replaced_model(img)
                loss = lam * loss_fn(preds.float(), y_a) + (1.0 - lam) * loss_fn(preds.float(), y_b)
            else:
                with autocast:
                    preds = replaced_model(img)
                loss = loss_fn(preds.float(), label)
            group_size = min(accum_steps, len(train_loader) - (i // accum_steps) * accum_steps)
            loss = loss / group_size
            if args.measure_alignment:
                replaced_model.eval()
                temp_initial = torchvision.models.resnet18(pretrained = False).to(device)
                temp_initial.load_state_dict(initial_aligned_state)
                cka_with_initial = compute_model_cka(replaced_model, temp_initial, train_loader, device, img, preserve_alignment = False)
                cka_with_random = compute_model_cka(replaced_model, random_baseline, train_loader, device, img, preserve_alignment = False)
                wandb.log({'cka_with_initial': cka_with_initial, 'cka_with_random': cka_with_random})
                replaced_model.train()
            if args.preserve_alignment:
                # if i % 100 == 0:
                temp_initial = torchvision.models.resnet18(pretrained = False).to(device)
                temp_initial.load_state_dict(initial_aligned_state)
                rep_sim_loss = compute_model_cka(replaced_model, temp_initial, train_loader, device, img, preserve_alignment = True)
                wandb.log({'rep_sim_loss': rep_sim_loss})
                loss = loss + args.rep_sim_weight * (1 - rep_sim_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite downstream loss at epoch {epoch + 1}, batch {i + 1}')
            loss.backward()
            if args.compute_norms:
                grad_norm = compute_grad_norm(replaced_model)
                weight_norm = compute_weight_norm(replaced_model)
                weight_norms.append(weight_norm)
                grad_norms.append(grad_norm)
                if i % 100 == 0:
                    grad_stats = compute_layer_grad_stats(replaced_model)
                    for layer_name, stats in grad_stats.items():
                        short_name = layer_name.replace('.weight', '')
                        wandb.log({
                            f'layer_grad_norm/{short_name}': stats['grad_norm'],
                            f'layer_relative_grad/{short_name}': stats['relative_grad'] 
                        })
            else:
                weight_norms.append(0.0)
                grad_norms.append(0.0)
            train_losses.append(loss.item() * group_size)
            if (i + 1) % accum_steps == 0 or i + 1 == len(train_loader):
                if args.ft_clip > 0:
                    nn.utils.clip_grad_norm_([p for p in replaced_model.parameters() if p.requires_grad], max_norm = args.ft_clip, error_if_nonfinite = True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler and args.scheduler_type != 'plateau':
                    scheduler.step()
            if i % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({'train_loss': np.mean(train_losses[-20:]), 'current_lr': current_lr,
                           'weight_norm': np.mean(weight_norms[-20:]), 'grad_norm': np.mean(grad_norms[-20:])})
            train_loss += loss.item() * group_size * img.shape[0]
            train_samples += img.size(0)
        avg_train_loss = train_loss / train_samples
        print(f'Epoch {epoch + 1}, Training Loss {avg_train_loss}')
        if args.evaluate == 'epoch' and run_eval:
            avg_val_loss, top1_acc, top5_acc = validate(epoch, replaced_model, val_loader, loss_fn, use_amp=args.use_amp)
            val_losses.append(avg_val_loss)
            top1_accs.append(top1_acc)
            top5_accs.append(top5_acc)
            if avg_val_loss <= min(val_losses):
                torch.save(replaced_model.state_dict(), f'saved_models/ft_{args.exp_name}.pt') 
            if scheduler and args.scheduler_type == 'plateau':
                scheduler.step(avg_val_loss)
    if run_eval and (args.evaluate != 'epoch' or not top1_accs):
        avg_val_loss, top1_acc, top5_acc = validate(args.ft_epochs - 1, replaced_model, val_loader, loss_fn, use_amp = args.use_amp)
        if not val_losses or avg_val_loss <= min(val_losses):
            torch.save(replaced_model.state_dict(), f'saved_models/ft_{args.exp_name}.pt')
        val_losses.append(avg_val_loss)
        top1_accs.append(top1_acc)
        top5_accs.append(top5_acc)
    torch.save(replaced_model.state_dict(), f'saved_models/ft_final_{args.exp_name}.pt')
    final_top1 = top1_accs[-1] if top1_accs else None
    best_top1 = max(top1_accs) if top1_accs else None
    log_json = {'train_losses': train_losses, 'val_losses': val_losses, 'top1_accs': top1_accs, 'top5_accs': top5_accs,
                'grad_norms': grad_norms, 'weight_norms': weight_norms, 'final_top1': final_top1, 'best_top1': best_top1}
    if run_eval:
        wandb.log({'final_top1': final_top1, 'best_top1': best_top1})
    return replaced_model, final_top1, log_json, orig_top1_acc

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = torch.topk(output, maxk, dim = 1, largest = True, sorted = True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def eval_loop(model, val_loader, device, use_amp = False):
    model.eval()
    top1_acc = 0
    top5_acc = 0
    total_samples = 0
    if use_amp:
        autocast = torch.autocast('cuda', dtype = torch.bfloat16)
    else:
        autocast = contextlib.nullcontext()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc = 'Iterating over test batches...', dynamic_ncols = True):
            img, label = batch[0]['images'].to(device, non_blocking=True), batch[0]['labels'].long().to(device, non_blocking=True)
            img = img.permute(0, 3, 1, 2)
            with autocast:
                outputs = model(img)
            acc1, acc5 = accuracy(outputs, label, topk=(1, 5))
            top1_acc += acc1.item() * img.size(0)
            top5_acc += acc5.item() * img.size(0)
            total_samples += img.size(0)
    top1_acc /= total_samples
    top5_acc /= total_samples
    return top1_acc, top5_acc

def eval_accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def cifar_finetune(exp_name, model, orig):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = cifar_dataset()
    accs = []
    ckpt_count = 0
    for epoch in range(1):
        epoch_losses = []
        model.train()
        for i, batch in enumerate(tqdm(train_loader, desc = 'Iterating over CIFAR')):
            x, y = batch
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        print(f'Epoch {epoch+1}: train loss = {np.mean(epoch_losses):.3f}')
        acc = eval_accuracy(model, test_loader)
        accs.append(acc)
        print(f'Epoch {epoch+1}: test acc = {acc:.3f}')
    orig_acc = eval_accuracy(orig, test_loader)
    return max(accs), orig_acc

def eval_ppl_runner(model, test_loader, device):
    criterion = nn.CrossEntropyLoss(ignore_index = -1)
    model = model.eval()
    total_loss = 0.0
    for batch in tqdm(test_loader, desc = 'Evaluating perplexity...'):
        inputs = {'input_ids': batch['input_ids'].to(device)}
        with torch.no_grad():
            lm_logits = model(**inputs).logits
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = inputs['input_ids'][..., 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.item()
        total_loss += loss
    average_nll = total_loss / len(test_loader)
    word_nll = average_nll * test_loader.dataset.dataset.numel() / test_loader.dataset.num_words
    return math.exp(word_nll)

def shift_logits_labels(input_ids, lm_logits):
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = input_ids[..., 1:].contiguous()
    shift_labels = shift_labels.view(-1)
    return shift_logits, shift_labels

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def wikitext_finetune(args, rnn_model, orig):
    print(args.ft_batch)
    train_loader, valid_loader, test_loader, vocab_size = get_text_dataloaders(args.model, args.ft_batch, args.num_workers, args.seq_len)
    if args.tune_rnn:
        for p in rnn_model.modules():
            if isinstance(p, nn.RNN): 
                p.requires_grad = True
            elif isinstance(p, nn.LayerNorm):
                p.requires_grad = True

    else:
        for p in rnn_model.parameters(): 
            p.requires_grad = True

    loss_fn = nn.CrossEntropyLoss(ignore_index = -100, label_smoothing = 0.1)
    optimizer = optim.AdamW(rnn_model.parameters(), lr = args.ft_lr, weight_decay = args.ft_wd)

    if args.use_scheduler:
        if args.scheduler_type == 'cosine': 
            steps_per_epoch = len(train_loader)
            total_steps  = args.ft_epochs * steps_per_epoch
            warmup_steps  = args.warmup_epochs * steps_per_epoch
            cosine_steps  = total_steps - warmup_steps
            sched_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, end_factor = 1.0, total_iters = warmup_steps)
            sched_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cosine_steps)  
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers = [sched_warmup, sched_cosine], milestones = [warmup_steps])
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 10)
    else:
        scheduler = None

    rnn_model = rnn_model.to(device)
    val_losses = []
    train_losses = []
    perplexities = []
    log_json = {}

    for epoch in range(args.ft_epochs):
        rnn_model.eval()
        valid_loss = 0.0
        hidden = None
        for batch in tqdm(valid_loader, desc = 'Iterating over validation set...'):
            inputs = {'input_ids': batch['input_ids'].to(device)}
            with torch.no_grad():
                lm_logits = rnn_model(**inputs).logits
            shift_logits, shift_labels = shift_logits_labels(inputs['input_ids'], lm_logits)
            loss = loss_fn(shift_logits, shift_labels)
            valid_loss += loss.item()
        avg_val_loss = valid_loss / len(valid_loader)
        wandb.log({'val_loss': avg_val_loss})
        val_losses.append(avg_val_loss)
        if avg_val_loss <= min(val_losses):
            torch.save(rnn_model.state_dict(), f'saved_models/ft_{args.exp_name}.pt')
        print(f'Epoch {epoch + 1}, Validation Loss {avg_val_loss}')

        perplexity = eval_ppl_runner(rnn_model, test_loader, device)
        print(f'Wikitext-103 Test Perplexity {perplexity}')
        perplexities.append(perplexity)

        rnn_model.train()
        train_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader, desc = 'Iterating over train loader', dynamic_ncols = True)):
            inputs = {'input_ids': batch['input_ids'].to(device)}
            lm_logits = rnn_model(**inputs).logits
            shift_logits, shift_labels = shift_logits_labels(inputs['input_ids'], lm_logits)
            loss = loss_fn(shift_logits, shift_labels)
            train_losses.append(loss.item())
            if i % 20 == 0:
                avg_train_loss = np.mean(train_losses[-20:])
                wandb.log({'ce_loss': loss})
            before_update_params = {name: param.clone() for name, param in rnn_model.named_parameters()}
            loss.backward()   
            grad_norm = get_grad_norm(rnn_model)
            wandb.log({'grad_norm': grad_norm})
            nn.utils.clip_grad_norm_(rnn_model.parameters(), max_norm = 1.0)
            if (i + 1) % args.accumulation == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Training Loss {avg_train_loss}')

    perplexity = eval_ppl_runner(rnn_model, test_loader, device)
    print(f'Wikitext-103 Test Perplexity {perplexity}')
    perplexities.append(perplexity)

    log_json = {'train_losses': train_losses, 'val_losses': val_losses, 'test_perplexity': perplexities}
    if not os.path.exists(f'logs/{args.exp_name}'):
        os.makedirs(f'logs/{args.exp_name}')
    with open(f'logs/{args.exp_name}/ft_results.json', 'w') as f:
        json.dump(log_json, f)

    orig_perplexity = 37.50
    return rnn_model, min(perplexities), log_json, orig_perplexity

def _move_text_batch(batch, prefix, device):
    inputs = {
        'input_ids': batch[f'{prefix}_input_ids'].to(device),
        'attention_mask': batch[f'{prefix}_attention_mask'].to(device),
    }
    tok_key = f'{prefix}_token_type_ids'
    if tok_key in batch:
        inputs['token_type_ids'] = batch[tok_key].to(device)
    return inputs

def _pool_bert_output(outputs, inputs, pooling = 'cls'):
    hidden = outputs.last_hidden_state
    if pooling == 'mean':
        mask = inputs['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim = 1).clamp_min(1.0)
        emb = (hidden * mask).sum(dim = 1) / denom
    else:
        emb = hidden[:, 0]
    return F.normalize(emb.float(), p = 2, dim = 1)

def _bert_embeddings(model, inputs, pooling = 'cls'):
    outputs = model(**inputs, return_dict = True)
    return _pool_bert_output(outputs, inputs, pooling = pooling)

def _build_retrieval_docs(model, batch, device, pooling = 'cls', use_hard_negatives = False):
    pos_inputs = _move_text_batch(batch, 'positive', device)
    pos_emb = _bert_embeddings(model, pos_inputs, pooling = pooling)
    doc_embs = [pos_emb]
    if use_hard_negatives and 'negative_input_ids' in batch:
        neg_inputs = _move_text_batch(batch, 'negative', device)
        neg_emb = _bert_embeddings(model, neg_inputs, pooling = pooling)
        doc_embs.append(neg_emb)
    return torch.cat(doc_embs, dim = 0)

def text_embedding_loss(model, batch, device, pooling = 'cls', temperature = 0.05, use_hard_negatives = False):
    query_inputs = _move_text_batch(batch, 'query', device)
    query_emb = _bert_embeddings(model, query_inputs, pooling = pooling)
    doc_emb = _build_retrieval_docs(model, batch, device, pooling = pooling, use_hard_negatives = use_hard_negatives)
    scores = torch.matmul(query_emb, doc_emb.t()) / temperature
    labels = torch.arange(query_emb.size(0), device = device)
    loss = F.cross_entropy(scores, labels)
    return loss, scores

def eval_text_embedding_retrieval(model, val_loader, device, args):
    was_training = model.training
    model.eval()
    query_embs = []
    doc_embs = []
    pos_doc_ids = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc = 'Evaluating retrieval...', dynamic_ncols = True):
            query_inputs = _move_text_batch(batch, 'query', device)
            q = _bert_embeddings(model, query_inputs, pooling = args.embedding_pool)
            docs = _build_retrieval_docs(model, batch, device, pooling = args.embedding_pool, use_hard_negatives = args.embedding_use_hard_negatives)
            start = sum(x.size(0) for x in doc_embs)
            query_embs.append(q.cpu())
            doc_embs.append(docs.cpu())
            pos_doc_ids.extend([start + i for i in range(q.size(0))])
    if was_training:
        model.train()
    if len(query_embs) == 0:
        return {'mrr@10': 0.0, 'recall@1': 0.0, 'recall@5': 0.0, 'recall@10': 0.0}

    query_embs = torch.cat(query_embs, dim = 0)
    doc_embs = torch.cat(doc_embs, dim = 0)
    scores = torch.matmul(query_embs, doc_embs.t())
    ranks = torch.argsort(scores, dim = 1, descending = True)
    pos = torch.tensor(pos_doc_ids, dtype = torch.long)

    rr10 = []
    recall1 = []
    recall5 = []
    recall10 = []
    for i in range(ranks.size(0)):
        hit = (ranks[i] == pos[i]).nonzero(as_tuple = False)
        if hit.numel() == 0:
            rank = ranks.size(1) + 1
        else:
            rank = int(hit[0, 0].item()) + 1
        rr10.append(1.0 / rank if rank <= 10 else 0.0)
        recall1.append(1.0 if rank <= 1 else 0.0)
        recall5.append(1.0 if rank <= 5 else 0.0)
        recall10.append(1.0 if rank <= 10 else 0.0)

    return {
        'mrr@10': float(np.mean(rr10)),
        'recall@1': float(np.mean(recall1)),
        'recall@5': float(np.mean(recall5)),
        'recall@10': float(np.mean(recall10)),
    }

def text_embedding_finetune(args, replaced_model, orig):
    os.makedirs('saved_models', exist_ok = True)
    os.makedirs(f'logs/{args.exp_name}', exist_ok = True)
    train_loader, val_loader = get_text_embedding_dataloaders(args.bert_tokenizer, args.ft_batch, args.num_workers, args.seq_len, 
                                                              dataset_name = args.embedding_dataset, dataset_config = args.embedding_dataset_config, 
                                                              train_split = args.embedding_train_split, val_split = args.embedding_val_split, 
                                                              val_fraction = args.embedding_val_fraction, 
                                                              max_train_samples = args.embedding_max_train_samples, 
                                                              max_val_samples = args.embedding_max_val_samples, query_max_len = args.query_max_len,
                                                              passage_max_len = args.passage_max_len, 
                                                              negatives_per_query = args.embedding_negatives_per_query,
                                                              seed = args.seed, streaming_train_samples = args.embedding_streaming_train_samples)
    for p in replaced_model.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(
    [
       {'params': [p for n, p in replaced_model.named_parameters() if not any(t in n for t in ('bias', 'bn', 'ln', 'norm', 'LayerNorm'))],
        'weight_decay': args.ft_wd},
       {'params': [p for n, p in replaced_model.named_parameters() if any(t in n for t in ('bias', 'bn', 'ln', 'norm', 'LayerNorm'))],
       'weight_decay': 0.0},
    ],
    lr = args.ft_lr
    )

    accum_steps = args.ft_accum
    if args.use_scheduler:
        steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
        total_steps = args.ft_epochs * steps_per_epoch
        warmup_steps = args.warmup_epochs * steps_per_epoch
        cosine_steps = max(1, total_steps - warmup_steps)
        sched_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, end_factor = 1.0, total_iters = warmup_steps)
        sched_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cosine_steps)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers = [sched_warmup, sched_cosine], milestones = [warmup_steps])
    else:
        scheduler = None

    replaced_model = replaced_model.to(device)
    orig = orig.to(device).eval()
    try:
        orig_metrics = eval_text_embedding_retrieval(orig, val_loader, device, args)
    except Exception:
        orig_metrics = {'mrr@10': 0.0, 'recall@1': 0.0, 'recall@5': 0.0, 'recall@10': 0.0}

    val_metrics = []
    train_losses = []
    log_json = {'orig_metrics': orig_metrics}
    best_mrr = -1.0

    for epoch in range(args.ft_epochs):
        metrics = eval_text_embedding_retrieval(replaced_model, val_loader, device, args)
        val_metrics.append(metrics)
        wandb.log({f'val_{k}': v for k, v in metrics.items()})
        if metrics['mrr@10'] >= best_mrr:
            best_mrr = metrics['mrr@10']
            torch.save(replaced_model.state_dict(), f'saved_models/ft_{args.exp_name}.pt')
        print(f'Epoch {epoch + 1}, Retrieval Metrics {metrics}')

        replaced_model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none = True)
        for i, batch in enumerate(tqdm(train_loader, desc = 'Iterating over retrieval train loader', dynamic_ncols = True)):
            loss, _scores = text_embedding_loss(replaced_model, batch, device, pooling = args.embedding_pool, temperature = args.embedding_temperature,
                                                use_hard_negatives = args.embedding_use_hard_negatives)
            loss = loss / accum_steps
            loss.backward()
            train_losses.append(loss.item() * accum_steps)
            train_loss += loss.item() * accum_steps
            if (i + 1) % accum_steps == 0:
                nn.utils.clip_grad_norm_([p for p in replaced_model.parameters() if p.requires_grad], max_norm = args.ft_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none = True)
                if scheduler:
                    scheduler.step()
            if i % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({'retrieval_train_loss': np.mean(train_losses[-20:]), 'current_lr': current_lr})
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Retrieval Training Loss {avg_train_loss}')

    metrics = eval_text_embedding_retrieval(replaced_model, val_loader, device, args)
    val_metrics.append(metrics)
    if metrics['mrr@10'] >= best_mrr:
        best_mrr = metrics['mrr@10']
        torch.save(replaced_model.state_dict(), f'saved_models/ft_{args.exp_name}.pt')
    log_json.update({'train_losses': train_losses, 'val_metrics': val_metrics})
    if not os.path.exists(f'logs/{args.exp_name}'):
        os.makedirs(f'logs/{args.exp_name}')
    with open(f'logs/{args.exp_name}/ft_results.json', 'w') as f:
        json.dump(log_json, f)
    return replaced_model, best_mrr, log_json, orig_metrics['mrr@10']
