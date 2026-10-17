import os
import math
import numpy as np
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import wandb
import bitsandbytes.optim as bnb_optim
import contextlib
import timm
import transformers

from tqdm import tqdm
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block
from transformers.models.bert.modeling_bert import BertLayer

from prior_utils import initialize_optimization_prior
from layer_registry import REPLACERS
from rep_sims import CKA, Procrustes, DifferentiableKNN
from modules import MemEffAttention
from dataloader import get_imagenet_dataloaders, cifar_dataset, get_text_dataloaders, get_text_embedding_dataloaders
from resnet_utils import *
from modules import BasicBlockCompat
from gpt2_models import gpt2_tiny, gpt2_small
from gpt2_utils import build_alignment_gpt2_small_to_large, get_gpt2_trainables
from bert_utils import build_alignment_bert_base_to_large, get_bert_trainables
from rnn_utils import build_alignment_rnn_rn18
from compare_params import compare_conv_weights
from vits import make_vit_tiny, make_vit_small, make_vit_base, make_vit_large
from vit_utils import build_alignment_vits, get_vit_trainables
from convnextv2_utils import build_alignment_convnextv2, get_convnextv2_trainables

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def list_target_layers(model, target_type):
    layers = []
    for idx, (name, module) in enumerate(model.named_modules(), start = 1):
        if isinstance(module, target_type):
            layers.append((idx, name))
        elif module.__class__.__name__ == target_type.__name__:
            layers.append((idx, name))
    return layers

def extract_input_shapes(model, dummy_input, target_type):
    '''
    Extracts input shapes for each layer across a network. We extract this input 
    shape to specify how large how linear replacement should be.
    '''
    shapes = []
    hooks = []
    def hook(module, inp, out):
        x = inp[0]
        shapes.append(tuple(x.shape[1:])) # Everything other than batch size --> ENSURE BATCH FIRST!
    for m in model.modules():
        if isinstance(m, target_type):
            hooks.append(m.register_forward_hook(hook))
        elif m.__class__.__name__ == target_type.__name__:
            hooks.append(m.register_forward_hook(hook))
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)
    model.eval()
    with torch.no_grad():
        if isinstance(dummy_input, dict):
            one = {k: v[:1].to(device) for k,v in dummy_input.items()}
            model(**one)
        else:
            one = dummy_input[:1].to(device)
            model(one)
    for h in hooks:
        h.remove()
    return shapes

def replace_nth_module(model, target_idx, input_shape, device, target_types, use_low_rank = True, rank = 1024, from_attn = False, use_amp = False):
    '''
    Replace the nth layer (given by target_idx)
    Does this recursively because why not?
    '''
    counter = {'i': 0}
    if not isinstance(target_types, tuple):
        target_types = (target_types,)
    replacers = tuple([REPLACERS[t] for t in target_types])
    def recurse(mod):
        for name, child in list(mod.named_children()):
            if isinstance(child, (target_types, replacers)):
                counter['i'] += 1
                if counter['i'] == target_idx and isinstance(child, target_types):
                    replacement = REPLACERS[type(child)]
                    if not from_attn:
                        setattr(mod, name, replacement(child, input_shape, device, low_rank = use_low_rank, rank = rank, amp = use_amp))
                    else:
                        setattr(mod, name, replacement.from_attn(child, target_idx))
                    print(f'Replaced {type(child)} #{target_idx} at {mod}.{name}')
                    return True
            elif child.__class__.__name__ in tuple(x.__name__ for x in target_types):
                counter['i'] += 1
                if counter['i'] == target_idx and child.__class__.__name__ in tuple(x.__name__ for x in target_types):
                    replacement = REPLACERS[[x for x in target_types if x.__name__ ==  child.__class__.__name__][0]]
                    if not from_attn:
                        setattr(mod, name, replacement(child, input_shape, device, low_rank = use_low_rank, rank = rank))
                    else:
                        r = r.from_attn(child, target_idx)
                        r = r.to(dtype = child.dtype)
                        setattr(mod, name, r)
                    print(f'Replaced {type(child)} #{target_idx} at {mod}.{name}')
                    return True
            else:
                if recurse(child):
                    return True
        return False
    if not recurse(model):
        prt_counter = counter['i']
        raise IndexError(f'Only {prt_counter} {target_types} layers; cannot replace #{target_idx} (counting {target_types}+{replacers})')
    return model

def get_nth_module(model, n, target_type):
    if not isinstance(target_type, tuple):
        target_type = (target_type,)
    replacers = tuple([REPLACERS[t] for t in target_type])
    count = 0
    for module in model.modules():
        if isinstance(module, (target_type, replacers)) or module.__class__.__name__ in tuple(x.__name__ for x in target_type):
            count += 1
            if count == n:
                return module
    raise IndexError

def get_norm_after_module(model, n, target_type):
    if not isinstance(target_type, tuple):
        target_type = (target_type,)
    replacers = tuple([REPLACERS[t] for t in target_type])
    count = 0
    modules = list(model.modules())
    for i, m in enumerate(modules):
        if isinstance(m, (target_type, replacers)) or m.__class__.__name__ in tuple(x.__name__ for x in target_type):
            count += 1
            if count == n:
                for j in range(i + 1, len(modules)):
                    if isinstance(modules[j], (nn.BatchNorm2d, nn.LayerNorm)):
                        return modules[j]
                return m
    raise IndexError(f'Only {count} {target_type}; cannot find {target_type} #{n}.')

def match_single_layer(args, index, orig_model, input_shapes, train_loader, device, target_type = nn.Conv2d, low_rank=True, rank=1024, epochs=5, lr=5e-3, 
                       distillation = False, temperature = 4, alpha = 0.5, from_attn = False, token_avg = False, last_token = False):
    if args.use_amp:
        autocast = torch.autocast('cuda', dtype = torch.bfloat16)
    else:
        autocast = contextlib.nullcontext()
    replaced_model = copy.deepcopy(orig_model).to(device)
    input_shape = input_shapes[index - 1]
    replace_nth_module(replaced_model, index, input_shape, device,
                       use_low_rank=low_rank, rank=rank, from_attn = from_attn, use_amp = args.use_amp)
    replaced_model = replaced_model.to(device)

    module_o = get_nth_module(orig_model, index, target_type)
    module_l = get_nth_module(replaced_model, index, target_type)
    activations_o = {}
    activations_l = {}
    h_o = module_o.register_forward_hook(lambda m, i, o: activations_o.setdefault('feat', o))
    h_l = module_l.register_forward_hook(lambda m, i, o: activations_l.setdefault('feat', o))

    for p in replaced_model.parameters():
        p.requires_grad = False
    for p in module_l.parameters():
        p.requires_grad = True

    if not args.bit8_adam:
        optimizer = optim.AdamW([p for p in replaced_model.parameters() if p.requires_grad], lr = lr, weight_decay = 0.0, fused = True)
    else:
        optimizer = bnb_optim.AdamW8bit([p for p in replaced_model.parameters() if p.requires_grad], lr = lr, weight_decay = 0.0, eps = 1e-6)
    cka = CKA(device, unbiased = True if args.repdist == 'UCKA' else False)
    proc = Procrustes(device)
    knn = DifferentiableKNN(device, k = 10, temperature = args.knn_temp)

    replaced_model.train()
    orig_model.eval()
    rep_losses = []
    for epoch in range(epochs):
        epoch_rep_losses = []
        for i, batch in enumerate(tqdm(train_loader, desc = f'Layer {index} Rep Sim Epoch {epoch}', dynamic_ncols = True)):
            if args.setting == 'imagenet':
                inputs = batch[0]['images']
                inputs = inputs.permute(0, 3, 1, 2)
                labels = inputs[0]['labels']
            elif args.setting == 'cifar':
                inputs, _ = batch
                inputs = inputs.to(device)
            elif args.setting == 'text':
                inputs = {'input_ids': batch['input_ids']}
                labels = inputs['input_ids'].clone()
            activations_o.clear(); activations_l.clear()
            
            if isinstance(inputs, dict):
                with autocast:
                    with torch.no_grad():
                        orig_model(**inputs, use_cache = False)
                    output = replaced_model(**inputs, use_cache = False)
                inputs = inputs['input_ids']
            else:
                with autocast:
                    with torch.no_grad():
                        orig_model(inputs)
                    output = replaced_model(inputs)

            if not token_avg and not last_token:
                fo = activations_o.reshape(inputs.size(0), -1)
                fl = activations_l.reshape(inputs.size(0), -1)
            elif last_token:
                fo = activations_o[:, -1, :].reshape(inputs.size(0), -1)
                fl = activations_l[:, -1, :].reshape(inputs.size(0), -1)
            else:
                fo = torch.mean(activations_o, dim = 1).reshape(inputs.size(0), -1)
                fl = torch.mean(activations_l, dim = 1).reshape(inputs.size(0), -1)
            fo, fl = fo.to(torch.float32), fl.to(torch.float32)
            if args.repdist == 'CKA' or args.repdist == 'UCKA':
                fo = fo / (fo.norm(dim=1, keepdim=True) + 1e-8)
                fl = fl / (fl.norm(dim=1, keepdim=True) + 1e-8)
                rep_sim_loss = 1.0 - cka.linear_CKA(fo, fl)
            elif args.repdist == 'KNN':
                rep_sim_loss, knn_info = knn.diff_knn_loss(fl, fo)
                wandb.log({f'{index}_kl': knn_info['kl'].detach()})
            else:
                fo = fo / (fo.norm(dim=1, keepdim=True) + 1e-8)
                fl = fl / (fl.norm(dim=1, keepdim=True) + 1e-8)
                rep_sim_loss = 1.0 - proc.orthogonal_procrustes_similarity(fo, fl)

            if args.task_loss and labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                label = batch[0]['labels'].squeeze().long()
                task_loss = loss_fn(output, label)
                loss += task_loss
            if args.local_loss:
                loss_fn = nn.MSELoss()
                mse_loss = loss_fn(fo, fl)
                loss += mse_loss

            if distillation:
                kl_loss_fn = nn.KLDivLoss(reduction = 'batchmean', log_target = True)
                with torch.no_grad():
                    orig_output = F.log_softmax(orig_output / temperature, dim = 1)
                new_output = F.log_softmax(new_output / temperature, dim = 1)
                kl = kl_loss_fn(new_output, orig_output) * (temperature ** 2)
                rep_sim_loss = rep_sim_loss * alpha + (1 - alpha) * kl

            wandb.log({f'{index}_loss': rep_sim_loss})

            optimizer.zero_grad(set_to_none = True)
            rep_sim_loss.backward()
            nn.utils.clip_grad_norm_(module_l.parameters(), max_norm = 1.0)
            optimizer.step()
            rep_losses.append(rep_sim_loss.item())
            epoch_rep_losses.append(rep_sim_loss.item())
        avg_train_loss = np.mean(epoch_rep_losses)
        print(f'Epoch {epoch}, Avg Loss {avg_train_loss}')

    h_o.remove()
    h_l.remove()

    return module_l.state_dict(), rep_losses

def match_all_layers(args, indices, orig, replaced_model, target_type, train_loader, device, 
                     lr = 1e-4, num_epochs = 10, rep_scheduler = False, token_avg = False, last_token = False,
                     reshape_tokens = False):
    if args.use_amp:
        autocast = torch.autocast('cuda', dtype = torch.bfloat16)
    else:
        autocast = contextlib.nullcontext()
    activations_o, activations_l = {}, {}
    hooks = []
    for idx in indices:
        if args.replace_norm:
            module_o = get_norm_after_module(orig, idx, target_type)
            module_l = get_norm_after_module(replaced_model, idx, target_type)
        else:
            module_o = get_nth_module(orig, idx, target_type)
            module_l = get_nth_module(replaced_model, idx, target_type)
        hooks.append(module_o.register_forward_hook(lambda m,i,o,idx=idx: activations_o.__setitem__(idx,o)))
        hooks.append(module_l.register_forward_hook(lambda m,i,o,idx=idx: activations_l.__setitem__(idx,o)))

    for p in replaced_model.parameters():
        p.requires_grad = False
    for idx in indices:
        module_l = get_nth_module(replaced_model, idx, target_type)
        for p in module_l.parameters():
            p.requires_grad = True
        
    for m in replaced_model.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = True

    if not args.bit8_adam:
        optimizer = optim.AdamW([p for p in replaced_model.parameters() if p.requires_grad], lr = lr, weight_decay = args.rep_wd, fused = True)
    else:
        print('Using 8-bit Adam!')
        optimizer = bnb_optim.AdamW8bit([p for p in replaced_model.parameters() if p.requires_grad], lr = lr, weight_decay = args.rep_wd, eps = 1e-6)
    scheduler = None
    if rep_scheduler:
        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch
        if num_epochs > 5:
            warmup_epochs = 5
            print(f'Building scheduler with {warmup_epochs} epochs of warmup')
            warmup_steps = warmup_epochs * steps_per_epoch
            cosine_steps = total_steps - warmup_steps
        else:
            warmup_ratio = 0.03
            print(f'Building scheduler with {warmup_ratio}% steps of warmup')
            warmup_steps = int(warmup_ratio * total_steps)
            cosine_steps = total_steps - warmup_steps
        sched_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, end_factor = 1.0, total_iters = warmup_steps)
        sched_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cosine_steps)  
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers = [sched_warmup, sched_cosine], milestones = [warmup_steps])
    cka = CKA(device, unbiased = True if args.repdist == 'UCKA' else False, log_hsic = args.log_hsic)
    proc = Procrustes(device)
    knn = DifferentiableKNN(device, k = args.knn_k, temperature = args.knn_temp, metric = 'cosine', zscore_logits = True)

    joint_losses = []
    layer_losses = {}

    replaced_model.train()
    orig.eval()
    for epoch in range(num_epochs):
        epoch_losses = []
        for step, batch in enumerate(tqdm(train_loader, desc = f'Joint Rep Sim Epoch {epoch}', dynamic_ncols = True)):
            if args.setting == 'imagenet':
                inputs = batch[0]['images']
                inputs = inputs.permute(0, 3, 1, 2)
                labels = batch[0]['labels']
            elif args.setting == 'cifar':
                inputs, _ = batch
                inputs = inputs.to(device)
            elif args.setting == 'text':
               inputs = {'input_ids': batch['input_ids'].to(device)}
            else:
                raise NotImplementedError()
            activations_o.clear(); activations_l.clear()

            if isinstance(inputs, dict):
                with autocast:
                    with torch.no_grad():
                        orig(**inputs, use_cache = False)
                    output = replaced_model(**inputs, use_cache = False)
                    inputs = inputs['input_ids']
            else:
                with autocast:
                    with torch.no_grad():
                        orig(inputs)
                    output = replaced_model(inputs)

            loss = torch.tensor(0.0).to(device)
            per_layer_loss = []
            for i, idx in enumerate(indices):
                if isinstance(activations_o[idx], tuple):
                    activations_o[idx] = activations_o[idx][0]
                if isinstance(activations_l[idx], tuple):
                    activations_l[idx] = activations_l[idx][0]
                    
                if not token_avg and not last_token and not reshape_tokens:
                    fo = activations_o[idx].reshape(inputs.size(0), -1)
                    fl = activations_l[idx].reshape(inputs.size(0), -1)
                elif last_token:
                    fo = activations_o[idx][:, -1, :].reshape(inputs.size(0), -1)
                    fl = activations_l[idx][:, -1, :].reshape(inputs.size(0), -1)
                elif reshape_tokens:
                    B, N, D = activations_o[idx].shape
                    fo = activations_o[idx].reshape(B * N, D)
                    fl = activations_l[idx].reshape(B * N, D)
                else:
                    fo = torch.mean(activations_o[idx], dim = 1).reshape(inputs.size(0), -1)
                    fl = torch.mean(activations_l[idx], dim = 1).reshape(inputs.size(0), -1)
                fo, fl = fo.to(torch.float32), fl.to(torch.float32)
                fo = fo / (fo.norm(dim=1, keepdim=True) + 1e-8)
                fl = fl / (fl.norm(dim=1, keepdim=True) + 1e-8)
                if args.repdist == 'CKA' or args.repdist == 'UCKA':
                    cka_score = cka.linear_CKA(fo, fl) + 1e-8
                    rep_sim_loss = -torch.log(cka_score)
                    # rep_sim_loss = 1.0 - cka.linear_CKA(fo, fl)
                elif args.repdist == 'KNN':
                    rep_sim_loss = knn.diff_knn_loss(fl, fo)
                else:
                    rep_sim_loss = 1.0 - proc.orthogonal_procrustes_similarity(fo, fl)
                per_layer_loss.append(rep_sim_loss)
                if f'Layer_{idx}' in layer_losses:
                    layer_losses[f'Layer_{idx}'].append(rep_sim_loss.item())
                else:
                    layer_losses[f'Layer_{idx}'] = [rep_sim_loss.item()]
            per_layer_loss = torch.stack(per_layer_loss, dim = 0)
            
            if args.layer_weight:
                if args.weighting_strategy == 'complicated':
                    with torch.no_grad():
                        Lvec = per_layer_loss.detach()
                        S = Lvec.sum() + 1e-12
                        rho = 0.20
                        w_prop = Lvec / S
                        w = (1.0 - rho) * w_prop + rho * (1.0 / Lvec.numel())
                
                    loss = torch.sum(w * per_layer_loss)
                else:
                    total_unweighted_loss = torch.sum(per_layer_loss)
                    if len(indices) > 1:
                        w = per_layer_loss / (total_unweighted_loss + 1e-8)
                    else:
                        w = per_layer_loss / (total_unweighted_loss)
                    loss = torch.sum(w * per_layer_loss)
            else:
                loss = torch.mean(per_layer_loss)

            if args.task_loss and labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                label = batch[0]['labels'].squeeze().long()
                task_loss = loss_fn(output, label)
                loss += args.task_penalty * task_loss
                
            if args.local_loss:
                loss_fn = nn.MSELoss()
                mse_loss = loss_fn(fo, fl)
                loss += mse_loss
                
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in replaced_model.parameters() if p.requires_grad], max_norm=0.3)
            optimizer.step()
                
            if scheduler:
                scheduler.step()

            if step % 20 == 0:
                wandb_indices = '_'.join([str(x) for x in indices])
                wandb.log({f'{wandb_indices}_total_loss': loss})
                for i, wan_idx in enumerate(indices):
                    wandb.log({f'{wandb_indices}_per_layer_loss/{wan_idx}': layer_losses[f'Layer_{wan_idx}'][-1]})
                    if args.layer_weight:
                        wandb.log({f'{wandb_indices}_per_layer_weight/{wan_idx}': w[i].item()})
                wandb_indices = '_'.join([str(x) for x in indices])
                wandb.log({f'{wandb_indices}_total_loss': loss})
            epoch_losses.append(loss.item())
            joint_losses.append(loss.item())
        epoch_layer_losses = {layer:np.mean(values[-len(train_loader):]) for layer,values in layer_losses.items()}
        print(f'Epoch {epoch}, layer losses: {epoch_layer_losses}')
        print(f'Epoch {epoch} avg loss: {np.mean(epoch_losses):.4f}')

    for h in hooks:
        h.remove()

    return replaced_model, layer_losses

def effective_rank(X):
    _, S, _ = torch.svd_lowrank(X.float(), q=min(X.shape[0], X.shape[1], 100))
    S = S / (S.sum() + 1e-8)
    entropy = -(S * torch.log(S + 1e-8)).sum()
    return torch.exp(entropy)

def prep_spatial_tokens(z, pool_hw = None):
    if isinstance(z, tuple):
        z = z[0]
    assert z.dim() == 4
    if pool_hw is not None and pool_hw > 0:
        z = F.adaptive_avg_pool2d(z, (pool_hw, pool_hw))
    B, C, H, W = z.shape
    return z.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

def prep_flat(z, pool_hw = None):
    if isinstance(z, tuple):
        z = z[0]
    if z.dim() == 4 and pool_hw is not None and pool_hw > 0:
        z = F.adaptive_avg_pool2d(z, (pool_hw, pool_hw))
    return z.reshape(z.size(0), -1)

def freeze_alignment_heads(model):
    for name in ['fc', 'classifier', 'head', 'heads']:
        if hasattr(model, name):
            mod = getattr(model, name)
            if isinstance(mod, nn.Module):
                for p in mod.parameters():
                    p.requires_grad = False

def freeze_bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def match_all_groups(args, group_ids, orig, replaced_model, orig_hook_modules, repl_hook_modules,
                     train_loader, device, lr = 1e-4, num_epochs = 10, rep_scheduler = False,
                     extra_trainable_modules = None, extra_trainable_params = None,
                     token_avg = False, student_init_state = None, max_stage_steps = None,
                     rep_val_loader = None):
    activations_o, activations_l = {}, {}
    if args.use_amp:
        autocast = torch.autocast('cuda', dtype = torch.bfloat16)
    else:
        autocast = contextlib.nullcontext()
    hooks = []
    teacher_target_counts = {}
    for gid in group_ids:
        mo = orig_hook_modules[gid]
        print(f'Group {gid} teacher target modules: {mo}')
        ml = repl_hook_modules[gid]
        if isinstance(mo, (list, tuple)):
            teacher_target_counts[gid] = len(mo)
            for j, m in enumerate(mo):
                hooks.append(m.register_forward_hook(lambda m, i, o, gid=gid, j=j: activations_o.__setitem__((gid, j), o)))
        else:
            teacher_target_counts[gid] = 1
            hooks.append(mo.register_forward_hook(lambda m, i, o, gid=gid: activations_o.__setitem__((gid, 0), o)))
        hooks.append(ml.register_forward_hook(lambda m, i, o, gid=gid: activations_l.__setitem__(gid, o)))
    print(f'Teacher target counts per group: {teacher_target_counts}')
    stage_start_state = copy.deepcopy(replaced_model.state_dict())
    for p in replaced_model.parameters():
        p.requires_grad = False
    for gid in group_ids:
        for p in repl_hook_modules[gid].parameters():
            p.requires_grad = True

    if extra_trainable_modules is not None:
        for mod in extra_trainable_modules:
            for p in mod.parameters():
                p.requires_grad = True

    if extra_trainable_params is not None:
        for p in extra_trainable_params:
            p.requires_grad = True

    freeze_alignment_heads(replaced_model)
    if args.freeze_bn_stats_align:
        freeze_bn_eval(replaced_model)
    trainable_params = [p for p in replaced_model.parameters() if p.requires_grad]

    if not args.bit8_adam:
        optimizer = optim.AdamW(trainable_params, lr = lr, weight_decay = args.rep_wd, fused = True)
    else:
        optimizer = bnb_optim.AdamW8bit(trainable_params, lr = lr, weight_decay = args.rep_wd, eps = 1e-6)

    scheduler = None
    if rep_scheduler:
        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch if max_stage_steps is None else max_stage_steps
        warmup_ratio = 0.1 if max_stage_steps is not None else 0.3
        print(f'Building scheduler with {warmup_ratio}% steps of warmup')
        warmup_steps = int(warmup_ratio * total_steps)
        cosine_steps = max(1, total_steps - warmup_steps)
        sched_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, end_factor = 1.0, total_iters = warmup_steps)
        sched_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cosine_steps)  
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers = [sched_warmup, sched_cosine], milestones = [warmup_steps])

    cka = CKA(device, unbiased = True if args.repdist == 'UCKA' else False, log_hsic = args.log_hsic)
    proc = Procrustes(device)
    knn = DifferentiableKNN(device, k = args.knn_k, temperature = args.knn_temp)
    group_losses = {} 
    orig.eval()
    global_step = 0

    def _unpack_alignment_batch(batch):
        if args.setting == 'imagenet':
            inputs = batch[0]['images'].permute(0, 3, 1, 2).to(device)
            labels = batch[0].get('labels', None)
            if labels is not None:
                labels = labels.squeeze().long().to(device)
        elif args.setting == 'cifar':
            inputs, labels = batch
            inputs = inputs.to(device)
            if labels is not None:
                labels = labels.to(device)
        elif args.setting == 'text' or args.setting == 'embedding':
            inputs = {'input_ids': batch['input_ids'].to(device)}
            if 'attention_mask' in batch:
                inputs['attention_mask'] = batch['attention_mask'].to(device)
            if 'token_type_ids' in batch:
                inputs['token_type_ids'] = batch['token_type_ids'].to(device)
            labels = None
        else:
            raise NotImplementedError()
        return inputs, labels

    def _input_batch_size(inputs):
        if isinstance(inputs, dict):
            return inputs['input_ids'].size(0)
        return inputs.size(0)

    def _call_model(model, inputs):
        try:
            return model(**inputs, use_cache = False)
        except TypeError:
            return model(**inputs)

    def _forward_alignment_batch(inputs):
        activations_o.clear()
        activations_l.clear()
        if isinstance(inputs, dict):
            with autocast:
                with torch.no_grad():
                    _call_model(orig, inputs)
                output = _call_model(replaced_model, inputs)
            input_ref = inputs
        else:
            with autocast:
                with torch.no_grad():
                    orig(inputs)
                output = replaced_model(inputs)
            input_ref = inputs
        return output, input_ref

    def _attention_mask(input_ref):
        if isinstance(input_ref, dict) and 'attention_mask' in input_ref:
            return input_ref['attention_mask']
        return None

    def _token_average(z, input_ref, batch_size):
        mask = _attention_mask(input_ref)
        if z.dim() == 4 and args.pool_hw is not None and args.pool_hw > 0:
            z = F.adaptive_avg_pool2d(z, (args.pool_hw, args.pool_hw))
        if z.dim() <= 2:
            return z.reshape(batch_size, -1)
        if z.dim() == 3 and mask is not None and mask.size(1) == z.size(1):
            mask = mask.to(z.device).unsqueeze(-1).to(z.dtype)
            return (z * mask).sum(dim = 1).reshape(batch_size, -1) / mask.sum(dim = 1).clamp_min(1.0)
        return torch.mean(z, dim = 1).reshape(batch_size, -1)

    def _compute_per_group_loss(input_ref, append_group_losses = False):
        per_group_loss = []
        batch_size = _input_batch_size(input_ref)
        for i, gid in enumerate(group_ids):
            fl = activations_l[gid]
            if isinstance(fl, tuple):
                fl = fl[0]
            if args.post_activation:
                fl = F.relu(fl)
            is_token_loss = args.cka_loss == 'token_cka' or args.knn_loss == 'token_knn'
            if not is_token_loss:
                if not token_avg:
                    fl = prep_flat(fl, pool_hw = args.pool_hw)
                else:
                    fl = _token_average(fl, input_ref, batch_size)
            m = teacher_target_counts[gid]
            target_losses = []
            for j in range(m):
                fo_raw = activations_o[(gid, j)]
                if isinstance(fo_raw, tuple):
                    fo_raw = fo_raw[0]
                if args.post_activation:
                    fo_raw = F.relu(fo_raw)
                if not is_token_loss:
                    if not token_avg:
                        fo = prep_flat(fo_raw, pool_hw = args.pool_hw)
                    else:
                        fo = _token_average(fo_raw, input_ref, batch_size)
                else:
                    fo = fo_raw
                if args.repdist == 'CKA' or args.repdist == 'UCKA':
                    if args.cka_loss == 'normal':
                        l_j = 1.0 - cka.linear_CKA(fl, fo)
                    elif args.cka_loss == 'log':
                        cka_val = cka.linear_CKA(fl, fo) + 1e-8
                        l_j = -torch.log(cka_val)
                    elif args.cka_loss == 'fancy':
                        cka_val = cka.linear_CKA(fl, fo) + 1e-8
                        eps = 1e-6
                        cka_score = cka_val.clamp(min = 0.0, max = 1.0 - eps)
                        base = -math.log(eps)
                        l_j = (torch.log1p(-cka_score) + base) / base
                    elif args.cka_loss == 'token_cka':
                        if fl.dim() == 4 and fo.dim() == 4:
                            fl_tok = prep_spatial_tokens(fl, pool_hw = args.pool_hw)
                            fo_tok = prep_spatial_tokens(fo, pool_hw = args.pool_hw)
                            cka_val = cka.token_aligned_CKA(fl_tok, fo_tok)
                        elif fl.dim() == 3 and fo.dim() == 3:
                            is_vit_tokens = getattr(args, 'replacement_type', None) == 'progressive_align_vits'
                            cka_val = cka.token_aligned_CKA(fl, fo, exclude_cls = is_vit_tokens)
                        else:
                            fl_flat = prep_flat(fl, pool_hw = args.pool_hw)
                            fo_flat = prep_flat(fo, pool_hw = args.pool_hw)
                            cka_val = cka.linear_CKA(fl_flat, fo_flat)
                        l_j = -torch.log(cka_val + 1e-5)
                elif args.repdist == 'KNN':
                    if args.knn_loss == 'normal':
                        l_j = knn.diff_knn_multilabel_loss(fl, fo, negative_weight = args.knn_neg_w)
                    elif args.knn_loss == 'token_knn':
                        if fl.dim() == 4 and fo.dim() == 4:
                            fl_tok = prep_spatial_tokens(fl, pool_hw = args.pool_hw)
                            fo_tok = prep_spatial_tokens(fo, pool_hw = args.pool_hw)
                            l_j = knn.token_aligned_diff_knn_multilabel_loss(fl_tok, fo_tok, center_batch = True, negative_weight = args.knn_neg_w)
                        elif fl.dim() == 3 and fo.dim() == 3:
                            l_j = knn.token_aligned_diff_knn_multilabel_loss(fl, fo, center_batch = True, negative_weight = args.knn_neg_w)
                        else:
                            fl_flat = prep_flat(fl, pool_hw = args.pool_hw)
                            fo_flat = prep_flat(fo, pool_hw = args.pool_hw)
                            l_j = knn.diff_knn_multilabel_loss(fl, fo, negative_weight = args.knn_neg_w)
                else:
                    l_j = 1.0 - proc.orthogonal_procrustes_similarity(fl, fo)
                target_losses.append(l_j)
            target_losses = torch.stack(target_losses, dim = 0)
            if m == 1:
                rep_sim_loss = target_losses[0]
            else:
                rep_sim_loss = target_losses.mean()
            if append_group_losses:
                group_losses.setdefault(f'Group_{gid}', []).append(rep_sim_loss.item())
            per_group_loss.append(rep_sim_loss)
        return torch.stack(per_group_loss, dim = 0)

    def _compute_rep_val_loss(rep_val_inputs):
        was_training = replaced_model.training
        replaced_model.eval()
        with torch.no_grad():
            _output, input_ref = _forward_alignment_batch(rep_val_inputs)
            per_group_loss = _compute_per_group_loss(input_ref, append_group_losses = False)
            rep_val_loss = torch.mean(per_group_loss).item()
        if was_training:
            replaced_model.train()
            if args.freeze_bn_stats_align:
                freeze_bn_eval(replaced_model)
        return rep_val_loss
    
    rep_val_source = train_loader if rep_val_loader is None else rep_val_loader
    rep_val_batch = next(iter(rep_val_source))
    rep_val_inputs, _rep_val_labels = _unpack_alignment_batch(rep_val_batch)
    rep_val_interval = getattr(args, 'rep_val_interval', 50)
    rep_val_interval = max(1, rep_val_interval)
    wandb_indices = '_'.join([str(x) for x in group_ids])
    best_unweighted = _compute_rep_val_loss(rep_val_inputs)
    best_state = copy.deepcopy(replaced_model.state_dict())
    print(f'Initial rep-val alignment loss {best_unweighted:.6f}')
    for epoch in range(num_epochs):
        epoch_losses = []
        for step, batch in enumerate(tqdm(train_loader, desc=f'Joint Rep Sim (groups) Epoch {epoch}', dynamic_ncols=True)):
            if max_stage_steps is not None and global_step >= max_stage_steps:
                break
            inputs, labels = _unpack_alignment_batch(batch)
            replaced_model.train()
            if args.freeze_bn_stats_align:
                freeze_bn_eval(replaced_model)
            output, input_ref = _forward_alignment_batch(inputs)
            loss = torch.tensor(0.0, device=device)
            per_group_loss = _compute_per_group_loss(input_ref, append_group_losses = True)
            if args.layer_weight:
                if args.weighting_strategy == 'complicated':
                    with torch.no_grad():
                        Lvec = per_group_loss.detach()
                        S = Lvec.sum() + 1e-12
                        rho = 0.20
                        w_prop = Lvec / S
                        w = (1.0 - rho) * w_prop + rho * (1.0 / Lvec.numel())
                
                    loss = torch.sum(w * per_group_loss)
                else:
                    total_unweighted_loss = torch.sum(per_group_loss)
                    if len(group_ids) > 1:
                        w = per_group_loss / (total_unweighted_loss + 1e-8)
                    else:
                        w = per_group_loss / (total_unweighted_loss)
                    loss = torch.sum(w * per_group_loss)
            else:
                if args.rep_agg == 'sum':
                    loss = torch.sum(per_group_loss)
                else:
                    loss = torch.mean(per_group_loss)
            if args.anchor_to_init:
                anchor = torch.tensor(0.0, device=device)
                for name, p in replaced_model.named_parameters():
                    if p.requires_grad:
                        p0 = student_init_state[name].to(device)
                        anchor = anchor + (p - p0).pow(2).mean()
                loss = loss + args.anchor_lambda * anchor

            if args.task_loss and labels is not None:
                task_loss = F.cross_entropy(output, labels)
                loss = loss + task_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm = 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            global_step += 1
            if global_step % rep_val_interval == 0 or (max_stage_steps is not None and global_step >= max_stage_steps):
                current_rep_val = _compute_rep_val_loss(rep_val_inputs)
                if current_rep_val < best_unweighted:
                    best_unweighted = current_rep_val
                    best_state = copy.deepcopy(replaced_model.state_dict())
                try:
                    wandb.log({f'{wandb_indices}_rep_val_loss': current_rep_val, f'{wandb_indices}_best_rep_val_loss': best_unweighted})
                except Exception:
                    pass
            try:
                wandb.log({f'{wandb_indices}_total_loss': loss})
                for i, wan_idx in enumerate(group_ids):
                    wandb.log({f'{wandb_indices}_per_group_loss/{wan_idx}': group_losses[f'Group_{wan_idx}'][-1]})
                    if args.layer_weight:
                        wandb.log({f'{wandb_indices}_per_layer_weight/{wan_idx}': w[i].item()})
                unweighted_avg_loss = np.mean([group_losses[f'Group_{wan_idx}'][-1] for wan_idx in group_ids])
                wandb.log({f'{wandb_indices}_unweighted_avg_loss': unweighted_avg_loss})
                wandb_indices = '_'.join([str(x) for x in group_ids])
                wandb.log({f'{wandb_indices}_total_loss': loss})
            except Exception:
                pass

            epoch_losses.append(loss.item())
        print(f'Epoch {epoch} avg loss: {np.mean(epoch_losses):.4f}')
        if max_stage_steps is not None and global_step >= max_stage_steps:
            print(f'Reached max_stage_steps={max_stage_steps}, stopping alignment early.')
            break

    if best_state is not None:
        replaced_model.load_state_dict(best_state)
        print(f'Restored best alignment checkpoint with rep-val loss {best_unweighted:.6f}')
    if args.align_mix < 1.0:
        with torch.no_grad():
            current_state = replaced_model.state_dict()
            mixed_state = copy.deepcopy(current_state)
            for name, val in current_state.items():
                if name in stage_start_state and torch.is_floating_point(val):
                    mixed_state[name] = stage_start_state[name].to(val.device) + args.align_mix * (val - stage_start_state[name].to(val.device))
            replaced_model.load_state_dict(mixed_state)
        print(f'Mixed best alignment checkpoint with stage start using align_mix={args.align_mix}')
    for h in hooks:
        h.remove()
    return replaced_model, group_losses

def maybe_restrict_schedule(schedule, args, tag = 'alignment'):
    if args.indices is None or len(args.indices) == 0:
        return schedule

    keep = set(args.indices)
    restricted = [g for g in schedule if g in keep]
    print(f'[{tag}] restricted gids: {restricted}')

    if len(restricted) == 0:
        raise ValueError(f'No alignment gids survived args.indices={args.indices}')

    return restricted

def progressive_align_resnet(args, teacher_rn50, student_rn18, train_loader, val_loader, device):
    '''
    Progressive alignment: introduce RN18 blocks one-by-one (unfreeze), and at each
    step jointly CKA-match all introduced RN18 blocks to their paired ResNet groups.
    ResNet stays fixed; RN18 is the model being trained.
    '''
    schedule, teacher_hook_modules, student_hook_modules = build_alignment_rn50_rn18(teacher_rn50, student_rn18, 
                                                                                     anchor = args.resnet_anchor,
                                                                                     supervise_all_convs = args.supervise_blocks,
                                                                                     like_with_like = args.like_convs)
    schedule = maybe_restrict_schedule(schedule, args, tag = 'RN18<-TeacherResNet')
    print(f'[RN18←TeacherResNet] Progressive schedule ({len(schedule)} steps):', schedule)

    teacher_rn50 = teacher_rn50.to(device).eval()
    student_rn18 = student_rn18.to(device)
    if args.compare_weights:
        old_student_rn18 = copy.deepcopy(student_rn18)
    student_init = copy.deepcopy(student_rn18.state_dict())

    if args.reload_progressive:
        print(f'Reloading model: {args.prog_ckpt}')
        assert os.path.exists(f'saved_models/{args.exp_name}/{args.prog_ckpt}' + '.pt')
        student_rn18.load_state_dict(torch.load(f'saved_models/{args.exp_name}/{args.prog_ckpt}' + '.pt'))
        idx = args.prog_ckpt.find('stage')
        ckpt_idx = int(args.prog_ckpt[idx+5:])
        print(f'Starting from stage {ckpt_idx}')
        start_idx = ckpt_idx
    else:
        start_idx = 0
    num_steps = len(schedule)  # 8
    lrs = args.progressive_lrs
    epochs = args.progressive_epochs
    if len(lrs) < num_steps:     
        lrs += [lrs[-1]] * (num_steps - len(lrs))
    if len(epochs) < num_steps:  
        epochs += [epochs[-1]] * (num_steps - len(epochs))

    os.makedirs(f'saved_models/{args.exp_name}', exist_ok=True)
    log_json = {}

    if args.joint_rn:
        stage_specs = [('Joint', schedule, lrs[0], epochs[0], 'rn_align_joint.pt')]
    else:
        stage_specs = []
        for step_idx in range(start_idx, num_steps):
            stage_specs.append((f'Stage_{step_idx+1}', schedule[:step_idx + 1], lrs[step_idx], epochs[step_idx], f'rn_align_stage{step_idx+1}.pt'))

    for stage_name, active_gids, lr, num_epochs, ckpt_name in stage_specs:
        extra_trainables = get_resnet_trainables(student_rn18, active_gids, context_blocks = args.train_context_blocks)
        print(f'\n===== RN18←TeacherResNet {stage_name} | active groups: {active_gids} =====')
        student_rn18, stage_losses = match_all_groups(
            args, active_gids, teacher_rn50, student_rn18,
            teacher_hook_modules, student_hook_modules,
            train_loader, device,
            lr = lr, num_epochs = num_epochs,
            rep_scheduler = args.rep_scheduler,
            extra_trainable_modules = extra_trainables,
            student_init_state = student_init,
            max_stage_steps = args.max_stage_steps,
            rep_val_loader = val_loader
        )
        log_json[stage_name] = stage_losses
        if args.compare_weights:
            sdA = old_student_rn18.state_dict()
            sdB = student_rn18.state_dict()
            rows, summary = compare_conv_weights(sdA, sdB, device='cpu')
            for r in sorted(rows, key=lambda d: d['diff_rms'], reverse=True)[:10]:
                print(r['name'], r['shape'], 'diff_mean', r['diff_mean'], 'diff_std', r['diff_std'], 'diff_rms', r['diff_rms'], 
                      'std_a', r['std_a'], 'std_b', r['std_b'])
        ckpt_file = f'saved_models/{args.exp_name}/{ckpt_name}'
        torch.save(student_rn18.state_dict(), ckpt_file)
    return student_rn18, log_json

def progressive_align_vit(args, teacher_vit, student_vit, train_loader, val_loader, device):
    schedule, teacher_hook_modules, student_hook_modules = build_alignment_vits(teacher_vit, student_vit, like_with_like = args.like_vit, 
                                                                                end_only = args.vit_endonly, branch_end = args.vit_branch_end)    
    schedule = maybe_restrict_schedule(schedule, args, tag = 'ViT<-TeacherViT')
    print(f'[ViT←TeacherViT] Progressive schedule ({len(schedule)} steps):', schedule)
    teacher_vit = teacher_vit.to(device).eval()
    student_vit = student_vit.to(device)
    student_init = copy.deepcopy(student_vit.state_dict())
    if args.reload_progressive and not getattr(args, 'joint_vit', False):
        print(f'Reloading model: {args.prog_ckpt}')
        assert os.path.exists(f'saved_models/{args.exp_name}/{args.prog_ckpt}' + '.pt')
        student_vit.load_state_dict(torch.load(f'saved_models/{args.exp_name}/{args.prog_ckpt}' + '.pt'))
        idx = args.prog_ckpt.find('stage')
        ckpt_idx = int(args.prog_ckpt[idx + 5:])
        print(f'Starting from stage {ckpt_idx}')
        start_idx = ckpt_idx
    else:
        start_idx = 0
    num_steps = len(schedule)
    lrs = list(args.progressive_lrs)
    epochs = list(args.progressive_epochs)
    if len(lrs) < num_steps:
        lrs += [lrs[-1]] * (num_steps - len(lrs))
    if len(epochs) < num_steps:
        epochs += [epochs[-1]] * (num_steps - len(epochs))
    os.makedirs(f'saved_models/{args.exp_name}', exist_ok = True)
    log_json = {}
    if args.joint_vit:
        stage_specs = [('Joint', schedule, lrs[0], epochs[0], 'vit_align_joint.pt')]
    else:
        stage_specs = []
        for step_idx in range(start_idx, num_steps):
            stage_specs.append((f'Stage_{step_idx+1}', schedule[:step_idx + 1], lrs[step_idx], epochs[step_idx], f'vit_align_stage{step_idx+1}.pt'))
    for stage_name, active_gids, lr, num_epochs, ckpt_name in stage_specs:
        extra_trainables, extra_trainable_params = get_vit_trainables(student_vit, active_gids, end_only = args.vit_endonly or args.vit_branch_end, 
                                                                      context_blocks = args.train_context_blocks)
        print(f'\n===== ViT←TeacherViT {stage_name} | active groups: {active_gids} =====')
        student_vit, stage_losses = match_all_groups(args, active_gids, teacher_vit, student_vit, 
                                                     teacher_hook_modules, student_hook_modules, train_loader, device,
                                                     lr = lr, num_epochs = num_epochs, rep_scheduler = args.rep_scheduler, 
                                                     extra_trainable_modules = extra_trainables, extra_trainable_params = extra_trainable_params, 
                                                     student_init_state = student_init, max_stage_steps = args.max_stage_steps,
                                                     rep_val_loader = val_loader)
        log_json[stage_name] = stage_losses
        ckpt_file = f'saved_models/{args.exp_name}/{ckpt_name}'
        torch.save(student_vit.state_dict(), ckpt_file)
        print(f'[ViT←TeacherViT] Saved {ckpt_file}')
    return student_vit, log_json

def progressive_align_cnxv2(args, teacher_cnxv2, student_cnxv2, train_loader, val_loader, device):
    schedule, teacher_hook_modules, student_hook_modules = build_alignment_convnextv2(teacher_cnxv2, student_cnxv2)
    schedule = maybe_restrict_schedule(schedule, args, tag = 'ConvNeXtV2←TeacherConvNeXtV2')
    print(f'[ConvNeXtV2←TeacherConvNeXtV2] Progressive schedule ({len(schedule)} steps):', schedule)
    teacher_cnxv2 = teacher_cnxv2.to(device).eval()
    student_cnxv2 = student_cnxv2.to(device)
    student_init = copy.deepcopy(student_cnxv2.state_dict())
    if args.reload_progressive and not getattr(args, 'joint_cnxv2', False):
        print(f'Reloading model: {args.prog_ckpt}')
        assert os.path.exists(f'saved_models/{args.exp_name}/{args.prog_ckpt}' + '.pt')
        student_cnxv2.load_state_dict(torch.load(f'saved_models/{args.exp_name}/{args.prog_ckpt}' + '.pt'))
        idx = args.prog_ckpt.find('stage')
        ckpt_idx = int(args.prog_ckpt[idx + 5:])
        print(f'Starting from stage {ckpt_idx}')
        start_idx = ckpt_idx
    else:
        start_idx = 0
    num_steps = len(schedule)
    lrs = list(args.progressive_lrs)
    epochs = list(args.progressive_epochs)
    if len(lrs) < num_steps:
        lrs += [lrs[-1]] * (num_steps - len(lrs))
    if len(epochs) < num_steps:
        epochs += [epochs[-1]] * (num_steps - len(epochs))
    os.makedirs(f'saved_models/{args.exp_name}', exist_ok = True)
    log_json = {}
    if args.joint_cnxv2:
        stage_specs = [('Joint', schedule, lrs[0], epochs[0], 'cnxv2_align_joint.pt')]
    else:
        stage_specs = []
        for step_idx in range(start_idx, num_steps):
            stage_specs.append((f'Stage_{step_idx+1}', schedule[:step_idx + 1], lrs[step_idx], epochs[step_idx], f'cnxv2_align_stage{step_idx+1}.pt'))
    for stage_name, active_gids, lr, num_epochs, ckpt_name in stage_specs:
        extra_trainables = get_convnextv2_trainables(student_cnxv2, active_gids)
        print(f'\n===== ConvNeXtV2←TeacherConvNeXtV2 {stage_name} | active groups: {active_gids} =====')
        student_cnxv2, stage_losses = match_all_groups(args, active_gids, teacher_cnxv2, student_cnxv2, 
                                                     teacher_hook_modules, student_hook_modules, train_loader, device,
                                                     lr = lr, num_epochs = num_epochs, rep_scheduler = args.rep_scheduler, 
                                                     extra_trainable_modules = extra_trainables, student_init_state = student_init, 
                                                     max_stage_steps = args.max_stage_steps, rep_val_loader = val_loader)
        log_json[stage_name] = stage_losses
        ckpt_file = f'saved_models/{args.exp_name}/{ckpt_name}'
        torch.save(student_cnxv2.state_dict(), ckpt_file)
        print(f'[ConvNeXtV2←TeacherConvNeXtV2] Saved {ckpt_file}')
    return student_cnxv2, log_json

def progressive_align_gpt2_small_to_large(args, teacher_gpt2l, student_gpt2s, train_loader, device, group_size = 3):
    schedule, teacher_hook_modules, student_hook_modules = build_alignment_gpt2_small_to_large(teacher_gpt2l, student_gpt2s, group_size = group_size,
                                                                                               include_embedding = True, include_final_ln = True)
    print(f'[GPT2←Teacher-GPT2] Progressive schedule ({len(schedule)} steps, group_size={group_size}): {schedule}')
    schedule = maybe_restrict_schedule(schedule, args, tag = 'GPT2<-TeacherGPT2')
    print(f'[GPT2←TeacherGPT2] Progressive schedule ({len(schedule)} steps, max_teacher_per_group={group_size}): {schedule}')
    
    teacher_gpt2l = teacher_gpt2l.to(device).eval()
    student_gpt2s = student_gpt2s.to(device)
    if hasattr(teacher_gpt2l, 'config'):
        teacher_gpt2l.config.use_cache = False
    if hasattr(student_gpt2s, 'config'):
        student_gpt2s.config.use_cache = False

    try:
        student_gpt2s.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {'use_reentrant': False})
    except TypeError:
        student_gpt2s.gradient_checkpointing_enable()
    except Exception:
        pass
    
    if hasattr(student_gpt2s, 'enable_input_require_grads'):
        try:
            student_gpt2s.enable_input_reqquire_grads()
        except:
            pass
    student_init = copy.deepcopy(student_gpt2s.state_dict())
        
    if args.reload_progressive:
        ckpt_path = f'saved_models/{args.exp_name}/{args.prog_ckpt}.pt'
        assert os.path.exists(ckpt_path)
        student_gpt2s.load_state_dict(torch.load(ckpt_path))
        idx = args.prog_ckpt.find('stage')
        if idx == -1:
            raise ValueError(f'Expected args.prog_ckpt to contain "stage", got {args.prog_ckpt}')
        start_idx = int(args.prog_ckpt[idx + 5:])
        print(f'Starting from stage {start_idx}')
    else:
        start_idx = 0

    num_steps = len(schedule)
    lrs = list(args.progressive_lrs)
    epochs = list(args.progressive_epochs)
    if len(lrs) < num_steps:
        lrs += [lrs[-1]] * (num_steps - len(lrs))
    if len(epochs) < num_steps: 
        epochs += [epochs[-1]] * (num_steps - len(epochs))
    
    os.makedirs(f'saved_models/{args.exp_name}', exist_ok = True)
    log_json = {}
    for step_idx in range(start_idx, num_steps):
        active_gids = schedule[:step_idx + 1]
        print(f'\n===== GPT2←TeacherGPT2 Alignment Step {step_idx+1}/{num_steps} | active groups: {active_gids} =====')
        stage_trainables = get_gpt2_trainables(student_gpt2s, active_gids)
        student_gpt2s, stage_losses = match_all_groups(args, active_gids, teacher_gpt2l, student_gpt2s, teacher_hook_modules, student_hook_modules, train_loader, device, 
                                                       lr = lrs[step_idx], num_epochs = epochs[step_idx], rep_scheduler = args.rep_scheduler, 
                                                       extra_trainable_modules = stage_trainables, token_avg = args.token_avg, student_init_state = student_init,
                                                       max_stage_steps = args.max_stage_steps
        )
        log_json[f'Stage_{step_idx+1}'] = stage_losses

        ckpt_file = f'saved_models/{args.exp_name}/gpt2_align_stage{step_idx+1}.pt'
        torch.save(student_gpt2s.state_dict(), ckpt_file)
        print(f'[GPT2←TeacherGPT2] Saved {ckpt_file}')

    return student_gpt2s, log_json

def progressive_align_bert_base_to_large(args, teacher_bertl, student_bertb, train_loader, val_loader, device):
    schedule, teacher_hook_modules, student_hook_modules = build_alignment_bert_base_to_large(teacher_bertl, student_bertb, include_embedding = True,
                                                                                              include_pooler = args.include_bert_pooler,
                                                                                              anchor = args.bert_anchor)
    schedule = maybe_restrict_schedule(schedule, args, tag = 'BERT<-TeacherBERT')
    print(f'[BERT←TeacherBERT] Progressive schedule ({len(schedule)} steps): {schedule}')

    teacher_bertl = teacher_bertl.to(device).eval()
    student_bertb = student_bertb.to(device)
    try:
        student_bertb.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {'use_reentrant': False})
    except TypeError:
        student_bertb.gradient_checkpointing_enable()
    except Exception:
        pass
    student_init = copy.deepcopy(student_bertb.state_dict())

    if args.reload_progressive and not getattr(args, 'joint_bert', False):
        ckpt_path = f'saved_models/{args.exp_name}/{args.prog_ckpt}.pt'
        assert os.path.exists(ckpt_path)
        student_bertb.load_state_dict(torch.load(ckpt_path))
        idx = args.prog_ckpt.find('stage')
        if idx == -1:
            raise ValueError(f'Expected args.prog_ckpt to contain "stage", got {args.prog_ckpt}')
        start_idx = int(args.prog_ckpt[idx + 5:])
        print(f'Starting from stage {start_idx}')
    else:
        start_idx = 0

    num_steps = len(schedule)
    lrs = list(args.progressive_lrs)
    epochs = list(args.progressive_epochs)
    if len(lrs) < num_steps:
        lrs += [lrs[-1]] * (num_steps - len(lrs))
    if len(epochs) < num_steps:
        epochs += [epochs[-1]] * (num_steps - len(epochs))

    os.makedirs(f'saved_models/{args.exp_name}', exist_ok = True)
    log_json = {}
    if args.joint_bert:
        stage_specs = [('Joint', schedule, lrs[0], epochs[0], 'bert_align_joint.pt')]
    else:
        stage_specs = []
        for step_idx in range(start_idx, num_steps):
            stage_specs.append((f'Stage_{step_idx+1}', schedule[:step_idx + 1], lrs[step_idx], epochs[step_idx], f'bert_align_stage{step_idx+1}.pt'))

    for stage_name, active_gids, lr, num_epochs, ckpt_name in stage_specs:
        stage_trainables = get_bert_trainables(student_bertb, active_gids, include_embedding = True, include_pooler = args.include_bert_pooler)
        print(f'\n===== BERT←TeacherBERT {stage_name} | active groups: {active_gids} =====')
        student_bertb, stage_losses = match_all_groups(args, active_gids, teacher_bertl, student_bertb, teacher_hook_modules, student_hook_modules, train_loader,
                                                       device, lr = lr, num_epochs = num_epochs, rep_scheduler = args.rep_scheduler,
                                                       extra_trainable_modules = stage_trainables, token_avg = args.token_avg,
                                                       student_init_state = student_init, max_stage_steps = args.max_stage_steps,
                                                       rep_val_loader = val_loader)
        log_json[stage_name] = stage_losses
        ckpt_file = f'saved_models/{args.exp_name}/{ckpt_name}'
        torch.save(student_bertb.state_dict(), ckpt_file)
        print(f'[BERT←TeacherBERT] Saved {ckpt_file}')

    return student_bertb, log_json

def progressive_align_rnn_to_rn18(args, teacher_rnn, student_rn18, train_loader, device):
    schedule, teacher_hooks, student_hooks = build_alignment_rnn_rn18(teacher_rnn, student_rn18)
    print('[RN18←RNN] Progressive schedule:', schedule)

    teacher_rnn = teacher_rnn.to(device).eval()
    student_rn18 = student_rn18.to(device)

    lrs = args.progressive_lrs
    epochs = args.progressive_epochs
    if len(lrs) < len(schedule):
        lrs += [lrs[-1]] * (len(schedule) - len(lrs))
    if len(epochs) < len(schedule):
        epochs += [epochs[-1]] * (len(schedule) - len(epochs))
    print(epochs)
    print(lrs)

    os.makedirs(f'saved_models/{args.exp_name}', exist_ok=True)
    log_json = {}

    for step_idx in range(len(schedule)):
        active = schedule[:step_idx + 1]
        print(f'\n===== RN18←RNN Alignment Step {step_idx+1}/{len(schedule)} | active groups: {active} =====')
        student_rn18, stage_losses = match_all_groups(
            args, active, teacher_rnn, student_rn18,
            teacher_hooks, student_hooks,
            train_loader, device,
            lr=lrs[step_idx], num_epochs=epochs[step_idx],
            rep_scheduler=getattr(args, 'rep_scheduler', False)
        )
        log_json[f'Stage_{step_idx+1}'] = stage_losses
        ckpt = f'saved_models/{args.exp_name}/rnn_align_stage{step_idx+1}.pt'
        torch.save(student_rn18.state_dict(), ckpt)
        print(f'[RN18←RNN] Saved {ckpt}')

    return student_rn18, log_json

def layer_replace(args, orig, indices, replacement_type = 'independent', low_rank = True, rank = 1024, reload_ft = False, reload_cka = False, reload_cka_model = None):
    if args.optimization_prior:
        return initialize_optimization_prior(args, orig, device, initialize_only = reload_ft or args.baseline)
    log_json = {}
    torch.cuda.empty_cache()
    if args.setting == 'imagenet':
        train_loader, val_loader = get_imagenet_dataloaders(args.imagenet, batch_size = args.rep_batch, data_dir = args.data_dir)
    elif args.setting == 'cifar':
        train_loader, val_loader = cifar_dataset(batch_size = args.rep_batch)
    elif args.setting == 'text':
        train_loader, val_loader, _, _ = get_text_dataloaders(args.model, batch_size = args.rep_batch, seq_len = args.seq_len, num_workers = args.num_workers)
    elif args.setting == 'embedding':
        train_loader, val_loader = get_text_embedding_dataloaders(args.bert_tokenizer, args.rep_batch, args.num_workers, args.seq_len, dataset_name = args.embedding_dataset, dataset_config = args.embedding_dataset_config,
                                                                  train_split = args.embedding_train_split, val_split = args.embedding_val_split,
                                                                  val_fraction = args.embedding_val_fraction, max_train_samples = args.embedding_max_train_samples,
                                                                  max_val_samples = args.embedding_max_val_samples, query_max_len = args.query_max_len,
                                                                  passage_max_len = args.passage_max_len, negatives_per_query = args.embedding_negatives_per_query,
                                                                  seed = args.seed, streaming_train_samples = args.embedding_streaming_train_samples)
    else:
        raise NotImplementedError()
    
    from_attn = False
    collect_input = True
    token_avg = False
    last_token = False
    if args.target_type == 'conv':
        target_type = nn.Conv2d
        if args.model == 'convnext':
            print('Changing replacers')
            REPLACERS[target_type] = REPLACERS[target_type][-1]
        else:
            REPLACERS[target_type] = REPLACERS[target_type][0]
            print('Using low_rank:', low_rank, 'Rank:', args.rank)
    elif args.target_type == 'mha':
        target_type = nn.MultiheadAttention
    elif args.target_type == 'vit':
        target_type = nn.Linear
    elif args.target_type == 'mematt':
        target_type = MemEffAttention
    elif args.target_type == 'llm_attention':
        target_type = Qwen2Attention
        from_attn = True
        collect_input = False
    elif args.target_type == 'attention':
        target_type = GPT2Attention
        collect_input = False
        if args.repdist == 'KNN':
            last_token = True
            args.token_avg = False
            print('Setting last_token to True...')
    elif args.target_type == 'bert' or replacement_type == 'progressive_align_berts':
        target_type = BertLayer
        collect_input = False
        token_avg = args.token_avg
    elif args.target_type == 'rnn':
        target_type = nn.RNN
    else:
        raise NotImplementedError
    print(f'{args.target_type} layers:')
    potential_layers = list_target_layers(orig, target_type)
    print(f'{len(potential_layers)} layers for replacement')
    for idx, name in potential_layers:
        print(f'  {idx}: {name}')
    print('Replacing indices', indices)

    replace_model_final = None
    if collect_input:
        batch = next(iter(train_loader))
        if args.setting == 'imagenet':
            dummy_input = torch.randn(size = batch[0]['images'].permute(0, 3, 1, 2).shape)
        else:
            dummy_input = torch.randn(size = batch[0].shape)
        input_shapes = extract_input_shapes(orig, dummy_input = dummy_input, target_type = target_type)
        print('Extracted following input shapes:', input_shapes, len(input_shapes))
    else:
        print('No input shape extraction')
        input_shapes = [None for i in range(len(indices))] if indices is not None else []

    if reload_ft or args.baseline:
        if replacement_type not in ['progressive_align_resnet', 'progressive_align_gpt2s', 'progressive_align_vits', 'progressive_align_cnxv2', 'progressive_align_berts']:
            ranks = [rank for _ in range(len(indices))] if args.ranks is None else args.ranks
            replace_model_final = copy.deepcopy(orig).to(device)
            for idx in indices:
                input_shape = input_shapes[idx - 1]
                rank = ranks[idx - 1]
                replace_nth_module(replace_model_final, idx, input_shape, device, target_type, use_low_rank=low_rank, rank=rank, from_attn = from_attn, use_amp = args.use_amp)
        elif replacement_type == 'progressive_align_resnet':
            print('Reloading ResNet-18!!')
            replace_model_final = torchvision.models.resnet18(pretrained = False).to(device)
        elif replacement_type == 'progressive_align_vits':
            if args.student_vit == 'vit_tiny':
                replace_model_final = make_vit_tiny()
            elif args.student_vit == 'vit_small':
                replace_model_final = make_vit_small()
            elif args.student_vit == 'vit_base':
                replace_model_final = make_vit_base()
            else:
                replace_model_final = make_vit_large()
        elif replacement_type == 'progressive_align_cnxv2':
            if args.student_cnxv2 == 'cnx2_atto':
                replace_model_final = timm.create_model('convnextv2_atto.fcmae_ft_in1k', pretrained = False)
            elif args.student_cnxv2 == 'cnx2_femto':
                replace_model_final = timm.create_model('convnextv2_femto.fcmae_ft_in1k', pretrained = False)
            elif args.student_cnxv2 == 'cnx2_pico':
                replace_model_final = timm.create_model('convnextv2_pico.fcmae_ft_in1k', pretrained = False)
            elif args.student_cnxv2 == 'cnx2_nano':
                replace_model_final = timm.create_model('convnextv2_nano.fcmae_ft_in1k', pretrained = False)
            elif args.student_cnxv2 == 'cnx2_tiny':
                replace_model_final = timm.create_model('convnextv2_tiny.fcmae_ft_in1k', pretrained = False)
            elif args.student_cnxv2 == 'cnx2_base':
                replace_model_final = timm.create_model('convnextv2_base.fcmae_ft_in1k', pretrained = False)
            else:
                replace_model_final = timm.create_model('convnextv2_large.fcmae_ft_in1k', pretrained = False)
        elif replacement_type == 'progressive_align_gpt2s':
            if args.student_gpt2 == 'gpt2_tiny':
                replace_model_final = gpt2_tiny()
            elif args.student_gpt2 == 'gpt2_small':
                replace_model_final = gpt2_small()
            elif args.student_gpt2 == 'gpt2':
                config = transformers.AutoConfig.from_pretrained('gpt2')
                replace_model_final = transformers.AutoModelForCausalLM.from_config(config) 
            else:
                config = transformers.AutoConfig.from_pretrained('gpt2-large')
                replace_model_final = transformers.AutoModelForCausalLM.from_config(config) 
        elif replacement_type == 'progressive_align_berts':
            config = transformers.AutoConfig.from_pretrained('bert-base-uncased')
            replace_model_final = transformers.AutoModel.from_config(config)
        else:
            config = transformers.AutoConfig.from_pretrained('gpt2')
            replace_model_final = transformers.AutoModelForCausalLM.from_config(config) 
        return replace_model_final
    
    elif reload_cka:
        print(f'Reloading {reload_cka_model}')
        replace_model_final = copy.deepcopy(orig).to(device)
        with open(f'logs/{reload_cka_model}/args.json', 'r') as f:
            reload_args = json.load(f)
        reload_indices = list(reload_args['indices'])
        print(f'Reloading indices... {reload_indices}')
        for idx in reload_indices:
            input_shape = input_shapes[idx - 1]
            replace_nth_module(replace_model_final, idx, input_shape, device, target_type,
                            use_low_rank=low_rank, rank=rank, from_attn = from_attn, use_amp = args.use_amp)
        chkpt = torch.load(f'saved_models/orig_{reload_cka_model}.pt')
        replace_model_final.load_state_dict(chkpt)
        indices = [idx for idx in indices if idx not in reload_indices]
        print(f'Remaining indices...{indices}')

    if replacement_type == 'independent':
        saved_states = {}
        for idx in indices:
            print(f'\n=== Independently Matching layer {idx} ===')
            state, cka_losses = match_single_layer(args, idx, orig, input_shapes,
                                                train_loader, device, repdist = args.repdist,
                                                low_rank = low_rank, rank = rank, lr = args.rep_lr, epochs = args.rep_epochs)
            log_json[f'Layer_{idx}'] = cka_losses
            saved_states[idx] = state
        if replace_model_final == None:
            assert not reload_cka, 'This behavior is unexpected when reload_cka is False!'
            replace_model_final = copy.deepcopy(orig).to(device)
        for idx in indices:
            input_shape = input_shapes[idx - 1]
            replace_nth_module(replace_model_final, idx, input_shape, device, target_type,
                            use_low_rank=low_rank, rank=rank, from_attn = from_attn)
            replacement_layer = get_nth_module(replace_model_final, idx, target_type, use_amp = args.use_amp)
            replacement_layer.load_state_dict(saved_states[idx])

    elif replacement_type == 'sequential':
        replace_model_final = copy.deepcopy(orig).to(device)
        all_losses = {}
        for idx in indices:
            print(f'\n=== Sequentially matching layer {idx} ===')
            state, losses = match_single_layer(args, idx, replace_model_final, input_shapes, train_loader,
                                               device, repdist = args.repdist, low_rank = low_rank, rank = rank,
                                               lr = args.rep_lr, epochs = args.rep_epochs)
            input_shape = input_shapes[idx - 1]
            replace_nth_module(replace_model_final, idx, input_shape, device, target_type, use_low_rank = low_rank, rank = rank, from_attn = from_attn, use_amp = args.use_amp)
            replacement_layer = get_nth_module(replace_model_final, idx, target_type)
            replacement_layer.load_state_dict(state)
            replace_model_final = replace_model_final.to(device)
            all_losses[f'Layer_{idx}'] = losses

    elif replacement_type == 'joint':
        prt_indices = ' '.join([str(x) for x in indices])
        print(f'\n=== Jointly matching layers {prt_indices} ===')
        replace_model_current = copy.deepcopy(orig).to(device)
        for idx in indices:
            input_shape = input_shapes[idx - 1]
            replace_nth_module(replace_model_current, idx, input_shape, device, target_type, use_low_rank = low_rank, rank = rank, from_attn = from_attn, use_amp = args.use_amp)
        print(replace_model_current)
        replace_model_current = replace_model_current.to(device)
        replace_model_final, log_json = match_all_layers(args, indices, orig, replace_model_current, target_type, train_loader, device, lr = args.rep_lr,
                                                         num_epochs = args.rep_epochs, rep_scheduler = args.rep_scheduler, token_avg = args.token_avg, 
                                                         last_token = last_token)
    
    elif replacement_type == 'progressive':
        prt_indices = ' '.join([str(x) for x in indices])
        print(f'\n=== Progressively matching layers {prt_indices} ===')
        indices = sorted(indices)
        num_stages = len(indices)
        replace_model_current = copy.deepcopy(orig).to(device)
        lrs = args.progressive_lrs
        epochs_per_stage = args.progressive_epochs
        ranks = [rank for _ in range(num_stages)] if args.ranks is None else args.ranks

        flip_mode = bool(getattr(args, 'model') == 'mlp')
        teacher_model = orig
        if flip_mode:
            print('[progressive] Flip RN18->MLP: teacher = RN18-as-MLP, student = RN18 (conv)')
            teacher_model = copy.deepcopy(orig).to(device)
            if isinstance(target_type, type) and target_type is nn.Conv2d:
                conv_shapes = input_shapes
            else:
                batch0 = next(iter(train_loader))
                if args.setting == 'imagenet':
                    dummy_conv = torch.randn(size=batch0[0]['images'].permute(0, 3, 1, 2).shape)
                else:
                    dummy_conv = torch.randn(size=batch0[0].shape)
                conv_shapes = extract_input_shapes(orig, dummy_input = dummy_conv, target_type = nn.Conv2d)

            for ci in range(1, len(conv_shapes) + 1):
                shp = conv_shapes[ci - 1]
                rank = ranks[ci - 1]
                replace_nth_module(
                    teacher_model, ci, shp, device, nn.Conv2d,
                    use_low_rank=low_rank, rank=rank, from_attn=False, use_amp=args.use_amp
                )

            target_type = nn.Conv2d
            teacher_model = teacher_model.to(device)

        cumulative_indices = []

        if args.reload_progressive:
            print(f'Reloading model: {args.prog_ckpt}')
            assert os.path.exists(f'saved_models/{args.exp_name}/{args.prog_ckpt}' + '.pt')
            ckpt_start_idx = int(args.prog_ckpt.split('_')[-2])
            length = int(args.prog_ckpt.split('_')[-1])
            assert ckpt_start_idx == indices[0]
            end_idx = ckpt_start_idx + length
            cumulative_indices = [x for x in range(ckpt_start_idx, end_idx + 1)]
            for j in cumulative_indices:
                input_shape = input_shapes[j - 1]
                rank = ranks[j - 1]
                if not flip_mode:
                    replace_nth_module(replace_model_current, j, input_shape, device, target_type,  use_low_rank = low_rank, rank = rank, 
                                    from_attn = from_attn, use_amp = args.use_amp)
            replace_model_current.load_state_dict(torch.load(f'saved_models/{args.exp_name}/{args.prog_ckpt}' + '.pt'))
            replace_model_current = replace_model_current.to(device)
            indices = [idx for idx in indices if idx not in cumulative_indices]
            num_stages = len(indices)

        if len(lrs) < num_stages:
            print(f'Warning: Number of LRs ({len(lrs)}) is less than number of stages ({num_stages}). Extending with last LR value.')
            lrs.extend([lrs[-1]] * (num_stages - len(lrs)))

        if len(epochs_per_stage) < num_stages:
            print(f'Warning: Number of Epochs ({len(epochs_per_stage)}) is less than number of stages ({num_stages}). Extending with last Epoch value.')
            epochs_per_stage.extend([epochs_per_stage[-1]] * (num_stages - len(epochs_per_stage)))

        if not os.path.exists(f'saved_models/{args.exp_name}'):
            os.makedirs(f'saved_models/{args.exp_name}')
        rep_scheduler = args.rep_scheduler
        for stage_idx, new_idx_to_add in enumerate(indices):
            current_lr = lrs[stage_idx]
            current_epochs = epochs_per_stage[stage_idx]
            print(f'\n=================Progressive Stage {stage_idx + 1}/{num_stages} =================')
            print(f'Adding layer: {new_idx_to_add} | LR for this stage: {current_lr} | Numbers of epochs: {current_epochs}')

            input_shape = input_shapes[new_idx_to_add - 1]
            if not flip_mode:
                rank = ranks[new_idx_to_add - 1]
                replace_nth_module(replace_model_current, new_idx_to_add, input_shape, device, target_type,
                                use_low_rank = low_rank, rank = rank, from_attn = from_attn, use_amp = args.use_amp)
            replace_model_current = replace_model_current.to(device)
            if from_attn or args.model == 'gpt2':
                replace_model_current.config.use_cache = False
                replace_model_current.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs = {'use_reentrant': False}
                )
                replace_model_current.enable_input_require_grads()
                replace_model_current.train()
            cumulative_indices.append(new_idx_to_add)
            if len(cumulative_indices) > 1 and not args.skip_indiv:
                print(f'---Individually training newly added layer: {[new_idx_to_add]} ---')
                replace_model_current, _ = match_all_layers(args, [new_idx_to_add], teacher_model, replace_model_current, target_type, train_loader, device, 
                                                            num_epochs = 10, lr = 1e-5, rep_scheduler = rep_scheduler, token_avg = args.token_avg, 
                                                            last_token = last_token, reshape_tokens = args.reshape_tokens)
                torch.cuda.empty_cache()
            print(f'--- Jointly training all modified layers: {cumulative_indices} ---')
            
            replace_model_current, stage_losses = match_all_layers(args, cumulative_indices, teacher_model, replace_model_current, target_type, train_loader, 
                                                                   device, num_epochs = current_epochs, lr = current_lr, rep_scheduler = rep_scheduler, 
                                                                   token_avg = args.token_avg, last_token = last_token, reshape_tokens = args.reshape_tokens)
            torch.cuda.empty_cache()
            log_json[f'Stage_{stage_idx+1}_Layer_{new_idx_to_add}'] = stage_losses
            replace_model_final = copy.deepcopy(replace_model_current)
            save_idx = new_idx_to_add - cumulative_indices[0]
            model_save_str = f'replace_progressive_{cumulative_indices[0]}_{save_idx}.pt'
            torch.save(replace_model_current.state_dict(), f'saved_models/{args.exp_name}/{model_save_str}')

    elif replacement_type == 'progressive_align_resnet':
        print(f'[RN18←TeacherResNet] Starting progressive student alignment (teacher={args.model} fixed).')
        teacher = orig.to(device)
        student = torchvision.models.resnet18(pretrained = False)
        student = student.to(device)
        replace_model_final, log_json = progressive_align_resnet(args, teacher, student, train_loader, val_loader, device)

    elif replacement_type == 'progressive_align_vits':
        print(f'[ViT←TeacherViT] Starting progressive student alignment (teacher={args.model} fixed).')
        teacher = orig.to(device)
        if args.student_vit == 'vit_tiny':
            student = make_vit_tiny()
        elif args.student_vit == 'vit_small':
            student = make_vit_small()
        elif args.student_vit == 'vit_base':
            student = make_vit_base()
        else:
            student = make_vit_large()
        student = student.to(device)
        replace_model_final, log_json = progressive_align_vit(args, teacher, student, train_loader, val_loader, device)

    elif replacement_type == 'progressive_align_cnxv2':
        print(f'[ConvNeXtV2←TeacherConvNeXtV2] Starting progressive student alignment (teacher={args.model} fixed).')
        teacher = orig.to(device)
        if args.student_cnxv2 == 'cnx2_atto':
            student = timm.create_model('convnextv2_atto.fcmae_ft_in1k', pretrained = False)
        elif args.student_cnxv2 == 'cnx2_femto':
            student = timm.create_model('convnextv2_femto.fcmae_ft_in1k', pretrained = False)
        elif args.student_cnxv2 == 'cnx2_pico':
            student = timm.create_model('convnextv2_pico.fcmae_ft_in1k', pretrained = False)
        elif args.student_cnxv2 == 'cnx2_nano':
            student = timm.create_model('convnextv2_nano.fcmae_ft_in1k', pretrained = False)
        elif args.student_cnxv2 == 'cnx2_tiny':
            student = timm.create_model('convnextv2_tiny.fcmae_ft_in1k', pretrained = False)
        elif args.student_cnxv2 == 'cnx2_base':
            student = timm.create_model('convnextv2_base.fcmae_ft_in1k', pretrained = False)
        else:
            student = timm.create_model('convnextv2_large.fcmae_ft_in1k', pretrained = False)
        student = student.to(device)
        replace_model_final, log_json = progressive_align_cnxv2(args, teacher, student, train_loader, val_loader, device)
    
    elif replacement_type == 'progressive_align_gpt2s':
        print('[GPT2←TeacherGPT2] Starting progressive student alignment (teacher=GPT-2 fixed).')
        teacher = orig.to(device)
        if args.student_gpt2 == 'gpt2_tiny':
            student = gpt2_tiny()
        elif args.student_gpt2 == 'gpt2_small':
            student = gpt2_small()
        elif args.student_gpt2 == 'gpt2':
            config = transformers.AutoConfig.from_pretrained('gpt2')
            student = transformers.AutoModelForCausalLM.from_config(config, attn_implementation='eager')
        else:
            config = transformers.AutoConfig.from_pretrained('gpt2-large')
            student = transformers.AutoModelForCausalLM.from_config(config, attn_implementation='eager')
        replace_model_final, log_json = progressive_align_gpt2_small_to_large(args, teacher, student, train_loader, device, group_size = 3)

    elif replacement_type == 'progressive_align_berts':
        print('[BERT←TeacherBERT] Starting progressive student alignment (teacher=BERT fixed).')
        teacher = orig.to(device)
        config = transformers.AutoConfig.from_pretrained('bert-base-uncased')
        student = transformers.AutoModel.from_config(config)
        replace_model_final, log_json = progressive_align_bert_base_to_large(args, teacher, student, train_loader, val_loader, device)

    elif replacement_type == 'progressive_align_rnn_to_rn18':
        print('[RN18←RNN] Starting progressive alignment (teacher = ImageRNN fixed).')
        teacher = orig.to(device)
        student = torchvision.models.resnet18(pretrained=False).to(device)
        replace_model_final, log_json = progressive_align_rnn_to_rn18(args, teacher, student, train_loader, device)
    else:
        raise NotImplementedError('Select an optimization strategy among independent, sequential, joint, progressive, progressive_align_rn18_to_rn50, progressive_align_gpt2_small_to_large, progressive_align_berts, progressive_align_rnn_to_rn18')

    os.makedirs(f'logs/{args.exp_name}', exist_ok = True)
    with open(f'logs/{args.exp_name}/rep_sim_matches.json', 'w') as f:
        json.dump(log_json, f)
    return replace_model_final
