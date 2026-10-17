import os
import re
import pickle
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import transformers
import scipy
import seaborn as sns

from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm
from itertools import islice
from contextlib import contextmanager

from rep_sims import CKA
from layer_replace import progressive_align_gpt2_small_to_large
from gpt2_utils import build_alignment_gpt2_small_to_large
from resnet_utils import build_alignment_rn50_rn18, build_alignment_resnet_conv
from dataloader import get_text_dataloaders, get_imagenet_dataloaders
from finetune import recalibrate_bn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_cka(args, cka_losses, rank = None):
    plt.figure(dpi = 150, figsize = (21, 4))
    for i, layer in enumerate(cka_losses):
        layer_cka_loss = np.array(cka_losses[layer])
        x = np.array([i for i in range(len(layer_cka_loss))])
        plt.plot(x, layer_cka_loss, label = f'layer_{i}', alpha = 0.7)
    plt.xlabel('Training Steps')
    plt.ylabel('1 - RepSim')
    plt.legend()
    title = 'RepSim Losses Across Layers'
    title += f' with rank {rank}' if rank != None else ''
    plt.title(title)
    os.makedirs(f'figures/{args.exp_name}/', exist_ok = True)
    plt.savefig(f'figures/{args.exp_name}/cka_losses.pdf', format = 'pdf')

def plot_ft(args, ft_json, orig_acc):
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 4), dpi = 100)
    
    train_losses = np.array(ft_json['train_losses'])
    x = np.array([i for i in range(len(train_losses))])
    axes[0].plot(x, train_losses)
    axes[0].set_title('Training Loss Over Steps')
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Training Loss')

    val_losses = np.array(ft_json['val_losses'])
    epochs = np.array([i for i in range(len(val_losses))])
    axes[1].plot(epochs, val_losses)
    axes[1].set_title('Validation Loss Over Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Validation Loss')

    accs = np.array(ft_json['top1_accs'])
    epochs = np.array([i for i in range(len(accs))])
    axes[2].plot(epochs, accs)
    axes[2].set_title('Top-1 Accuracy Over Epochs')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Top-1 Accuracy')
    axes[2].axhline(orig_acc, linestyle = '--', color = 'red')

    os.makedirs(f'figures/{args.exp_name}', exist_ok = True)
    plt.savefig(f'figures/{args.exp_name}/ft_plots.pdf', format = 'pdf')

def gram_linear(X):
    return X @ X.T

def gram_rbf(X, gamma = None):
    D2 = torch.cdist(X, X, p=2.0) ** 2
    if gamma is None:
        off = ~torch.eye(D2.size(0), dtype=torch.bool, device=D2.device)
        med = torch.median(D2[off])
        med = med if med > 0 else torch.tensor(1.0, device=X.device, dtype=X.dtype)
        gamma = 1.0 / (2.0 * med)
    return torch.exp(-gamma * D2)

def _gram(Z, kernel = 'linear'):
    return gram_linear(Z) if kernel == 'linear' else gram_rbf(Z)

def center_gram(K):
    if K.dtype != torch.float32:
        K = K.to(dtype = torch.float32)
    n = K.size(0)
    I = torch.eye(n, device=K.device)
    H = I - torch.ones((n, n), device=K.device) / n
    return H @ K @ H

@torch.no_grad()
def seq_plot_cka_matrices(group_idx = 0, plot = 'flattened', robust = True, show_colorbars = True):
    print('Plotting CKA matrices...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_batches = 5
    train_loader, _, _, _ = get_text_dataloaders('gpt2', batch_size = 64, seq_len = 200)
    gpt2_large = transformers.AutoModelForCausalLM.from_pretrained('gpt2-large')
    config = transformers.AutoConfig.from_pretrained('gpt2')
    gpt2_small = transformers.AutoModelForCausalLM.from_config(config)
    gpt2_large = gpt2_large.to(device)
    gpt2_small = gpt2_small.to(device)
    schedule, teacher_hook_modules, student_hook_modules = build_alignment_gpt2_small_to_large(gpt2_large, gpt2_small, group_size=3)

    activations_o, activations_l = {}, {}
    hooks = []
    mo = teacher_hook_modules[group_idx]
    ml = student_hook_modules[group_idx]
    hooks.append(mo.register_forward_hook(lambda m, i, o, gid=group_idx: activations_o.__setitem__(group_idx, o)))
    hooks.append(ml.register_forward_hook(lambda m, i, o, gid=group_idx: activations_l.__setitem__(group_idx, o)))
    cka = CKA(device)

    rows = num_batches
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(1.5*cols, 1.4*rows))

    for i, batch in enumerate(train_loader):
        if i == num_batches:
            break
        inputs = {'input_ids': batch['input_ids'].to(device)}
        with torch.no_grad():
            gpt2_large(**inputs, use_cache = False)
        output = gpt2_small(**inputs, use_cache = False)
        inputs = inputs['input_ids']
        fo = activations_o[group_idx]
        fl = activations_l[group_idx]
        if isinstance(fo, tuple):
            fo = fo[0]
        if isinstance(fl, tuple):
            fl = fl[0]

        if plot == 'flattened':
            fo = fo.reshape(inputs.size(0), -1)
            fl = fl.reshape(inputs.size(0), -1)
        elif plot == 'seq_average':
            fo = torch.mean(fo, dim = 1).reshape(inputs.size(0), -1)
            fl = torch.mean(fl, dim = 1).reshape(inputs.size(0), -1)
        else:
            raise NotImplementedError
        fo = fo / (fo.norm(dim = 1, keepdim = True) + 1e-8)
        fl = fl / (fl.norm(dim = 1, keepdim = True) + 1e-8)
        K_fo = _gram(fo)
        K_fl = _gram(fl)
        Kc_fo = center_gram(K_fo)
        Kc_fl = center_gram(K_fl)

        contrib = Kc_fo * Kc_fl
        cka_val = cka.linear_CKA(fo, fl)

        def norm_for(mat):
            if robust:
                m = mat.detach()
                lo = torch.quantile(m, 0.01).item()
                hi = torch.quantile(m, 0.99).item()
                return TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
            return None

        mats = [
            (Kc_fl, f'{plot}: GPT-2'),
            (Kc_fo, f'{plot}: GPT-2-L'),
            (contrib, f'$\\tilde{{K}}_X\\odot\\tilde{{K}}_Y$ (CKA={cka_val:.3f})'),
        ]

        for j, (M, title) in enumerate(mats):
            ax = axes[i][j]
            im = ax.imshow(
                M.detach().cpu().numpy(),
                interpolation='nearest',
                aspect='equal',
                norm=norm_for(M),
            )
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if show_colorbars:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(f'figures/cka_matrices_{group_idx}_{plot}.png', bbox_inches='tight', dpi = 500)
    plt.show()
    pass

@torch.no_grad()
def seq_plot_cka_hist(group_idx = 0, plot = 'flattened', robust = True):
    print('Plotting CKA histogram...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_batches = 5
    train_loader, _, _, _ = get_text_dataloaders('gpt2', batch_size = 64, seq_len = 200)
    gpt2_large = transformers.AutoModelForCausalLM.from_pretrained('gpt2-large')
    config = transformers.AutoConfig.from_pretrained('gpt2')
    gpt2_small = transformers.AutoModelForCausalLM.from_config(config)
    gpt2_large = gpt2_large.to(device)
    gpt2_small = gpt2_small.to(device)
    schedule, teacher_hook_modules, student_hook_modules = build_alignment_gpt2_small_to_large(gpt2_large, gpt2_small, group_size=3)

    activations_o, activations_l = {}, {}
    hooks = []
    mo = teacher_hook_modules[group_idx]
    ml = student_hook_modules[group_idx]
    hooks.append(mo.register_forward_hook(lambda m, i, o, gid=group_idx: activations_o.__setitem__(group_idx, o)))
    hooks.append(ml.register_forward_hook(lambda m, i, o, gid=group_idx: activations_l.__setitem__(group_idx, o)))
    cka = CKA(device)

    rows = num_batches
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(1.6*cols, 1.4*rows))

    for i, batch in enumerate(train_loader):
        if i == num_batches:
            break
        inputs = {'input_ids': batch['input_ids'].to(device)}
        with torch.no_grad():
            gpt2_large(**inputs, use_cache = False)
        output = gpt2_small(**inputs, use_cache = False)
        inputs = inputs['input_ids']
        fo = activations_o[group_idx]
        fl = activations_l[group_idx]
        if isinstance(fo, tuple):
            fo = fo[0]
        if isinstance(fl, tuple):
            fl = fl[0]

        if plot == 'flattened':
            fo = fo.reshape(inputs.size(0), -1)
            fl = fl.reshape(inputs.size(0), -1)
        elif plot == 'seq_average':
            fo = torch.mean(fo, dim = 1).reshape(inputs.size(0), -1)
            fl = torch.mean(fl, dim = 1).reshape(inputs.size(0), -1)
        else:
            raise NotImplementedError
        fo = fo / (fo.norm(dim = -1, keepdim = True) + 1e-8)
        fl = fl / (fl.norm(dim = -1, keepdim = True) + 1e-8)

        K_fo = _gram(fo)
        K_fl = _gram(fl)
        Kc_fo = center_gram(K_fo)
        Kc_fl = center_gram(K_fl)

        contrib = Kc_fo * Kc_fl
        cka_val = cka.linear_CKA(fo, fl, unbiased = False)
        print('Biased CKA Value:', cka_val)
        cka_val = cka.linear_CKA(fo, fl, unbiased = True)
        print('Unbiased CKA Value:', cka_val)

        mats = [(K_fo, f'{plot}: GPT-2'),
            (K_fl, f'{plot}: GPT-2-L'),
            (contrib, f'$\\tilde{{K}}_X\\odot\\tilde{{K}}_Y$ (CKA={cka_val:.3f})'),
        ]

        for j, (M, title) in enumerate(mats):
            M = M.detach().cpu().numpy().ravel()
            # M_z_score = scipy.stats.zscore(M)
            # M = M[M_z_score < 3]
            ax = axes[i][j]
            hist = ax.hist(
                M
            )
            ax.set_title(title, fontsize=9)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0)) 
            ax.tick_params(axis='x', labelsize = 6)
            ax.tick_params(axis='y', labelsize = 6)
    fig.tight_layout()
    fig.savefig(f'figures/cka_hist_{group_idx}_{plot}.png', bbox_inches='tight', dpi = 500)
    plt.show()
    pass

def compare_runs(run1, run2, savename = '', label1 = '', label2 = ''):
    import json
    with open(f'logs/{run1}/ft_results.json', 'r') as f:
        ft1 = json.load(f)
    with open(f'logs/{run2}/ft_results.json', 'r') as f:
        ft2 = json.load(f)
    
    accs1 = ft1['top1_accs']
    accs2 = ft2['top1_accs']
    train_losses1 = ft1['train_losses']
    train_losses2 = ft2['train_losses']
    val_losses1 = ft1['val_losses']
    val_losses2 = ft2['val_losses']
    epochs1 = np.array([i for i in range(len(accs1))])
    epochs2 = np.array([i for i in range(len(accs2))])
    steps1 = np.array([i for i in range(len(train_losses1))])
    steps2 = np.array([i for i in range(len(train_losses2))])

    fig, axes = plt.subplots(dpi = 150, figsize = (14, 4), nrows = 1, ncols = 3)
    axes[0].plot(steps1, train_losses1, label = f'Run 1: {label1}', alpha = 0.7)
    axes[0].plot(steps2, train_losses2, label = f'Run 2: {label2}', alpha = 0.7)
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Training Loss')
    axes[0].legend()
    axes[0].set_title('Training Loss Comparison')
    axes[1].plot(epochs1, accs1, label = f'Run 1: {label1}', alpha = 0.7)
    axes[1].plot(epochs2, accs2, label = f'Run 2: {label2}', alpha = 0.7)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Top-1 Accuracy')
    axes[1].set_title('Top-1 Accuracy Comparison')
    axes[2].plot(epochs1[:100], val_losses1, label = f'Run 1: {label1}', alpha = 0.7)
    axes[2].plot(epochs2[:100], val_losses2, label = f'Run 2: {label2}', alpha = 0.7)
    axes[2].set_xlabel('Steps')
    axes[2].set_ylabel('Validation Loss')
    axes[2].set_title('Validation Loss Comparison')
    fig.tight_layout()
    fig.savefig(f'figures/{savename}.png', format = 'png', dpi = 150)

def compare_runs_cka(run1, run2, savename = '', label1 = '', label2 = '', metric = ''):
    with open(f'logs/{run1}/ft_results_seed42.json', 'r') as f:
        ft1 = json.load(f)
    with open(f'logs/{run2}/ft_results_seed0.json', 'r') as f:
        ft2 = json.load(f)
    
    accs1 = ft1['top1_accs']
    accs2 = ft2['top1_accs']
    train_losses1 = ft1['train_losses']
    train_losses2 = ft2['train_losses']
    val_losses1 = ft1['val_losses']
    val_losses2 = ft2['val_losses']
    epochs1 = np.array([i for i in range(len(accs1))])
    epochs2 = np.array([i for i in range(len(accs2))])
    steps1 = np.array([i for i in range(len(train_losses1))])
    steps2 = np.array([i for i in range(len(train_losses2))])

    with open(f'logs/{run2}/rep_sim_matches.json', 'r') as f:
        cka1 = json.load(f)
    cka_losses = cka1['Stage_16']
    cka_losses = np.array(list(cka_losses.values()))
    avg_losses = np.mean(cka_losses, axis = 0)
    cka_steps = np.array([i for i in range(len(avg_losses))])

    fig, axes = plt.subplots(dpi = 150, figsize = (19, 4), nrows = 1, ncols = 4)
    axes[0].plot(steps1, train_losses1, label = f'Run 1: {label1}', alpha = 0.7)
    axes[0].plot(steps2, train_losses2, label = f'Run 2: {label2}', alpha = 0.7)
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Training Loss')
    axes[0].legend()
    axes[0].set_title('Training Loss Comparison')
    axes[1].plot(epochs1, accs1, label = f'Run 1: {label1}', alpha = 0.7)
    axes[1].plot(epochs2, accs2, label = f'Run 2: {label2}', alpha = 0.7)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Top-1 Accuracy')
    axes[1].set_title('Top-1 Accuracy Comparison')
    axes[2].plot(epochs1[:100], val_losses1, label = f'Run 1: {label1}', alpha = 0.7)
    axes[2].plot(epochs2[:100], val_losses2, label = f'Run 2: {label2}', alpha = 0.7)
    axes[2].set_xlabel('Steps')
    axes[2].set_ylabel('Validation Loss')
    axes[2].set_title('Validation Loss Comparison')
    axes[3].plot(cka_steps, avg_losses, label = f'Run 2: {label2}', alpha = 0.7, color = 'green')
    axes[3].set_xlabel('Steps')
    axes[3].set_ylabel(f'Average {metric}')
    axes[3].set_title(f'Average {metric} Over Layers')
    fig.tight_layout()
    fig.savefig(f'figures/{savename}.png', format = 'png', dpi = 150)

@torch.no_grad()
def image_plot_cka_hist(group_idx = 0, plot = 'flattened', robust = True):
    print('Plotting CKA histogram...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_batches = 5
    train_loader, _ = get_imagenet_dataloaders(batch_size = 256, data_dir = '../../imagenet_pytorch')
    resnet34 = torchvision.models.resnet34(pretrained=False).to(device)
    resnet18 = torchvision.models.resnet18(pretrained=False).to(device)
    schedule, teacher_hook_modules, student_hook_modules = build_alignment_rn50_rn18(resnet34, resnet18)

    activations_o, activations_l = {}, {}
    hooks = []
    teacher_target_counts = {}
    mo = teacher_hook_modules[group_idx]
    ml = student_hook_modules[group_idx]
    if isinstance(mo, (list, tuple)):
        teacher_target_counts[group_idx] = len(mo)
        for j, m in enumerate(mo):
            hooks.append(m.register_forward_hook(lambda m, i, o, gid=group_idx, j=j: activations_o.__setitem__((group_idx, j), o)))
    else:
        teacher_target_counts[group_idx] = 1
        hooks.append(mo.register_forward_hook(lambda m, i, o, gid=group_idx: activations_o.__setitem__((group_idx, 0), o)))
    hooks.append(ml.register_forward_hook(lambda m, i, o, gid=group_idx: activations_l.__setitem__(group_idx, o)))
    cka = CKA(device)
    ucka = CKA(device, unbiased = True)

    rows = num_batches
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(1.6*cols, 1.4*rows))
    for i, batch in enumerate(train_loader):
        if i == num_batches:
            break
        inputs = batch[0]['images'].permute(0, 3, 1, 2)     
        with torch.no_grad():
            resnet34(inputs)
        output = resnet18(inputs)
        fl = activations_l[group_idx]
        if isinstance(fl, tuple):
            fl = fl[0]
        if plot == 'flattened':
            fl = fl.reshape(inputs.size(0), -1)
        else:
            raise NotImplementedError
        m = teacher_target_counts[group_idx]
        target_losses = []
        fo_activations = []
        unbiased_losses = []
        biased_losses = []
        for j in range(m):
            fo_raw = activations_o[(group_idx, j)]
            if plot == 'flattened':
                fo = fo_raw.reshape(inputs.size(0), -1)
            else:
                raise NotImplementedError
            l_j = -torch.log(cka.linear_CKA(fl, fo) + 1e-8)
            target_losses.append(l_j)
            l_j = cka.linear_CKA(fl, fo)
            biased_losses.append(l_j)
            l_j = ucka.linear_CKA(fl, fo)
            unbiased_losses.append(l_j)
            fo_activations.append(fo)
        biased_losses = torch.stack(biased_losses, dim=0)
        cka_val = biased_losses.mean()
        print('Biased CKA Value:', biased_losses)
        print('Unbiased CKA Value:', unbiased_losses)
        fo = fo_activations[1]
        K_fo = _gram(fo)
        K_fl = _gram(fl)
        # Kc_fo = center_gram(K_fo)
        # Kc_fl = center_gram(K_fl)
        Kc_fo = K_fo
        Kc_fl = K_fl
        contrib = Kc_fo * Kc_fl

        mats = [(K_fo, f'{plot}: ResNet-34'),
            (K_fl, f'{plot}: ResNet-18'),
            (contrib, f'$\\tilde{{K}}_X\\odot\\tilde{{K}}_Y$ (CKA={cka_val:.3f})'),
        ]

        for j, (M, title) in enumerate(mats):
            M = M.detach().cpu().numpy().ravel()
            ax = axes[i][j]
            hist = ax.hist(
                M
            )
            ax.set_title(title, fontsize=9)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0)) 
            ax.tick_params(axis='x', labelsize = 6)
            ax.tick_params(axis='y', labelsize = 6)
    fig.tight_layout()
    fig.savefig(f'figures/new_image_cka_hist_{group_idx}_{plot}.png', bbox_inches='tight', dpi = 500)
    plt.show()
    pass

def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def _list_stage_ckpts(exp_name):
    d = f'saved_models/{exp_name}'
    files = [f for f in os.listdir(d) if f.startswith('rn18_align_stage') and f.endswith('.pt')]
    files.sort(key=lambda s: int(re.findall(r'stage(\d+)', s)[0]))
    return [os.path.join(d, f) for f in files]

@torch.no_grad()
def probe_all_group_losses(teacher, student, tmods, smods, loader, num_batches=2):
    cka = CKA(device, unbiased=False)
    losses = None
    counts = 0

    # hook all gids once
    activ_t = {}
    activ_s = {}
    hooks = []
    for gid, (tlist, smod) in enumerate(zip(tmods, smods)):
        # in your resnet_utils you wrap chosen teacher conv in a list
        tmod = tlist[0]
        hooks.append(tmod.register_forward_hook(lambda m,i,o,gid=gid: activ_t.__setitem__(gid, o)))
        hooks.append(smod.register_forward_hook(lambda m,i,o,gid=gid: activ_s.__setitem__(gid, o)))

    teacher.eval()
    student.train()

    for bi, batch in enumerate(loader):
        if bi >= num_batches:
            break
        x = batch[0]['images'].permute(0,3,1,2).to(device)

        activ_t.clear()
        activ_s.clear()

        teacher(x)
        student(x)
        # compute per-gid -log(CKA)
        gid_vals = []
        for gid in range(len(smods)):
            ft = activ_t[gid]
            fs = activ_s[gid]
            if isinstance(ft, tuple): ft = ft[0]
            if isinstance(fs, tuple): fs = fs[0]

            ft = ft.reshape(x.size(0), -1).float()
            fs = fs.reshape(x.size(0), -1).float()

            cka_val = cka.linear_CKA(fs, ft) + 1e-8
            gid_vals.append(float((-torch.log(cka_val)).cpu()))

        gid_vals = np.asarray(gid_vals, dtype=np.float32)
        if losses is None:
            losses = gid_vals
        else:
            losses += gid_vals
        counts += 1

    for h in hooks:
        h.remove()

    return losses / max(counts, 1)

def probe_alignment(exp_name):
    args = _load_json(f'logs/{exp_name}/args.json')
    # teacher selection
    m = args['model']
    if m == 'rn34':
        teacher = torchvision.models.resnet34(pretrained=False).to(device)
    elif m == 'rn50':
        teacher = torchvision.models.resnet50(pretrained=False).to(device)
    elif m == 'rn101':
        teacher = torchvision.models.resnet101(pretrained=False).to(device)
    elif m == 'rn152':
        teacher = torchvision.models.resnet152(pretrained=False).to(device)
    else:
        raise ValueError(m)

    supervise_all_convs = bool(args.get('supervise_blocks', True))
    anchor = args.get('resnet_anchor', 'mid')
    like_with_like = bool(args.get('like_convs', False))

    # probe loader (use a fixed few batches)
    train_loader, _ = get_imagenet_dataloaders(args['imagenet'], batch_size=256, data_dir=args['data_dir'])

    ckpts = _list_stage_ckpts(exp_name)
    M = []
    for ck in tqdm(ckpts, desc='Probing checkpoints'):
        student = torchvision.models.resnet18(pretrained=False).to(device)
        student.load_state_dict(torch.load(ck, map_location='cpu'))
        student = student.to(device)
        # recalibrate_bn(student, train_loader, num_batches=20)
        train_loader, _ = get_imagenet_dataloaders(args['imagenet'], batch_size=256, data_dir=args['data_dir'])
        schedule, tmods, smods = build_alignment_resnet_conv(
            teacher, student,
            supervise_all_convs=supervise_all_convs,
            anchor=anchor,
            like_with_like=like_with_like,
            print_name_map=False
        )   
        s_name = {id(m): n for n, m in student.named_modules()}
        gid_to_stage = {gid: s_name[id(smods[gid])].split('.')[0] for gid in range(len(smods))}

        v = probe_all_group_losses(teacher, student, tmods, smods, train_loader, num_batches=20)
        M.append(v)

    M = np.stack(M, axis=0)

    # summarize by stage: median + IQR across gids in that stage
    stages = ['layer1', 'layer2', 'layer3', 'layer4']
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    axes = axes.flatten()
    x = np.arange(M.shape[0])

    for ax, st in zip(axes, stages):
        gids = [gid for gid in range(M.shape[1]) if gid_to_stage[gid] == st]
        S = M[:, gids]  # (num_ckpts, num_gids_in_stage)
        med = np.median(S, axis=1)
        q25 = np.percentile(S, 25, axis=1)
        q75 = np.percentile(S, 75, axis=1)
        ax.plot(x, med, label='median')
        ax.fill_between(x, q25, q75, alpha=0.25, label='IQR')
        ax.set_title(st)
        ax.set_ylabel('-log(CKA)')
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('checkpoint index (rn18_align_stagek)')
    axes[-2].set_xlabel('checkpoint index (rn18_align_stagek)')
    fig.suptitle(f'Alignment Probe on {m}, student = rn18', fontsize=16)
    plt.tight_layout()
    fig.savefig(f'figures/{exp_name}_alignment_probe.png', format='png', dpi=150)
    plt.show()

def _sorted_align_steps(rep_log):
    keys = [k for k in rep_log.keys() if k.startswith('Stage_')]
    keys.sort(key=lambda s: int(s.split('_')[1]))
    return keys

def _concat_segments(rep_log, step_keys):
    per_group = {}
    boundaries = []
    T = 0
    for sk in step_keys:
        seg = rep_log[sk]  # dict Group_i -> [losses...]
        seg_len = len(next(iter(seg.values())))
        for gname, vals in seg.items():
            per_group.setdefault(gname, []).extend(vals)
        T += seg_len
        boundaries.append(T)

    groups = sorted(per_group.keys(), key=lambda s: int(s.split('_')[1]))
    arr = np.full((len(groups), T), np.nan, dtype=np.float32)
    for i, g in enumerate(groups):
        v = np.asarray(per_group[g], dtype=np.float32)
        arr[i, : len(v)] = v
    return groups, arr, boundaries

def _gid_to_stage_map(args_dict):
    # Rebuild teacher/student just to get student module names per gid
    teacher_name = args_dict['model']
    if teacher_name == 'rn34':
        teacher = torchvision.models.resnet34(pretrained=False)
    elif teacher_name == 'rn50':
        teacher = torchvision.models.resnet50(pretrained=False)
    elif teacher_name == 'rn101':
        teacher = torchvision.models.resnet101(pretrained=False)
    elif teacher_name == 'rn152':
        teacher = torchvision.models.resnet152(pretrained=False)
    else:
        raise ValueError(f'Unsupported teacher model in this script: {teacher_name}')

    student = torchvision.models.resnet18(pretrained=False)

    supervise_all_convs = bool(args_dict.get('supervise_blocks', True))
    anchor = args_dict.get('resnet_anchor', 'mid')
    like_with_like = bool(args_dict.get('like_convs', False))

    schedule, tmods, smods = build_alignment_resnet_conv(
        teacher, student,
        supervise_all_convs=supervise_all_convs,
        anchor=anchor,
        like_with_like=like_with_like,
        print_name_map=False
    )

    s_name = {id(m): n for n, m in student.named_modules()}
    gid_to_stage = {}
    for gid, smod in enumerate(smods):
        nm = s_name[id(smod)]  # e.g. 'layer2.0.conv1'
        stage = nm.split('.')[0]  # layer1..layer4
        gid_to_stage[gid] = stage
    return gid_to_stage

def resnet_layer_loss(exp_name):
    args = _load_json(f'logs/{exp_name}/args.json')
    rep = _load_json(f'logs/{exp_name}/rep_sim_matches.json')

    step_keys = _sorted_align_steps(rep)
    groups, arr, boundaries = _concat_segments(rep, step_keys)
    gid_to_stage = _gid_to_stage_map(args)

    # group index -> gid int
    gids = [int(g.split('_')[1]) for g in groups]

    stages = ['layer1', 'layer2', 'layer3', 'layer4']
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    axes = axes.flatten()

    for ax, stage in zip(axes, stages):
        rows = [i for i, gid in enumerate(gids) if gid_to_stage.get(gid) == stage]
        if len(rows) == 0:
            ax.set_title(stage + ' (no groups)')
            continue
        sub = arr[rows, :]  # (num_groups_in_stage, T)

        med = np.nanmedian(sub, axis=0)
        q25 = np.nanpercentile(sub, 25, axis=0)
        q75 = np.nanpercentile(sub, 75, axis=0)

        x = np.arange(sub.shape[1])
        ax.plot(x, med, label='median')
        ax.fill_between(x, q25, q75, alpha=0.25, label='IQR')

        # show alignment-step boundaries
        for b in boundaries:
            ax.axvline(b, linewidth=0.5, alpha=0.25)

        ax.set_title(stage)
        ax.set_ylabel('-log(CKA)')
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('optimizer update index (concatenated across progressive steps)')
    axes[-2].set_xlabel('optimizer update index (concatenated across progressive steps)')
    fig.suptitle(f'Layerwise Alignment Losses Across Training ({args["model"]} teacher)', fontsize=16)
    plt.tight_layout()
    fig.savefig(f'figures/{exp_name}_layerwise_alignment_loss.png', format='png', dpi=150)
    plt.show()

def calculate_cka_scores(imgs, acts1, acts2, avg = False):
    cka = CKA(device)
    cka_scores = dict()
    for l1 in acts1:
        activations1 = acts1[l1]
        if avg:
            activations1 = torch.mean(activations1, dim = 1)
        else:
            activations1 = activations1.reshape(imgs.size(0), -1)
        for l2 in acts2:
            activations2 = acts2[l2]
            if avg:
                activations2 = torch.mean(activations2, dim = 1)
            else:
                activations2 = activations2.reshape(imgs.size(0), -1)
            with torch.no_grad():
                cka_l1_l2 = cka.linear_CKA(activations1, activations2)
            cka_scores[(l1, l2)] = cka_l1_l2.detach().cpu().item()
    del acts1, acts2
    return cka_scores

class ResNetConvFeatureExtractor(nn.Module):
    def __init__(self, backbone, *, detach = True, to_cpu = False, clone = False):
        super().__init__()
        self.backbone = backbone
        self.detach = detach
        self.to_cpu = to_cpu
        self.clone = clone

        self._handles = []
        self.activations = {}
        self._register_conv_hooks()

    def _register_conv_hooks(self) -> None:
        self.remove_hooks()
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

    def _make_hook(self, name):
        def hook(_module, _inp, out):
            x = out
            if self.detach:
                x = x.detach()
            if self.clone:
                x = x.clone()
            if self.to_cpu:
                x = x.cpu()
            self.activations[name] = x
        return hook

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def forward(self, x, *, return_logits = False):
        self.activations = {}
        logits = self.backbone(x)
        if return_logits:
            return logits, self.activations
        return self.activations

def _natural_key(s):
    s = str(s)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def _dict_to_matrix(sim_dict, rn18_order=None, other_order=None, rn18_first_in_key=True):
    rows = []
    for (a, b), v in sim_dict.items():
        rn18_layer, other_layer = (a, b) if rn18_first_in_key else (b, a)
        rows.append((other_layer, rn18_layer, float(v)))

    df = pd.DataFrame(rows, columns=['other_layer', 'rn18_layer', 'cka'])
    mat = df.pivot(index='other_layer', columns='rn18_layer', values='cka')

    # ---- ONLY CHANGE: default to natural/alphanumeric order when no order is provided
    if rn18_order is None:
        rn18_order = sorted(mat.columns, key=_natural_key)
    if other_order is None:
        other_order = sorted(mat.index, key=_natural_key)

    mat = mat.reindex(columns=rn18_order)
    mat = mat.reindex(index=other_order)
    # ---- end change

    return mat

def plot_similarity_heatmap(model1, model2, batch, device, model1_order = None, other_order = None, model1_first_in_key = True, 
                            figsize = None, title = 'Layer similarity (CKA)', use_seaborn = True, xlabel = 'ResNet-18', ylabel = 'ResNet-50'):
    imgs = batch[0]['images'].permute(0, 3, 1, 2).to(device)
    model1_acts = model1(imgs)
    model2_acts = model2(imgs)
    sim_dict = calculate_cka_scores(imgs, model1_acts, model2_acts)
    with open('not_cka_scores.pkl', 'wb') as f:
        pickle.dump(sim_dict, f)
    torch.cuda.empty_cache()

    mat = _dict_to_matrix(sim_dict, rn18_order=model1_order, other_order=other_order, rn18_first_in_key=model1_first_in_key)
    if model1_order is not None:
        mat = mat.reindex(columns=model1_order)
    if other_order is not None:
        mat = mat.reindex(index=other_order)
    if figsize is None:
        figsize = (8, 6)
    if use_seaborn:
        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            mat,
            vmin=0, vmax=1,
            cmap='viridis',
            cbar_kws={'label': 'CKA similarity'},
            linewidths=0.0
        )
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('figures/layer_similarity_heatmap.png', format='png', dpi=150)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat.to_numpy(), vmin=0, vmax=1, aspect='auto')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('CKA similarity')
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(mat.columns, rotation=90)
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(mat.index)

        plt.tight_layout()
        fig.savefig('figures/layer_similarity_heatmap.png', format='png', dpi=150)

if __name__ == '__main__':
    # run1 = 'base_rn18_seed42'
    # run1 = 'log_untrained_rn34_rn18_progressive_replace'
    # run2 = 'knn_untrained_rn34_rn18_progressive_replace'
    # compare_runs(run1, run2, savename = 'new_training_acc_curve_pt_untrained_rn101_rn18', label1 = 'Base RN-18',
    #              label2 = 'Untrained RN-101 --> RN-18')
    # compare_runs_cka(run1, run2, savename = 'cka_new_training_acc_curve_pt_untrained_rn34_rn18_knn', label1 = 'CKA RN-34 --> RN-18',
    #              label2 = 'DiffKNN RN-34 --> RN-18', metric = 'DiffKNN')
    # image_plot_cka_hist(group_idx = 0, plot = 'flattened', robust = True)
    # probe_alignment('log_untrained_rn50_rn18_progressive_replace')
    # resnet_layer_loss('log_untrained_rn50_rn18_progressive_replace')

    model1 = torchvision.models.resnet18(pretrained=False).to(device)
    model1.load_state_dict(torch.load('saved_models/orig_theseus_untrained_rn101_rn18_progressive_replace.pt', map_location='cpu'))
    model1 = ResNetConvFeatureExtractor(model1, detach = True, to_cpu = False, clone = False)
    model2 = torchvision.models.resnet101(pretrained=False).to(device)
    model2 = ResNetConvFeatureExtractor(model2, detach = True, to_cpu = False, clone = False)
    probe_loader, _ = get_imagenet_dataloaders(batch_size=256, data_dir='../../imagenet_pytorch')
    batch = next(iter(probe_loader))
    plot_similarity_heatmap(model1, model2, batch, device, use_seaborn = True, ylabel = 'ResNet-101')
    pass