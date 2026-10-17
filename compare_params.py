import torch
import torchvision

@torch.no_grad()
def compare_conv_weights(sd_a, sd_b, *, device='cpu', dtype=torch.float32, eps=1e-12):
    keys_a = list(sd_a.keys())
    keys_b = list(sd_b.keys())
    if keys_a != keys_b:
        missing = set(keys_a) - set(keys_b)
        extra   = set(keys_b) - set(keys_a)
        raise ValueError(f'State dict keys differ. missing={len(missing)} extra={len(extra)}')
    def is_conv_weight(k, v):
        return k.endswith('.weight') and isinstance(v, torch.Tensor) and v.ndim == 4

    rows = []
    total_elems = 0
    sum_sq_diff = 0.0 
    sum_diff = 0.0 
    sum_sq_diff_centered = 0.0
    sum_a = 0.0
    sum_sq_a = 0.0
    sum_b = 0.0
    sum_sq_b = 0.0
    conv_keys = [k for k in keys_a if is_conv_weight(k, sd_a[k])]
    if len(conv_keys) == 0:
        raise ValueError('No conv weights found (expected 4D tensors ending with .weight).')
    for k in conv_keys:
        wa = sd_a[k].detach().to(device=device, dtype=dtype)
        wb = sd_b[k].detach().to(device=device, dtype=dtype)
        diff = (wa - wb)
        n = diff.numel()
        diff_mean = diff.mean().item()
        diff_std = diff.std(unbiased=False).item()
        diff_l2 = diff.norm().item()
        diff_rms = (diff.pow(2).mean()).sqrt().item()
        diff_linf = diff.abs().max().item()
        wa_std = wa.std(unbiased=False).item()
        wb_std = wb.std(unbiased=False).item()
        a_flat = wa.reshape(-1)
        b_flat = wb.reshape(-1)
        denom = (a_flat.norm() * b_flat.norm()).clamp_min(eps)
        cos = (a_flat @ b_flat / denom).item()

        rows.append({'name': k, 'numel': n, 'diff_mean': diff_mean, 'diff_std': diff_std, 'diff_l2': diff_l2, 'diff_rms': diff_rms, 
                     'diff_linf': diff_linf, 'std_a': wa_std, 'std_b': wb_std, 'cos': cos, 'shape': tuple(wa.shape)})
        total_elems += n
        sum_sq_diff += diff.pow(2).sum().item()
        sum_diff += diff.sum().item()
        sum_a += wa.sum().item()
        sum_sq_a += wa.pow(2).sum().item()
        sum_b += wb.sum().item()
        sum_sq_b += wb.pow(2).sum().item()
    global_diff_mean = sum_diff / max(total_elems, 1)
    for k in conv_keys:
        wa = sd_a[k].detach().to(device=device, dtype=dtype)
        wb = sd_b[k].detach().to(device=device, dtype=dtype)
        diff = (wa - wb)
        sum_sq_diff_centered += (diff - global_diff_mean).pow(2).sum().item()
    global_diff_std = (sum_sq_diff_centered / max(total_elems, 1)) ** 0.5
    global_l2 = (sum_sq_diff) ** 0.5
    global_rms = (sum_sq_diff / max(total_elems, 1)) ** 0.5
    mean_a = sum_a / max(total_elems, 1)
    mean_b = sum_b / max(total_elems, 1)
    var_a = (sum_sq_a / max(total_elems, 1)) - mean_a**2
    var_b = (sum_sq_b / max(total_elems, 1)) - mean_b**2
    global_std_a = max(var_a, 0.0) ** 0.5
    global_std_b = max(var_b, 0.0) ** 0.5

    summary = {'num_conv_tensors': len(conv_keys), 'num_conv_params': total_elems, 'global_diff_mean': global_diff_mean, 
               'global_diff_std': global_diff_std, 'global_l2': global_l2, 'global_rms': global_rms, 'global_std_a': global_std_a, 
               'global_std_b': global_std_b}
    return rows, summary

if __name__ == '__main__':
    modelA = torchvision.models.resnet18(pretrained=False)
    modelB = torchvision.models.resnet18(pretrained=False)
    modelB.load_state_dict(torch.load('saved_models/orig_untrained_rn34_rn18_progressive_replace.pt'))
    sdA = modelA.state_dict()
    sdB = modelB.state_dict()
    rows, summary = compare_conv_weights(sdA, sdB, device='cpu')
    print(summary)
    # top-20 most different conv tensors by RMS diff
    for r in sorted(rows, key=lambda d: d['diff_rms'], reverse=True)[:20]:
        print(r['name'], r['shape'], 'diff_std', r['diff_std'], 'diff_rms', r['diff_rms'], 'std_a', r['std_a'], 'std_b', r['std_b'])