import argparse
import gc
import json
import torch

from pathlib import Path
from args import get_args
from prior_utils import make_prior_model, prior_batch, prior_fingerprint, prior_mode, prior_named_parameters, prior_schedule
from prior_utils import prior_trainable_parameters, prior_view_geometry
from run_prior import build_command, get_run_args
from slurm_scripts.run_prior_array import parse_with

def check(device):
    torch.set_num_threads(2)
    rows = []
    for family in ['resnet', 'vit']:
        for method in ['CKA', 'KNN']:
            for mode in ['separate', 'shared']:
                launch = parse_with(get_run_args, [family, method, '--data_dir', '/unused', '--prior_descriptor', 'spatial_only',
                                   '--prior_start_stage', '3', '--prior_update_scope', 'suffix', '--prior_view_mode', mode])
                args = parse_with(get_args, build_command(launch)[2:])
                args.knn_k = 3
                teacher = make_prior_model('rn50' if family == 'resnet' else 'vit_tiny', 1000, None if family == 'resnet' else 12).to(device)
                student = make_prior_model('rn18' if family == 'resnet' else 'vit_tiny', 0).to(device)
                before = prior_fingerprint(teacher)
                stage = prior_schedule(teacher, student, 'cumulative', 3, 'suffix')[-1]
                prior_trainable_parameters(student, stage[2])
                images = tuple(torch.randn(8, 3, 224, 224, generator = torch.Generator().manual_seed(39 + i)).to(device) for i in range(2))
                def forbidden(*args, **kwargs):
                    raise AssertionError('Classifier head was called')
                if family == 'resnet':
                    teacher.fc.forward = forbidden
                    student.fc.forward = forbidden
                else:
                    teacher.model.heads.forward = forbidden
                    student.model.heads.forward = forbidden
                targets = [2, 3] if family == 'resnet' else [8, 11]
                with prior_mode(teacher), prior_mode(student):
                    loss, _, features, guides = prior_batch(args, teacher, student, images, [2, 3], targets)
                    loss.backward()
                active = [(name, p, kind) for name, p, kind in prior_named_parameters(student) if p.requires_grad]
                connected = [(name, p) for name, p, _ in active if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0]
                if len(connected) != len(active) or not torch.isfinite(loss):
                    raise AssertionError('Missing or non-finite active-parameter gradient')
                if before != prior_fingerprint(teacher) or any(p.grad is not None for p in teacher.parameters()):
                    raise AssertionError('Guide changed or received gradients')
                metrics = prior_view_geometry([x.detach() for x in features[-1]], guides[-1], args)
                row = {'family': family, 'method': method, 'mode': mode, 'batch_per_view': 8, 'size': 224,
                       'loss': loss.item(), 'active_tensors': len(active), 'nonzero_finite_gradient_tensors': len(connected),
                       'guide_unchanged': True, 'classifiers_unused': True, 'metrics': metrics}
                print(json.dumps({key: value for key, value in row.items() if key != 'metrics'}), flush = True)
                rows.append(row)
                del active, connected, stage, images, features, guides, teacher, student, loss
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices = ['cpu', 'cuda'], default = 'cpu')
    parser.add_argument('--output', default = 'prior_views_actual_checks.json')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    report = {'torch': torch.__version__, 'device': args.device, 'synthetic': True, 'rows': check(torch.device(args.device))}
    Path(args.output).write_text(json.dumps(report, indent = 2, allow_nan = False))

if __name__ == '__main__':
    main()
