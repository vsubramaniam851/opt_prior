import argparse
import gc
import json
import torch
import torchvision

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from prior_utils import make_prior_model, prior_fingerprint, prior_mode, prior_endpoints, prior_batch

def get_diagnostic_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices = ['resnet', 'vit', 'all'], default = 'all')
    parser.add_argument('--device', choices = ['cpu', 'cuda'], default = 'cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch', type = int, default = 8)
    parser.add_argument('--threads', type = int, default = 2)
    parser.add_argument('--output', default = 'prior_shape_diagnostics.json')
    return parser.parse_args()

def diagnose(args):
    if args.batch < 4 or args.threads < 1:
        raise ValueError('Use a batch of at least four images and at least one CPU thread')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    cases = []
    if args.family in ['resnet', 'all']:
        cases += [('rn18', guide, None) for guide in ['rn50', 'rn101', 'rn152']]
    if args.family in ['vit', 'all']:
        cases += [('vit_tiny', 'vit_tiny', depth) for depth in [8, 12, 24]]
        cases += [('vit_tiny', guide, None) for guide in ['vit_base', 'vit_large']]
    generator = torch.Generator().manual_seed(91)
    images = torch.randn(args.batch, 3, 224, 224, generator = generator).to(device)
    config = SimpleNamespace(use_amp = False, pool_hw = 4, repdist = 'CKA', cka_loss = 'debiased', knn_k = 3, knn_temp = 0.2)
    report = {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'device': str(device),
              'images': 'synthetic Gaussian tensors, not ImageNet', 'shape': list(images.shape),
              'normalization': 'batch statistics without running-statistic updates', 'knn_k': 3, 'knn_temperature': 0.2,
              'tests': [], 'complete': False}
    path = Path(args.output)
    path.parent.mkdir(parents = True, exist_ok = True)
    for student_name, guide_name, depth in cases:
        teacher = make_prior_model(guide_name, 1000, depth).to(device)
        student = make_prior_model(student_name, 0).to(device)
        teacher.requires_grad_(False)
        guide_hash = prior_fingerprint(teacher)
        student_indices, teacher_indices = prior_endpoints(teacher, student, 3, 'cumulative')
        head_student = student.fc if student_name == 'rn18' else student.model.heads
        head_teacher = teacher.fc if student_name == 'rn18' else teacher.model.heads
        stem = student.conv1 if student_name == 'rn18' else student.model.conv_proj
        for method in ['CKA', 'KNN']:
            config.repdist = method
            student.zero_grad(set_to_none = True)
            with patch.object(head_teacher, 'forward', side_effect = AssertionError('Guide head was used')):
                with patch.object(head_student, 'forward', side_effect = AssertionError('Student head was used')):
                    with prior_mode(teacher), prior_mode(student):
                        loss, losses, _, _ = prior_batch(config, teacher, student, images, student_indices, teacher_indices)
                        loss.backward()
            gradients = [p.grad for p in student.parameters() if p.grad is not None]
            finite = bool(torch.isfinite(loss)) and all(bool(g.isfinite().all()) for g in gradients)
            unchanged = guide_hash == prior_fingerprint(teacher)
            if not finite or not unchanged or stem.weight.grad.norm().item() == 0 or any(p.grad is not None for p in teacher.parameters()):
                raise AssertionError('Production-shape alignment diagnostic failed')
            result = {'student': student_name, 'guide': guide_name, 'guide_depth_override': depth, 'method': method,
                      'endpoints': [[i + 1, j + 1] for i, j in zip(student_indices, teacher_indices)],
                      'loss': loss.item(), 'endpoint_losses': [value.item() for value in losses],
                      'stem_gradient_norm': stem.weight.grad.norm().item(), 'finite_gradients': finite, 'guide_unchanged': unchanged,
                      'classifier_heads_unused': True}
            report['tests'].append(result)
            path.write_text(json.dumps(report, indent = 2))
            print(json.dumps(result), flush = True)
            del loss, losses, gradients, _
        del teacher, student, head_teacher, head_student, stem
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    report['complete'] = True
    path.write_text(json.dumps(report, indent = 2))
    return report

if __name__ == '__main__':
    diagnose(get_diagnostic_args())
