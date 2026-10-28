import argparse
import json
import torch

from pathlib import Path
from audit_prior_mechanism import gradient_probe
from prior_utils import make_prior_model, prior_fingerprint, prior_state
from test_optimization_prior import test_args

def forbidden(*args):
    raise RuntimeError('Classifier head was called')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default = 'bn_actual_gradient_checks.json')
    output = Path(parser.parse_args().output)
    torch.set_num_threads(2)
    reports = []
    for method in ['CKA', 'KNN']:
        for scope in ['prefix', 'suffix']:
            for affine in ['train', 'fixed']:
                teacher = make_prior_model('rn50', 1000).requires_grad_(False)
                student = make_prior_model('rn18', 0)
                teacher.fc.forward = forbidden
                student.fc.forward = forbidden
                initial = prior_state(student)
                hashes = [prior_fingerprint(teacher), prior_fingerprint(student)]
                images = torch.randn(8, 3, 224, 224, generator = torch.Generator().manual_seed(80))
                args = test_args(repdist = method, prior_descriptor = 'spatial', prior_start_stage = 3,
                                 prior_update_scope = scope, prior_bn_affine = affine, pool_hw = 4)
                report = gradient_probe(args, teacher, student, images, initial)
                conv = report['branches']['total']['groups']['conv']
                assert conv['gradient_tensors'] == conv['nonzero_gradient_tensors'] == conv['trainable_tensors']
                assert conv['gradient_rms'] > 0
                assert hashes == [prior_fingerprint(teacher), prior_fingerprint(student)]
                assert all(param.grad is None for param in teacher.parameters())
                result = {'method': method, 'scope': scope, 'affine': affine, 'batch': 8, 'resolution': 224,
                          'knn_k': 3, 'states_unchanged': True, 'classifiers_used': False, **report}
                reports.append(result)
                print(method, scope, affine, 'nonzero conv tensors', conv['nonzero_gradient_tensors'], '/', conv['trainable_tensors'], flush = True)
    output.parent.mkdir(parents = True, exist_ok = True)
    output.write_text(json.dumps({'torch': torch.__version__,
             'device': 'cpu', 'synthetic_inputs': True, 'cases': reports}, indent = 2, allow_nan = False))

if __name__ == '__main__':
    main()
