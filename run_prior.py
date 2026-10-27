import argparse
import shlex
import subprocess
import sys

from pathlib import Path

def get_run_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('family', choices = ['resnet', 'vit'])
    parser.add_argument('method', choices = ['baseline', 'CKA', 'KNN'])
    parser.add_argument('--data_dir', required = True)
    parser.add_argument('--guide', default = None)
    parser.add_argument('--depth', type = int, default = None)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--teacher_seed', type = int, default = 1000)
    parser.add_argument('--phase', choices = ['align', 'train', 'full'], default = 'full')
    parser.add_argument('--steps', type = int, default = 1000)
    parser.add_argument('--rep_batch', type = int, default = 128)
    parser.add_argument('--rep_lr', type = float, default = None)
    parser.add_argument('--rep_val_size', type = int, default = 4096)
    parser.add_argument('--rep_val_interval', type = int, default = 250)
    parser.add_argument('--align_mix', type = float, default = 1.0)
    parser.add_argument('--prior_schedule', choices = ['cumulative', 'frozen'], default = 'cumulative')
    parser.add_argument('--prior_start_stage', type = int, default = 1)
    parser.add_argument('--prior_update_scope', choices = ['prefix', 'suffix'], default = 'prefix')
    parser.add_argument('--prior_min_rank_ratio', type = float, default = 0.0)
    parser.add_argument('--prior_norm', choices = ['batch', 'eval'], default = 'batch')
    parser.add_argument('--prior_descriptor', choices = ['pooled', 'spatial'], default = 'pooled')
    parser.add_argument('--epochs', type = int, default = None)
    parser.add_argument('--workers', type = int, default = 8)
    parser.add_argument('--exp_name', default = None)
    parser.add_argument('--amp', action = 'store_true')
    parser.add_argument('--dry_run', action = 'store_true')
    return parser.parse_args()

def build_command(args):
    resnet = args.family == 'resnet'
    guide = args.guide or ('rn50' if resnet else 'vit_tiny')
    allowed = ['rn18', 'rn34', 'rn50', 'rn101', 'rn152'] if resnet else ['vit_tiny', 'vit_small', 'vit_base', 'vit_large']
    if guide not in allowed or (resnet and args.depth is not None):
        raise ValueError('The guide must match the family; --depth is only available for ViTs')
    if not resnet and args.depth is not None and args.depth < 4:
        raise ValueError('The ViT guide must have at least four blocks')
    start = getattr(args, 'prior_start_stage', 1)
    scope = getattr(args, 'prior_update_scope', 'prefix')
    if not 1 <= start <= 4 or scope not in ['prefix', 'suffix']:
        raise ValueError('Use prior_start_stage in 1-4 and prior_update_scope prefix or suffix')
    if scope != 'prefix' and args.prior_schedule != 'cumulative':
        raise ValueError('Suffix updates require the cumulative schedule')
    if args.method == 'baseline' and (start != 1 or scope != 'prefix'):
        raise ValueError('Alignment stage settings do not apply to a baseline')
    depth = args.depth
    if not resnet and guide == 'vit_tiny' and depth is None:
        depth = 4 if args.method == 'baseline' else 12
    guide_tag = guide if depth is None else f'{guide}_d{depth}'
    student = 'rn18' if resnet else 'vit_tiny'
    name = f'prior_v2_{guide_tag}_{student}_{args.method.lower()}_s{args.seed}_g{args.teacher_seed}'
    if args.prior_schedule != 'cumulative':
        name += '_frozen'
    if args.align_mix != 1.0:
        name += f'_mix{args.align_mix:g}'
    if args.prior_min_rank_ratio != 0.0:
        name += f'_rank{args.prior_min_rank_ratio:g}'
    if args.prior_norm != 'batch':
        name += '_evalnorm'
    if args.method == 'baseline':
        name = f'prior_v2_{student}_baseline_s{args.seed}'
    if args.amp:
        name += '_bf16'
    if args.prior_descriptor != 'pooled' and args.method != 'baseline':
        name += '_spatial'
    if start != 1:
        name += f'_from{start}'
    if scope != 'prefix':
        name += '_suffix'
    name = args.exp_name or name
    method = 'CKA' if args.method == 'baseline' else args.method
    epochs = args.epochs if args.epochs is not None else (100 if resnet else 300)
    lr = args.rep_lr if args.rep_lr is not None else (1e-4 if resnet else 5e-5)
    command = [sys.executable, 'theseus.py', '--optimization_prior', '--untrained', '--exp_name', name,
               '--setting', 'imagenet', '--imagenet', 'imagenet-1k', '--data_dir', args.data_dir,
               '--model', guide, '--replace', 'progressive_align_resnet' if resnet else 'progressive_align_vits',
               '--seed', str(args.seed), '--teacher_seed', str(args.teacher_seed), '--num_workers', str(args.workers),
               '--repdist', method, '--cka_loss', 'debiased', '--knn_loss', 'soft', '--knn_k', '20', '--knn_temp', '0.2',
               '--rep_batch', str(args.rep_batch), '--rep_val_size', str(args.rep_val_size), '--rep_split_seed', '0',
               '--rep_val_interval', str(args.rep_val_interval), '--max_stage_steps', str(args.steps),
               '--progressive_lrs', str(lr), '--progressive_epochs', '1', '--rep_wd', '0', '--rep_scheduler',
               '--pool_hw', '4', '--align_mix', str(args.align_mix), '--prior_norm', args.prior_norm,
               '--prior_schedule', args.prior_schedule, '--prior_min_rank_ratio', str(args.prior_min_rank_ratio), '--ft_batch', '256' if resnet else '128',
               '--ft_accum', '1' if resnet else '4', '--ft_optimizer', 'sgd' if resnet else 'adamw',
               '--ft_lr', '0.1' if resnet else '5e-4', '--ft_wd', '1e-4' if resnet else '0.05',
               '--ft_epochs', str(epochs), '--label_smoothing', '0.1', '--mixup_alpha', '0' if resnet else '0.8',
               '--ft_clip', '0' if resnet else '1', '--scheduler', '--scheduler_type', 'cosine',
               '--warmup_epochs', '5' if resnet else '20']
    if args.prior_descriptor != 'pooled':
        command += ['--prior_descriptor', args.prior_descriptor]
    if start != 1:
        command += ['--prior_start_stage', str(start)]
    if scope != 'prefix':
        command += ['--prior_update_scope', scope]
    if not resnet:
        command += ['--student_vit', 'vit_tiny']
    if depth is not None:
        command += ['--teacher_depth', str(depth)]
    if args.method == 'baseline':
        command += ['--baseline']
    if args.phase == 'align':
        command += ['--align_only']
    elif args.phase == 'train':
        command += ['--reload_ft']
    if args.amp:
        command += ['--use_amp']
    return command

def main():
    args = get_run_args()
    args.data_dir = str(Path(args.data_dir).expanduser().resolve())
    command = build_command(args)
    print(shlex.join(command), flush = True)
    if not args.dry_run:
        subprocess.run(command, cwd = Path(__file__).resolve().parent, check = True)

if __name__ == '__main__':
    main()
