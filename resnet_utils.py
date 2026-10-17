import json
import torch.nn as nn

from torchvision.models.resnet import Bottleneck as TVBottleneck, BasicBlock as TVBasicBlock
from modules import BasicBlockCompat

def stage_names():
    return ['layer1', 'layer2', 'layer3', 'layer4']

RESNET_STAGE_BLOCKS = (TVBottleneck, TVBasicBlock, BasicBlockCompat)

def _anchored_indices(n_source, n_target, anchor):
    if n_target > n_source:
        raise ValueError(f'Cannot choose {n_target} anchors from {n_source} teacher blocks.')
    if anchor == 'early':
        return list(range(n_target))
    if anchor == 'late':
        return list(range(n_source - n_target, n_source))
    if anchor == 'mid':
        return _evenly_spaced_indices(n_source, n_target)
    raise ValueError("anchor must be one of: 'early', 'mid', 'late'")

def get_stage_blocks(stage_seq, *, stage_name = ''):
    blocks = []
    for m in stage_seq.children():
        if isinstance(m, RESNET_STAGE_BLOCKS):
            blocks.append(m)
        else:
            raise TypeError(
                f'Unexpected module {type(m)} inside stage {stage_name}. '
                f'Expected one of: {[t.__name__ for t in RESNET_STAGE_BLOCKS]}'
            )
    if len(blocks) == 0:
        raise ValueError(f'Stage {stage_name} contained 0 recognized ResNet blocks.')
    return blocks

def _module_id_to_name_map(model):
    return {id(m): name for name, m in model.named_modules()}

def _evenly_spaced_indices(n_source, n_target):
    if n_source < 1:
        raise ValueError(f'n_source must be >= 1, got {n_source}')
    if n_target < 1:
        raise ValueError(f'n_target must be >= 1, got {n_target}')

    if n_target == 1:
        return [(n_source - 1) // 2]

    return [round(i * (n_source - 1) / (n_target - 1)) for i in range(n_target)]

def get_resnet_stem_conv(model, model_name = ''):
    name = model_name or 'resnet'

    if not hasattr(model, 'conv1') or not isinstance(model.conv1, nn.Conv2d):
        raise AttributeError(f'{name} did not have conv1 as nn.Conv2d.')

    return model.conv1

def get_block_last_conv(block):
    if isinstance(block, (TVBasicBlock, BasicBlockCompat)):
        if not hasattr(block, 'conv2') or not isinstance(block.conv2, nn.Conv2d):
            raise TypeError(f'Expected BasicBlock-like block to have conv2, got {type(block)}')
        return block.conv2

    if isinstance(block, TVBottleneck):
        if not hasattr(block, 'conv3') or not isinstance(block.conv3, nn.Conv2d):
            raise TypeError(f'Expected Bottleneck block to have conv3, got {type(block)}')
        return block.conv3

    raise TypeError(f'Unknown block type: {type(block)}')

def _get_resnet_block_span(active_gids):
    block_gids = [g for g in active_gids if g >= 1]
    if len(block_gids) == 0:
        return None
    return min(block_gids), max(block_gids)

def get_resnet_trainables(model, active_gids, context_blocks = 0):
    if not active_gids:
        return []

    trainable_modules = []
    if 0 in active_gids:
        if hasattr(model, 'conv1'):
            trainable_modules.append(model.conv1)
        if hasattr(model, 'bn1'):
            trainable_modules.append(model.bn1)

    block_span = _get_resnet_block_span(active_gids)
    if block_span is None:
        return trainable_modules

    start_gid, end_gid = block_span
    start_gid = max(1, start_gid - context_blocks)
    gid = 1
    for lname in stage_names():
        stage = getattr(model, lname)
        for block in get_stage_blocks(stage, stage_name = lname):
            if start_gid <= gid <= end_gid:
                trainable_modules.append(block)
            gid += 1

    return trainable_modules

def build_alignment_resnet_lastconv(teacher, student, anchor = 'mid', print_name_map = True):
    teacher_hook_modules = [[get_resnet_stem_conv(teacher, model_name = 'teacher')]]
    student_hook_modules = [get_resnet_stem_conv(student, model_name = 'student')]

    for lname in stage_names():
        if not hasattr(teacher, lname) or not hasattr(student, lname):
            raise AttributeError(f'Both teacher and student must have stage {lname}.')

        t_blocks = get_stage_blocks(getattr(teacher, lname), stage_name = f'teacher.{lname}')
        s_blocks = get_stage_blocks(getattr(student, lname), stage_name = f'student.{lname}')
        t_indices = _anchored_indices(len(t_blocks), len(s_blocks), anchor)

        for s_block, t_idx in zip(s_blocks, t_indices):
            teacher_hook_modules.append([get_block_last_conv(t_blocks[t_idx])])
            student_hook_modules.append(get_block_last_conv(s_block))

    schedule = list(range(len(student_hook_modules)))

    print(f'[Alignment] Created {len(schedule)} alignment groups (resnet conv1 + last pre-activation block convs)')
    for i, (t_list, s_mod) in enumerate(zip(teacher_hook_modules, student_hook_modules)):
        print(f'  Group {i}: student {type(s_mod).__name__} <- 1 teacher conv')

    if print_name_map:
        t_name = _module_id_to_name_map(teacher)
        s_name = _module_id_to_name_map(student)

        student_to_teacher = {}
        for t_list, s_mod in zip(teacher_hook_modules, student_hook_modules):
            s_mod_name = s_name.get(id(s_mod), f'<unnamed:{type(s_mod).__name__}>')
            t_mod_names = [t_name.get(id(tm), f'<unnamed:{type(tm).__name__}>') for tm in t_list]
            student_to_teacher[s_mod_name] = t_mod_names

        print('\n[Alignment name map] student_module_name -> [teacher_module_names]')
        print(json.dumps(student_to_teacher, indent = 2, sort_keys = True))

    return schedule, teacher_hook_modules, student_hook_modules

# Backward-compatible wrappers. Old conv-specific args are intentionally ignored.
def build_alignment_resnet_conv(teacher, student, supervise_all_convs = True, *, anchor = 'mid',
                                like_with_like = False, print_name_map = True):
    return build_alignment_resnet_lastconv(teacher, student, anchor = anchor, print_name_map = print_name_map)

def build_alignment_rn50_rn18(teacher, student, *, anchor = 'mid', supervise_all_convs = True,
                              like_with_like = False):
    return build_alignment_resnet_lastconv(teacher, student, anchor = anchor, print_name_map = True)
