import json
import torch.nn as nn

def _unwrap_convnextv2(model):
    if hasattr(model, 'module'):
        return _unwrap_convnextv2(model.module)
    if hasattr(model, 'model'):
        inner = model.model
        if hasattr(inner, 'stem') and hasattr(inner, 'stages'):
            return inner
    if hasattr(model, 'stem') and hasattr(model, 'stages'):
        return model
    raise TypeError('Expected a timm ConvNeXt/ConvNeXtV2 model or a wrapper around one.')

def get_convnextv2_block_branch_end(block):
    if hasattr(block, 'mlp') and hasattr(block.mlp, 'fc2') and isinstance(block.mlp.fc2, nn.Conv2d):
        return block.mlp.fc2
    raise TypeError(f'Expected ConvNeXtV2 block.mlp.fc2 to be nn.Conv2d, got {type(block)}')

def get_convnextv2_block_last_conv(block):
    if hasattr(block, 'conv_dw') and isinstance(block.conv_dw, nn.Conv2d):
        return block.conv_dw
    raise TypeError(f'Expected ConvNeXtV2 block to have conv_dw, got {type(block)}')

def get_convnextv2_stem_conv(model, model_name = ''):
    stem = get_convnextv2_stem(model, model_name = model_name)
    if isinstance(stem, nn.Conv2d):
        return stem
    for m in stem.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise TypeError(f'{model_name or "convnextv2"} stem did not contain any nn.Conv2d.')

def get_convnextv2_stem(model, model_name = ''):
    cnx = _unwrap_convnextv2(model)
    name = model_name or 'convnextv2'
    if not hasattr(cnx, 'stem') or not isinstance(cnx.stem, nn.Module):
        raise AttributeError(f'{name} did not have stem.')
    return cnx.stem

def get_convnextv2_stages(model, model_name = ''):
    cnx = _unwrap_convnextv2(model)
    name = model_name or 'convnextv2'
    if not hasattr(cnx, 'stages'):
        raise AttributeError(f'{name} did not have stages.')
    stages = list(cnx.stages.children())
    if len(stages) == 0:
        raise ValueError(f'{name} had 0 stages.')
    for i, stage in enumerate(stages):
        if not hasattr(stage, 'downsample') or not hasattr(stage, 'blocks'):
            raise TypeError(f'Unexpected stage {i} inside {name}. Expected stage to have downsample and blocks.')
    return stages

def get_convnextv2_stage_blocks(stage, *, stage_name = ''):
    if not hasattr(stage, 'blocks'):
        raise AttributeError(f'{stage_name or "stage"} did not have blocks.')
    blocks = list(stage.blocks.children())
    if len(blocks) == 0:
        raise ValueError(f'{stage_name or "stage"} had 0 blocks.')
    return blocks

def get_convnextv2_final_norm(model, model_name = ''):
    cnx = _unwrap_convnextv2(model)
    name = model_name or 'convnextv2'
    if hasattr(cnx, 'norm_pre') and isinstance(cnx.norm_pre, nn.Module):
        return cnx.norm_pre
    raise AttributeError(f'{name} did not have norm_pre.')

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

def _get_convnextv2_block_span(active_gids, final_norm_gid):
    block_gids = [g for g in active_gids if 1 <= g < final_norm_gid]
    if len(block_gids) == 0:
        return None
    return min(block_gids), max(block_gids)

def get_convnextv2_trainables(model, active_gids, context_blocks = 0):
    if not active_gids:
        return []
    cnx = _unwrap_convnextv2(model)
    stages = get_convnextv2_stages(cnx)
    trainable_modules = []
    if 0 in active_gids:
        trainable_modules.append(get_convnextv2_stem(cnx))

    final_norm_gid = 1 + sum(len(get_convnextv2_stage_blocks(stage, stage_name = f'stage{i}')) for i, stage in enumerate(stages))
    block_span = _get_convnextv2_block_span(active_gids, final_norm_gid)
    gid = 1
    for i, stage in enumerate(stages):
        blocks = get_convnextv2_stage_blocks(stage, stage_name = f'stage{i}')
        stage_start = gid
        stage_end = gid + len(blocks) - 1

        if block_span is not None:
            start_gid, end_gid = block_span
            start_gid = max(1, start_gid - context_blocks)
            overlap_start = max(stage_start, start_gid)
            overlap_end = min(stage_end, end_gid)
            if overlap_start <= overlap_end:
                local_start = overlap_start - stage_start
                local_end = overlap_end - stage_start
                if local_start == 0 and hasattr(stage, 'downsample') and stage.downsample is not None:
                    trainable_modules.append(stage.downsample)
                trainable_modules.extend(list(blocks[local_start:local_end + 1]))
        gid += len(blocks)
    if final_norm_gid in active_gids:
        trainable_modules.append(get_convnextv2_final_norm(cnx))
    return trainable_modules

def build_alignment_convnextv2(teacher, student, branch_end = False, print_name_map = True):
    teacher_cnx = _unwrap_convnextv2(teacher)
    student_cnx = _unwrap_convnextv2(student)
    teacher_hook_modules = [[get_convnextv2_stem_conv(teacher_cnx, model_name = 'teacher')]]
    student_hook_modules = [get_convnextv2_stem_conv(student_cnx, model_name = 'student')]
    teacher_stages = get_convnextv2_stages(teacher_cnx, model_name = 'teacher')
    student_stages = get_convnextv2_stages(student_cnx, model_name = 'student')
    if len(teacher_stages) != len(student_stages):
        raise ValueError('Teacher and student must have the same number of ConvNeXtV2 stages.')
    for stage_idx, (t_stage, s_stage) in enumerate(zip(teacher_stages, student_stages)):
        t_blocks = get_convnextv2_stage_blocks(t_stage, stage_name = f'teacher.stage{stage_idx}')
        s_blocks = get_convnextv2_stage_blocks(s_stage, stage_name = f'student.stage{stage_idx}')
        t_indices = _evenly_spaced_indices(len(t_blocks), len(s_blocks))
        for s_block, t_idx in zip(s_blocks, t_indices):
            if branch_end:
                teacher_hook_modules.append([get_convnextv2_block_branch_end(t_blocks[t_idx])])
                student_hook_modules.append(get_convnextv2_block_branch_end(s_block))
            else:
                teacher_hook_modules.append([get_convnextv2_block_last_conv(t_blocks[t_idx])])
                student_hook_modules.append(get_convnextv2_block_last_conv(s_block))
    schedule = list(range(len(student_hook_modules)))
    print('[Alignment] Created ConvNeXtV2 alignment groups (stem conv + block conv_dw)')
    for i, (t_list, s_mod) in enumerate(zip(teacher_hook_modules, student_hook_modules)):
        print(f'  Group {i}: student {type(s_mod).__name__} <- 1 teacher module')
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
