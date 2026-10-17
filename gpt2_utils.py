import json
import torch.nn as nn

from transformers.models.gpt2.modeling_gpt2 import GPT2Block

def _unwrap_gpt2(model):
    if hasattr(model, 'module'):
        return _unwrap_gpt2(model.module)
    if hasattr(model, 'model') and hasattr(model.model, 'transformer'):
        return _unwrap_gpt2(model.model)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model
    raise TypeError('Expected a GPT-2 LM/model or a wrapper around one.')

def _module_id_to_name_map(model):
    return {id(m): name for name, m in model.named_modules()}

def _depth_partitions(n_teacher, n_student):
    if n_teacher < 1:
        raise ValueError(f'n_teacher must be >= 1, got {n_teacher}')
    if n_student < 1:
        raise ValueError(f'n_student must be >= 1, got {n_student}')

    spans = []
    for i in range(n_student):
        start = int(i * n_teacher / n_student)
        end = int((i + 1) * n_teacher / n_student)
        end = max(end, start + 1)
        spans.append((start, end))
    return spans

def _pick_anchor_index(start, end, anchor='late'):
    if end <= start:
        raise ValueError(f'Invalid span [{start}, {end})')
    if anchor == 'early':
        return start
    if anchor == 'mid':
        return (start + end - 1) // 2
    if anchor == 'late':
        return end - 1
    raise ValueError(f'Unknown anchor={anchor}. Expected one of early, mid, late.')

def _dedupe_modules(modules):
    unique = []
    seen = set()
    for mod in modules:
        mid = id(mod)
        if mid in seen:
            continue
        seen.add(mid)
        unique.append(mod)
    return unique

def get_gpt2_blocks(model, model_name=''):
    gpt2 = _unwrap_gpt2(model)
    name = model_name or 'gpt2'
    if not hasattr(gpt2, 'transformer') or not hasattr(gpt2.transformer, 'h'):
        raise AttributeError(f'{name} does not have transformer.h.')

    blocks = list(gpt2.transformer.h.children())
    if len(blocks) == 0:
        raise ValueError(f'{name} had 0 transformer blocks.')
    if not all(isinstance(b, GPT2Block) for b in blocks):
        raise TypeError(f'{name}.transformer.h contained a non-GPT2Block module.')
    return blocks

def get_gpt2_input_embedding(model, model_name=''):
    gpt2 = _unwrap_gpt2(model)
    name = model_name or 'gpt2'
    if not hasattr(gpt2.transformer, 'wte') or not isinstance(gpt2.transformer.wte, nn.Module):
        raise AttributeError(f'{name} does not have transformer.wte.')
    return gpt2.transformer.wte

def get_gpt2_final_norm(model, model_name=''):
    gpt2 = _unwrap_gpt2(model)
    name = model_name or 'gpt2'
    if not hasattr(gpt2.transformer, 'ln_f') or not isinstance(gpt2.transformer.ln_f, nn.LayerNorm):
        raise AttributeError(f'{name} does not have transformer.ln_f as nn.LayerNorm.')
    return gpt2.transformer.ln_f

def get_gpt2_trainables(model, active_gids, *, include_embedding=True, include_final_ln=True):
    if not active_gids:
        return []

    gpt2 = _unwrap_gpt2(model)
    blocks = get_gpt2_blocks(gpt2)
    num_blocks = len(blocks)
    block_gid_offset = 1 if include_embedding else 0
    final_ln_gid = block_gid_offset + num_blocks if include_final_ln else None

    active_block_gids = [g for g in active_gids if block_gid_offset <= g < block_gid_offset + num_blocks]
    trainable_modules = []

    if include_embedding:
        touches_embedding = 0 in active_gids
        touches_earliest_block = len(active_block_gids) > 0 and min(active_block_gids) == block_gid_offset
        if touches_embedding or touches_earliest_block:
            if hasattr(gpt2.transformer, 'wte') and isinstance(gpt2.transformer.wte, nn.Module):
                trainable_modules.append(gpt2.transformer.wte)
            if hasattr(gpt2.transformer, 'wpe') and isinstance(gpt2.transformer.wpe, nn.Module):
                trainable_modules.append(gpt2.transformer.wpe)

    if active_block_gids:
        start_block = min(g - block_gid_offset for g in active_block_gids)
        end_block = max(g - block_gid_offset for g in active_block_gids)
        trainable_modules.extend(list(blocks[start_block:end_block + 1]))

    if include_final_ln and final_ln_gid in active_gids:
        trainable_modules.append(get_gpt2_final_norm(gpt2))

    return _dedupe_modules(trainable_modules)

def build_alignment_gpt2_small_to_large(teacher, student, group_size=3, *, include_embedding=True,
                                        include_final_ln=True, print_name_map=True):
    teacher = _unwrap_gpt2(teacher)
    student = _unwrap_gpt2(student)
    teacher_blocks = get_gpt2_blocks(teacher, model_name='teacher')
    student_blocks = get_gpt2_blocks(student, model_name='student')

    teacher_hook_modules = []
    student_hook_modules = []
    group_desc = []

    if include_embedding:
        teacher_hook_modules.append(get_gpt2_input_embedding(teacher, model_name='teacher'))
        student_hook_modules.append(get_gpt2_input_embedding(student, model_name='student'))
        group_desc.append('token embedding')

    spans = _depth_partitions(len(teacher_blocks), len(student_blocks))
    for s_idx, (s_block, (t_start, t_end)) in enumerate(zip(student_blocks, spans)):
        t_idx = _pick_anchor_index(t_start, t_end, anchor='late')
        teacher_hook_modules.append(teacher_blocks[t_idx])
        student_hook_modules.append(s_block)
        group_desc.append(f'student block {s_idx} <- teacher block {t_idx} (span [{t_start}, {t_end - 1}], anchor=late)')

    if include_final_ln:
        teacher_hook_modules.append(get_gpt2_final_norm(teacher, model_name='teacher'))
        student_hook_modules.append(get_gpt2_final_norm(student, model_name='student'))
        group_desc.append('final layernorm')

    schedule = list(range(len(student_hook_modules)))
    print(
        f'[gpt2] Teacher has {len(teacher_blocks)} blocks, '
        f'Student has {len(student_blocks)} blocks, '
        f'include_embedding={include_embedding}, '
        f'include_final_ln={include_final_ln}, '
        f'anchor=late'
    )
    for gid, (t_mod, s_mod, desc) in enumerate(zip(teacher_hook_modules, student_hook_modules, group_desc)):
        print(f'  Group {gid}: student {type(s_mod).__name__} <- teacher {type(t_mod).__name__} | {desc}')

    if print_name_map:
        t_name = _module_id_to_name_map(teacher)
        s_name = _module_id_to_name_map(student)
        student_to_teacher = {}
        teacher_to_student = {}
        for t_mod, s_mod in zip(teacher_hook_modules, student_hook_modules):
            s_mod_name = s_name.get(id(s_mod), f'<unnamed:{type(s_mod).__name__}>')
            t_mod_name = t_name.get(id(t_mod), f'<unnamed:{type(t_mod).__name__}>')
            student_to_teacher[s_mod_name] = t_mod_name
            teacher_to_student[t_mod_name] = s_mod_name

        print('\n[Alignment name map] student_module_name -> teacher_module_name')
        print(json.dumps(student_to_teacher, indent=2, sort_keys=True))
        print('\n[Alignment name map] teacher_module_name -> student_module_name')
        print(json.dumps(teacher_to_student, indent=2, sort_keys=True))

    return schedule, teacher_hook_modules, student_hook_modules