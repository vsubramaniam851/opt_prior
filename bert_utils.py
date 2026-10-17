import json
import torch.nn as nn
import transformers

from transformers.models.bert.modeling_bert import BertEmbeddings, BertLayer, BertPooler

def _unwrap_bert(model):
    if hasattr(model, 'module'):
        return _unwrap_bert(model.module)
    if hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
        return model.bert
    if hasattr(model, 'model') and hasattr(model.model, 'bert'):
        return _unwrap_bert(model.model.bert)
    if hasattr(model, 'model') and hasattr(model.model, 'encoder') and hasattr(model.model, 'embeddings'):
        return model.model
    if hasattr(model, 'encoder') and hasattr(model, 'embeddings'):
        return model
    raise TypeError('Expected a BERT model or a wrapper around one.')

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

def _pick_anchor_index(start, end, anchor = 'late'):
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

def get_bert_embeddings(model, model_name = ''):
    bert = _unwrap_bert(model)
    name = model_name or 'bert'
    if not hasattr(bert, 'embeddings') or not isinstance(bert.embeddings, BertEmbeddings):
        raise AttributeError(f'{name} does not have BertEmbeddings at embeddings.')
    return bert.embeddings

def get_bert_layers(model, model_name = ''):
    bert = _unwrap_bert(model)
    name = model_name or 'bert'
    if not hasattr(bert, 'encoder') or not hasattr(bert.encoder, 'layer'):
        raise AttributeError(f'{name} does not have encoder.layer.')

    blocks = list(bert.encoder.layer.children())
    if len(blocks) == 0:
        raise ValueError(f'{name} had 0 encoder layers.')
    if not all(isinstance(b, BertLayer) for b in blocks):
        raise TypeError(f'{name}.encoder.layer contained a non-BertLayer module.')
    return blocks

def get_bert_pooler(model, model_name = ''):
    bert = _unwrap_bert(model)
    name = model_name or 'bert'
    if not hasattr(bert, 'pooler') or bert.pooler is None:
        raise AttributeError(f'{name} does not have a pooler.')
    if not isinstance(bert.pooler, BertPooler):
        raise TypeError(f'{name}.pooler was not a BertPooler.')
    return bert.pooler

def get_bert_trainables(model, active_gids, *, include_embedding = True, include_pooler = True):
    if not active_gids:
        return []

    bert = _unwrap_bert(model)
    blocks = get_bert_layers(bert)
    num_blocks = len(blocks)
    block_gid_offset = 1 if include_embedding else 0
    pooler_gid = block_gid_offset + num_blocks if include_pooler else None

    active_block_gids = [g for g in active_gids if block_gid_offset <= g < block_gid_offset + num_blocks]
    trainable_modules = []

    if include_embedding:
        touches_embedding = 0 in active_gids
        touches_earliest_block = len(active_block_gids) > 0 and min(active_block_gids) == block_gid_offset
        if touches_embedding or touches_earliest_block:
            trainable_modules.append(get_bert_embeddings(bert))

    if active_block_gids:
        start_block = min(g - block_gid_offset for g in active_block_gids)
        end_block = max(g - block_gid_offset for g in active_block_gids)
        trainable_modules.extend(list(blocks[start_block:end_block + 1]))

    if include_pooler and pooler_gid in active_gids:
        trainable_modules.append(get_bert_pooler(bert))

    return _dedupe_modules(trainable_modules)

def build_alignment_bert_base_to_large(teacher, student, *, include_embedding = True,
                                       include_pooler = True, anchor = 'late', print_name_map = True):
    teacher = _unwrap_bert(teacher)
    student = _unwrap_bert(student)
    teacher_blocks = get_bert_layers(teacher, model_name = 'teacher')
    student_blocks = get_bert_layers(student, model_name = 'student')

    teacher_hook_modules = []
    student_hook_modules = []
    group_desc = []

    if include_embedding:
        teacher_hook_modules.append(get_bert_embeddings(teacher, model_name = 'teacher'))
        student_hook_modules.append(get_bert_embeddings(student, model_name = 'student'))
        group_desc.append('embeddings')

    spans = _depth_partitions(len(teacher_blocks), len(student_blocks))
    for s_idx, (s_block, (t_start, t_end)) in enumerate(zip(student_blocks, spans)):
        t_idx = _pick_anchor_index(t_start, t_end, anchor = anchor)
        teacher_hook_modules.append(teacher_blocks[t_idx])
        student_hook_modules.append(s_block)
        group_desc.append(f'student layer {s_idx} <- teacher layer {t_idx} (span [{t_start}, {t_end - 1}], anchor={anchor})')

    if include_pooler:
        teacher_hook_modules.append(get_bert_pooler(teacher, model_name = 'teacher'))
        student_hook_modules.append(get_bert_pooler(student, model_name = 'student'))
        group_desc.append('pooler')

    schedule = list(range(len(student_hook_modules)))
    print(
        f'[bert] Teacher has {len(teacher_blocks)} layers, '
        f'Student has {len(student_blocks)} layers, '
        f'include_embedding={include_embedding}, '
        f'include_pooler={include_pooler}, '
        f'anchor={anchor}'
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
        print(json.dumps(student_to_teacher, indent = 2, sort_keys = True))
        print('\n[Alignment name map] teacher_module_name -> student_module_name')
        print(json.dumps(teacher_to_student, indent = 2, sort_keys = True))

    return schedule, teacher_hook_modules, student_hook_modules
