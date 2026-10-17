import json
import torch.nn as nn

from torchvision.models.vision_transformer import VisionTransformer

def _unwrap_vit(model):
    if isinstance(model, VisionTransformer):
        return model
    if hasattr(model, 'module'):
        return _unwrap_vit(model.module)
    if hasattr(model, 'model') and isinstance(model.model, VisionTransformer):
        return model.model
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        return model
    if hasattr(model, 'model') and hasattr(model.model, 'encoder') and hasattr(model.model.encoder, 'layers'):
        return model.model
    raise TypeError('Expected a VisionTransformer or a wrapper around one.')

def get_vit_blocks(model, model_name = ''):
    vit = _unwrap_vit(model)
    name = model_name or 'vit'
    if not hasattr(vit, 'encoder') or not hasattr(vit.encoder, 'layers'):
        raise AttributeError(f'{name} does not have vit.encoder.layers.')

    blocks = []
    for m in vit.encoder.layers.children():
        if hasattr(m, 'ln_1') and hasattr(m, 'ln_2') and hasattr(m, 'mlp'):
            blocks.append(m)
        else:
            raise TypeError(f'Unexpected module {type(m)} inside {name}.encoder.layers.'
                            'Expected transformer blocks with ln_1, ln_2, and mlp.')

    if len(blocks) == 0:
        raise ValueError(f'{name} had 0 encoder blocks.')
    return blocks

def get_vit_endonly_layers(model, model_name = ''):
    vit = _unwrap_vit(model)
    name = model_name or 'vit'
    layers = []
    layers.append((get_vit_conv_layer(vit, model_name = name), 'conv'))
    for block in get_vit_blocks(vit, model_name = name):
        layers.append((block, 'block'))
    return layers

def get_vit_branch_end_layers(model, model_name = ''):
    vit = _unwrap_vit(model)
    name = model_name or 'vit'
    layers = []
    layers.append((get_vit_conv_layer(vit, model_name = name), 'conv'))
    for block in get_vit_blocks(vit, model_name = name):
        mlp_linears = []
        for m in block.mlp.modules():
            if isinstance(m, nn.Linear):
                mlp_linears.append(m)
        if len(mlp_linears) != 2:
            raise ValueError(f'Expected exactly 2 nn.Linear layers inside block.mlp, got {len(mlp_linears)}')
        layers.append((mlp_linears[1], 'branch'))
    return layers

def get_vit_conv_layer(model, model_name = ''):
    vit = _unwrap_vit(model)
    name = model_name or 'vit'

    if not hasattr(vit, 'conv_proj'):
        raise AttributeError(f'{name} does not have conv_proj.')

    if isinstance(vit.conv_proj, nn.Conv2d):
        return vit.conv_proj

    for m in vit.conv_proj.modules():
        if isinstance(m, nn.Conv2d):
            return m

    raise TypeError(f'{name}.conv_proj did not contain any nn.Conv2d.')

def get_layers_from_vit_block(block):
    layers = []

    if not hasattr(block, 'ln_1') or not isinstance(block.ln_1, nn.LayerNorm):
        raise TypeError('Expected block.ln_1 to be nn.LayerNorm')
    if not hasattr(block, 'ln_2') or not isinstance(block.ln_2, nn.LayerNorm):
        raise TypeError('Expected block.ln_2 to be nn.LayerNorm')
    if not hasattr(block, 'mlp'):
        raise TypeError('Expected block to have an mlp submodule')
    layers.append((block.ln_1, 'layernorm'))
    layers.append((block.ln_2, 'layernorm'))
    mlp_linears = []
    for m in block.mlp.modules():
        if isinstance(m, nn.Linear):
            mlp_linears.append(m)
    if len(mlp_linears) != 2:
        raise ValueError(f'Expected exactly 2 nn.Linear layers inside block.mlp, got {len(mlp_linears)}')
    layers.append((mlp_linears[0], 'linear'))
    layers.append((mlp_linears[1], 'linear'))

    return layers

def _vit_layout(model, model_name = '', end_only = False):
    vit = _unwrap_vit(model)
    name = model_name or 'vit'
    blocks = get_vit_blocks(vit, model_name = name)
    if end_only:
        layers_per_block = 1
    else:
        layers_per_block = len(get_layers_from_vit_block(blocks[0]))
    final_ln_gid = 1 + len(blocks) * layers_per_block
    return vit, blocks, layers_per_block, final_ln_gid

def _get_vit_block_span(active_gids, layers_per_block, final_ln_gid):
    block_gids = [g for g in active_gids if 1 <= g < final_ln_gid]
    if len(block_gids) == 0:
        return None
    start_block = min((g - 1) // layers_per_block for g in block_gids)
    end_block = max((g - 1) // layers_per_block for g in block_gids)
    return start_block, end_block

def get_vit_alignable_layers(model, model_name = ''):
    vit, blocks, _layers_per_block, _final_ln_gid = _vit_layout(model, model_name = model_name)
    name = model_name or 'vit'
    layers = []
    layers.append((get_vit_conv_layer(vit, model_name = name), 'conv'))
    for block in blocks:
        for layer, kind in get_layers_from_vit_block(block):
            layers.append((layer, kind))

    if not hasattr(vit, 'encoder') or not hasattr(vit.encoder, 'ln') or not isinstance(vit.encoder.ln, nn.LayerNorm):
        raise TypeError(f'{name}.encoder.ln was missing or was not nn.LayerNorm')

    layers.append((vit.encoder.ln, 'layernorm'))

    if len(layers) == 0:
        raise ValueError(f'{name} had 0 alignable layers.')
    return layers

def get_vit_global_trainable_params(model, active_gids = None, end_only = False):
    vit = _unwrap_vit(model)
    params = []

    if active_gids is not None:
        _vit, _blocks, layers_per_block, final_ln_gid = _vit_layout(vit, end_only = end_only)
        block_span = _get_vit_block_span(active_gids, layers_per_block, final_ln_gid)
        allow_global = 0 in active_gids or (block_span is not None and block_span[0] == 0)
        if not allow_global:
            return params

    if hasattr(vit, 'class_token') and vit.class_token is not None:
        params.append(vit.class_token)

    if hasattr(vit, 'encoder') and hasattr(vit.encoder, 'pos_embedding'):
        params.append(vit.encoder.pos_embedding)

    return params

def get_trainable_modules_up_to(model, active_gids, end_only = False, context_blocks = 0):
    if not active_gids:
        return []

    vit, blocks, layers_per_block, final_ln_gid = _vit_layout(model, end_only = end_only)

    trainable_modules = []
    if 0 in active_gids:
        trainable_modules.append(get_vit_conv_layer(vit))

    block_span = _get_vit_block_span(active_gids, layers_per_block, final_ln_gid)
    if block_span is not None:
        start_block, end_block = block_span
        start_block = max(0, start_block - context_blocks)
        trainable_modules.extend(list(blocks[start_block:end_block + 1]))

    if (not end_only) and final_ln_gid in active_gids:
        trainable_modules.append(vit.encoder.ln)

    return trainable_modules

def get_vit_trainables(model, active_gids, end_only = False, context_blocks = 0):
    return get_trainable_modules_up_to(model, active_gids, end_only = end_only, context_blocks = context_blocks), \
            get_vit_global_trainable_params(model, active_gids = active_gids, end_only = end_only)

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

def build_alignment_vits(teacher, student, like_with_like = False, print_name_map = True,
                         end_only = False, branch_end = False):
    if branch_end:
        teacher_layers = get_vit_branch_end_layers(teacher, model_name = 'teacher')
        student_layers = get_vit_branch_end_layers(student, model_name = 'student')
    elif end_only:
        teacher_layers = get_vit_endonly_layers(teacher, model_name = 'teacher')
        student_layers = get_vit_endonly_layers(student, model_name = 'student')
    else:
        teacher_layers = get_vit_alignable_layers(teacher, model_name = 'teacher')
        student_layers = get_vit_alignable_layers(student, model_name = 'student')

    teacher_hook_modules = []
    student_hook_modules = []

    num_teacher = len(teacher_layers)
    num_student = len(student_layers)

    print(
        f'[vit] Teacher has {num_teacher} alignable layers, '
        f'Student has {num_student} alignable layers, '
        f'like_with_like={like_with_like}'
    )

    if like_with_like:
        teacher_by_kind = {}
        student_by_kind = {}

        for mod, kind in teacher_layers:
            teacher_by_kind.setdefault(kind, []).append(mod)

        for mod, kind in student_layers:
            student_by_kind.setdefault(kind, []).append(mod)

        kind_to_teacher_indices = {}
        for kind, s_mods in student_by_kind.items():
            t_mods = teacher_by_kind.get(kind, [])
            if len(t_mods) == 0:
                raise ValueError(f'No teacher layers of kind {kind} were found.')
            kind_to_teacher_indices[kind] = _evenly_spaced_indices(len(t_mods), len(s_mods))

        kind_offsets = {}
        for s_mod, s_kind in student_layers:
            offset = kind_offsets.get(s_kind, 0)
            t_idx = kind_to_teacher_indices[s_kind][offset]
            t_mod = teacher_by_kind[s_kind][t_idx]

            teacher_hook_modules.append([t_mod])
            student_hook_modules.append(s_mod)

            kind_offsets[s_kind] = offset + 1
    else:
        teacher_indices = _evenly_spaced_indices(num_teacher, num_student)

        for s_idx, (s_mod, _s_kind) in enumerate(student_layers):
            t_mod = teacher_layers[teacher_indices[s_idx]][0]
            teacher_hook_modules.append([t_mod])
            student_hook_modules.append(s_mod)

    schedule = list(range(len(student_hook_modules)))

    print(
        f'[Alignment] Created {len(schedule)} alignment groups '
        f'(vit layers), like_with_like={like_with_like}'
    )
    for i, (t_list, s_mod) in enumerate(zip(teacher_hook_modules, student_hook_modules)):
        print(f'  Group {i}: student {type(s_mod).__name__} <- 1 teacher layer')

    if print_name_map:
        t_name = _module_id_to_name_map(teacher)
        s_name = _module_id_to_name_map(student)

        student_to_teacher = {}
        teacher_to_student = {}

        for t_list, s_mod in zip(teacher_hook_modules, student_hook_modules):
            s_mod_name = s_name.get(id(s_mod), f'<unnamed:{type(s_mod).__name__}>')
            t_mod_names = [t_name.get(id(tm), f'<unnamed:{type(tm).__name__}>') for tm in t_list]

            student_to_teacher[s_mod_name] = t_mod_names

            for t_mod_name in t_mod_names:
                teacher_to_student.setdefault(t_mod_name, []).append(s_mod_name)

        teacher_to_student_simple = {}
        for t_mod_name, s_mod_names in teacher_to_student.items():
            if len(s_mod_names) == 1:
                teacher_to_student_simple[t_mod_name] = s_mod_names[0]
            else:
                teacher_to_student_simple[t_mod_name] = s_mod_names

        print('\n[Alignment name map] student_module_name -> [teacher_module_names]')
        print(json.dumps(student_to_teacher, indent = 2, sort_keys = True))

        print('\n[Alignment name map] teacher_module_name -> student_module_name')
        print(json.dumps(teacher_to_student_simple, indent = 2, sort_keys = True))

    return schedule, teacher_hook_modules, student_hook_modules
