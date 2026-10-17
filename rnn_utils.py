import torch.nn as nn

def build_alignment_rnn_rn18(teacher_rnn, student_rn18):
    rnn_layers = list(teacher_rnn.rnn)
    rn18_blocks = [
        student_rn18.layer1[0], student_rn18.layer1[1],
        student_rn18.layer2[0], student_rn18.layer2[1],
        student_rn18.layer3[0], student_rn18.layer3[1],
        student_rn18.layer4[0], student_rn18.layer4[1],
    ]
    schedule = list(range(len(rn18_blocks)))
    orig_hook_modules = {i: rnn_layers[min(i, len(rnn_layers) - 1)] for i in schedule}
    repl_hook_modules = {i: rn18_blocks[i] for i in schedule}
    return schedule, orig_hook_modules, repl_hook_modules