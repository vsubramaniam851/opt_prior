import torch.nn as nn

from modules import LinearizedConv2d, LinearizedVIT, MemEffAttention, MLP, LinearAttention, SwitchMoE, RNNAtt, ConvNextPatchMLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

REPLACERS = {
    nn.Conv2d: (LinearizedConv2d, ConvNextPatchMLP),
    nn.MultiheadAttention: LinearAttention,
    nn.Linear: nn.Conv2d,
    MemEffAttention: MLP,
    Qwen2Attention: SwitchMoE,
    GPT2Attention: RNNAtt
}