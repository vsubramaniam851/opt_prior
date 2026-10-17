import torch
import torch.nn as nn
import transformers

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

def _build_gpt2(n_layer, n_embd, n_head, vocab_size = 50257, n_positions = 1024, resid_pdrop = 0.1, embd_pdrop = 0.1, attn_pdrop = 0.1):
    assert n_embd % n_head == 0, 'n_embd must be divisible by n_head'

    config = GPT2Config(vocab_size = vocab_size, n_positions = n_positions, n_ctx = n_positions, n_embd = n_embd, n_layer = n_layer, n_head = n_head, 
			n_inner = 4 * n_embd, activation_function = 'gelu_new', resid_pdrop = resid_pdrop, embd_pdrop = embd_pdrop, attn_pdrop = attn_pdrop, 
                        bos_token_id = 50256, eos_token_id = 50256)

    model = GPT2LMHeadModel(config)
    return model

def gpt2_tiny(vocab_size = 50257, n_positions = 1024):
    return _build_gpt2(n_layer = 4, n_embd = 256, n_head = 4, vocab_size = vocab_size, n_positions = n_positions)

def gpt2_small(vocab_size = 50257, n_positions = 1024):
    return _build_gpt2(n_layer = 8, n_embd = 384, n_head = 6, vocab_size = vocab_size, n_positions = n_positions)
