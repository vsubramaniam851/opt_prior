import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_CLASSES = 1000
DROPOUT = 0.0
ATTN_DROPOUT = 0.0

class ViT(nn.Module):
    def __init__(self, *, image_size = 224, patch_size = 16, num_classes = 1000, hidden_dim = 768, num_layers = 12, num_heads = 12, 
                 mlp_dim = 3072, dropout = 0.0, attention_dropout = 0.0, representation_size = None):
        super().__init__()
        self.model = VisionTransformer(image_size = image_size, patch_size = patch_size, num_layers = num_layers, num_heads = num_heads, 
                                       hidden_dim = hidden_dim, mlp_dim = mlp_dim, dropout = dropout, attention_dropout = attention_dropout, 
                                       num_classes = num_classes, representation_size = representation_size, 
                                       norm_layer = nn.LayerNorm
        )

    def forward(self, x):
        return self.model(x)

def make_vit_tiny():
    return ViT(
        image_size = IMAGE_SIZE, patch_size = PATCH_SIZE, num_classes = NUM_CLASSES,
        hidden_dim = 192, num_layers = 4, num_heads = 3, mlp_dim = 192 * 4,
        dropout = DROPOUT, attention_dropout = ATTN_DROPOUT,
    )

def make_vit_small():
    return ViT(
        image_size = IMAGE_SIZE, patch_size = PATCH_SIZE, num_classes = NUM_CLASSES,
        hidden_dim = 384, num_layers = 8, num_heads = 6, mlp_dim = 384 * 4,
        dropout=DROPOUT, attention_dropout=ATTN_DROPOUT,
    )

def make_vit_base():
    return ViT(
        image_size = IMAGE_SIZE, patch_size = PATCH_SIZE, num_classes = NUM_CLASSES,
        hidden_dim = 768, num_layers = 12, num_heads = 12, mlp_dim = 768 * 4,
        dropout = DROPOUT, attention_dropout = ATTN_DROPOUT,
    )

def make_vit_large():
    return ViT(
        image_size = IMAGE_SIZE, patch_size = PATCH_SIZE, num_classes = NUM_CLASSES,
        hidden_dim = 1024, num_layers = 24, num_heads = 16, mlp_dim = 1024 * 4,
        dropout = DROPOUT, attention_dropout = ATTN_DROPOUT,
    )