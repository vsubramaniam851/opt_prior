import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias = True, init = True, amp = False, dropout = 0.1):
        super(LowRankLinear, self).__init__()
        # NOTE: Both of these layers should be tuned by CKA
        self.rank = rank
        if amp and torch.cuda.is_available(): ##NOTE: I will add this as an option but would not recommend using this.
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        self.V = nn.Linear(in_features, rank, bias = False, dtype = dtype)
        self.U = nn.Linear(rank, out_features, bias = bias, dtype = dtype)
        self.dropout = nn.Dropout(dropout)
        if init:
            nn.init.xavier_uniform_(self.V.weight)
            nn.init.xavier_uniform_(self.U.weight)
        if bias:
            nn.init.zeros_(self.U.bias)

    @torch.no_grad()
    def conv_init_weights(self, module, H_out, W_out):
        kernel_mat = module.weight.flatten(1)
        U_k, S_k, V_k = torch.linalg.svd(kernel_mat, full_matrices=False)

        keep = min(self.U.weight.shape[1], S_k.numel())
        sqrt_S = torch.sqrt(S_k[:keep])
        U_base = U_k[:, :keep] * sqrt_S.unsqueeze(0)
        U_tiled = U_base.repeat_interleave(H_out * W_out, dim=0)
        self.U.weight.zero_()
        self.U.weight[:, :keep].copy_(U_tiled) 
        V_base = torch.diag(sqrt_S) @ V_k[:keep]
        self.V.weight.zero_()
        self.V.weight[:keep, :V_base.shape[1]].copy_(V_base)
        self.U.weight.mul_(0.15)
        self.V.weight.mul_(0.15)
        rank = self.U.weight.shape[1]
        pad = rank - keep
        if pad > 0:
            nn.init.orthogonal_(self.U.weight[:, keep:])
            nn.init.orthogonal_(self.V.weight[keep:, :]) 

    def forward(self, x):
        z = self.V(x)
        # z = self.dropout(z)
        return self.U(z)

class LinearizedConv2d(nn.Module):
    '''
    Linear Replacement for a 2D Convolutional Layer. Takes an input size and specific convolutional and designs the 
    linear layer with the correct input and correct output shapes. We do have to run a quick forward pass through the conv
    layer to get the correct sizes in this implementation.
    '''
    def __init__(self, conv, input_size, device, low_rank = True, rank = 1024, svd_init = False, amp = False):
        super(LinearizedConv2d, self).__init__()
        assert isinstance(conv, nn.Conv2d)
        C_in = conv.in_channels
        _, H_in, W_in = input_size
        with torch.no_grad():
            dummy = torch.zeros(1, C_in, H_in, W_in).to(device)
            out = conv(dummy)
        _, C_out, H_out, W_out = out.shape
        self.out_channels = C_out
        self.H_out = H_out
        self.W_out = W_out
        flat_dim = C_in * H_in * W_in
        out_dim = C_out * H_out * W_out
        if low_rank:
            self.linear = LowRankLinear(flat_dim, out_dim, rank = rank, init = (not svd_init), amp = amp)
            if svd_init:
                self.linear.conv_init_weights(conv, H_out, W_out)
        else:
            self.linear = nn.Linear(flat_dim, out_dim)
    
    def forward(self, x):
        B = x.size(0)
        x_flat = x.reshape(B, -1)
        y = self.linear(x_flat)
        return y.reshape(B, self.out_channels, self.H_out, self.W_out)

class LinearizedVIT(nn.Module):
    def __init__(self, linear, input_shape, device, low_rank = True, rank = 1024, amp = False):
        super(LinearizedVIT, self).__init__()
        in_features = linear.in_features
        assert in_features == input_shape[-1], 'Final dimension should match'
        self.out_features = linear.out_features
        self.num_patches = input_shape[0]
        flat_dim = in_features * self.num_patches
        out_dim = self.out_features * self.num_patches
        if low_rank:
            self.linear = LowRankLinear(flat_dim, out_dim, rank = rank)
        else:
            self.linear = nn.Linear(flat_dim, out_dim)
        
    def forward(self, x):
        B = x.size(0)
        x_flat = x.reshape(B, -1)
        y = self.linear(x_flat)
        return y.reshape(B, self.num_patches, self.out_features)
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, proj_bias = True, attn_drop = 0.0, proj_drop = 0.0, amp = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(self, init_attn_std = None, init_proj_std = None, factor = 1.0):
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x, is_causal = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x

class MemEffAttention(Attention):
    def forward(self, x):
        return super().forward(x)
    
class MLP(nn.Module):
    def __init__(self, att, input_shape, device, low_rank = False, rank = None, amp = False):
        super(MLP, self).__init__()
        output_dim = att.dim
        input_dim = input_shape[-1]
        
        self.linear1 = nn.Linear(input_dim, input_dim * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(input_dim * 4, output_dim)
        self.out_linear = nn.Linear(output_dim, output_dim)
        self.gamma = nn.Parameter(torch.ones(1) * 1e-3)

    def forward(self, x):
        B = x.size(0)
        out1 = self.gelu(self.linear1(x))
        out2 = self.gelu(self.linear2(out1))
        return self.gamma * self.out_linear(out2)
    
class BasicBlockCompat(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, stride = 1, norm_layer=None, amp=False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.downsample = None
        if stride != 1 or inplanes != outplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                norm_layer(outplanes),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_hidden, activation = 'silu', dropout = 0.0):
        super(ExpertFFN, self).__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU() if activation == 'silu' else nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(self.act(x))
        return self.fc2(x)

class SwitchMoETokenMixer(nn.Module):
    def __init__(self, d_model, d_hidden, n_experts = 16, capacity_factor = 1.25, router_jitter = 0.0, drop_tokens = False, expert_dropout = 0.0,
                activation = 'silu'):
        super(SwitchMoETokenMixer, self).__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.router_jitter = router_jitter
        self.drop_tokens = drop_tokens
        self.expert_dropout = expert_dropout
        self.activation = activation

        self.router = nn.Linear(d_model, n_experts, bias = False)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_hidden, activation, expert_dropout) for _ in range(n_experts)])

    def _capacity(self, T):
        capacity = self.capacity_factor * T / self.n_experts
        return math.ceil(capacity)

    def forward(self, x, training = False):
        B, L, D = x.shape
        T = B * L
        flat = x.view(T, D)
        
        logits = self.router(flat.to(torch.float32))
        if training and self.cfg.router_jitter > 0.0:
            logits = logits + (torch.rand_like(logits) - 0.5) * 2 * self.cfg.router_jitter
        probs = F.softmax(logits, dim = -1)
        probs = probs.to(flat.dtype)
        scores, expert_idx = probs.max(dim = -1)

        out = torch.zeros_like(flat)
        residual_mask = torch.ones(T, dtype = torch.bool, device = x.device)
        cap = self._capacity(T)

        for e in range(self.n_experts):
            sel = (expert_idx == e).nonzero(as_tuple = False).squeeze(-1)
            if sel.numel() == 0:
                continue
            if sel.numel() > cap:
                active = sel[:cap]
                overflow = sel[cap:]
                if not self.drop_tokens:
                    residual_mask[overflow] = True
                sel = active
                
            residual_mask[sel] = False
            x_e = flat.index_select(0, sel)
            y_e = self.experts[e](x_e)
            y_e = y_e.to(flat.dtype)
            g = scores.index_select(0, sel).unsqueeze(-1)
            out.index_add_(0, sel, g * y_e)
            out.index_copy_(0, sel, scores[sel, None] * y_e)

        out[residual_mask].add_(flat[residual_mask])
        return out.view(B, L, D) 

class SwitchMoE(nn.Module):
    def __init__(self, hidden_size, layer_idx, n_experts = 16, expert_mult = 4.0, capacity_factor = 1.25, router_jitter = 0.0, drop_tokens = False, expert_dropout = 0.0,
                activation = 'silu', amp = False):
        super().__init__()
        self.layer_idx = layer_idx
        d_model = hidden_size
        d_hidden = int(expert_mult * d_model)
        self.mixer = SwitchMoETokenMixer(d_model = d_model, d_hidden = d_hidden, capacity_factor = capacity_factor, router_jitter = router_jitter, 
                                                    drop_tokens = drop_tokens, expert_dropout = expert_dropout, activation = activation)

    @classmethod
    def from_attn(cls, attn_like, layer_idx, **overrides):
        mod = cls(attn_like.config.hidden_size, layer_idx, **overrides)
        ref = attn_like.o_proj.weight
        mod = mod.to(dtype = ref.dtype)
        return mod

    def forward(self, hidden_states, position_embeddings, attention_mask = None, 
                past_key_values = None, cache_position = None, **kwargs):
        return self.mixer(hidden_states), None

def repeat_kv(hidden_states, n_rep):
    b, h_kv, l, d = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return (hidden_states[:, :, None, :, :]
            .expand(b, h_kv, n_rep, l, d)
            .reshape(b, h_kv * n_rep, l, d))

def _phi_elu(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x, alpha=1.0) + 1.0

class LinearAttention(nn.Module):
    def __init__(self, config, feature_map = _phi_elu, eps = 1e-6, hidden_dim = 768, layer_idx = 1, num_attention_heads = 16, amp = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        assert hidden_dim % num_attention_heads == 0, 'Hidden dim should be divisible by number of attention heads!'
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.eps = eps
        self.feature_map = feature_map

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias = True)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias = True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias = False)

    @classmethod
    def from_attn(cls, attn_like, **overrides):
        mod = cls(attn_like.config, layer_idx = getattr(attn_like, 'layer_idx', 0), **overrides)
        ref = attn_like.o_proj.weight
        mod = mod.t(dtype = ref.dtype)
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            src = getattr(attn_like, name, None)
            dst = getattr(mod, name, None)
            if src is not None and dst is not None and src.weight.shape == dst.weight.shape:
                dst.weight.data.copy_(src.weight.data)
                if getattr(src, 'bias', None) is not None and getattr(dst, 'bias', None) is not None:
                    dst.bias.data.copy_(src.bias.data)
        return mod

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values = None, cache_position = None, **kwargs):
        B, L, Dm = hidden_states.shape
        H, Dh = self.n_heads, self.head_dim

        # Q: [B, H, L, Dh]; K/V: [B, H_kv, L, Dh]
        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.n_kv_heads, Dh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.n_kv_heads, Dh).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.n_kv_groups)
        v = repeat_kv(v, self.n_kv_groups)
        q = self.feature_map(q)
        k = self.feature_map(k)

        y = torch.empty(B, H, L, Dh, device=hidden_states.device, dtype=hidden_states.dtype)
        S = torch.zeros(B, H, Dh, Dh, device=hidden_states.device, dtype=hidden_states.dtype)
        z = torch.zeros(B, H, Dh, device=hidden_states.device, dtype=hidden_states.dtype)
        for t in range(L):
            kt = k[:, :, t, :]
            vt = v[:, :, t, :]
            S = S + torch.einsum('bhd,bhe->bhde', kt, vt)
            z = z + kt
            qt = q[:, :, t, :]
            num_t = torch.einsum('bhd,bhde->bhe', qt, S)
            den_t = (qt * z).sum(dim=-1, keepdim=True) + self.eps
            y[:, :, t, :] = num_t / den_t
        y = y.transpose(1, 2).contiguous().view(B, L, H * Dh)
        out = self.o_proj(y)
        return out, None

class RNNAtt(nn.Module):
    def __init__(self, attn, input_shape, device, num_layers = 2, rnn_dropout = 0.1, nonlinearity = 'tanh', low_rank = False, rank = None, amp = False):
        super(RNNAtt, self).__init__()
        hidden_dim = attn.config.hidden_size
        self.rnn = nn.ModuleList([])
        for _ in range(num_layers):
            self.rnn.append(nn.RNN(hidden_dim, hidden_dim, num_layers = 1, batch_first = True, nonlinearity = nonlinearity, bias = True))
        self._stable_init()
        self.dropout = nn.Dropout(rnn_dropout)

        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def _stable_init(self, rho=0.9, ih_gain=0.05):
        for rnn in self.rnn:
            nn.init.orthogonal_(rnn.weight_hh_l0, gain=rho)
            nn.init.xavier_uniform_(rnn.weight_ih_l0, gain=ih_gain) 
            nn.init.zeros_(rnn.bias_hh_l0) 
            nn.init.zeros_(rnn.bias_ih_l0) 

    def forward(self, hidden_states, position_embeddings = None, attention_mask = None, 
                past_key_values = None, cache_position = None, **kwargs):
        hidden = self.init_hidden(hidden_states.shape[0])
        for rnn_layer in self.rnn:
            h_in = hidden.unsqueeze(0) if hidden.dim() == 2 else hidden
            hidden_states, hidden = rnn_layer(hidden_states, h_in)
            hidden_states = self.dropout(hidden_states)
        return hidden_states, None
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim).to(self.device)

class ImageNetRNN(nn.Module):
    def __init__(self, hidden = 64, num_layers = 8, num_classes = 1000):
        super().__init__()
        self.patch = nn.Conv2d(3, hidden, kernel_size=4, stride=4, padding=0, bias=False)
        self.rnn = nn.ModuleList([nn.RNN(hidden, hidden, batch_first=True) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, num_classes)
    
    def _to_seq(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)
        return x, H, W

    def _to_img(self, x, H, W):
        B, N, C = x.shape
        return x.transpose(1, 2).view(B, C, H, W)

    def forward(self, x):
        x = self.patch(x)
        seq, H, W = self._to_seq(x)
        for r in self.rnn:
            out, _ = r(seq)
            seq = out
        seq = self.norm(seq)
        x = self._to_img(seq, H, W)
        feat = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.head(feat)

class ConvNextPatchMLP(nn.Module):
    def __init__(self, conv, input_size, device, low_rank=True, rank=1024, svd_init=False, amp=False,):
        super().__init__()
        assert isinstance(conv, nn.Conv2d)

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels

        self.kernel_size = conv.kernel_size
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)

        self.stride = conv.stride
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

        self.padding = conv.padding
        if isinstance(self.padding, int):
            self.padding = (self.padding, self.padding)

        self.dilation = conv.dilation
        if isinstance(self.dilation, int):
            self.dilation = (self.dilation, self.dilation)

        self.groups = conv.groups
        C_in = self.in_channels
        _, H_in, W_in = input_size
        with torch.no_grad():
            dummy = torch.zeros(1, C_in, H_in, W_in, device=device)
            out = conv(dummy)
        _, C_out, H_out, W_out = out.shape

        self.H_out = H_out
        self.W_out = W_out

        kH, kW = self.kernel_size
        d_in_group = (C_in // self.groups) * kH * kW
        d_out_group = C_out // self.groups

        hidden_dim = rank if rank is not None else d_in_group * 4

        self.mlp = nn.Sequential(
            nn.Linear(d_in_group, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_out_group),
        )

        self.gamma = nn.Parameter(torch.ones(1) * 1e-3)

    def forward(self, x):
        B, C, H, W = x.shape
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        B_, D, L = patches.shape

        G = self.groups
        kH, kW = self.kernel_size
        d_in_group = (self.in_channels // G) * kH * kW
        d_out_group = self.out_channels // G

        assert D == d_in_group * G, f'Unexpected patch dim: {D} != {d_in_group} * {G}'
        patches = patches.view(B_ * G, d_in_group, L).transpose(1, 2)
        out = self.mlp(patches)
        out = self.gamma * out
        out = out.transpose(1, 2).contiguous().view(B_, self.out_channels, L)
        out = out.view(B_, self.out_channels, self.H_out, self.W_out)
        return out