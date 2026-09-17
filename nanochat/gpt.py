"""
Nanochat-style GPT backbone
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- unified SDPA / Flash Attention 3 attention path
"""

from functools import partial
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.moe_ve_gpt import MoEValueEmbeddings, ValueEmbeddings

# Our custom attention module defaults to SDPA and can opt into FA3 on Hopper.
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    head_dim: int = 0  # 0 = derive as n_embd // n_head; Qwen3 uses explicit head_dim
    weight_tying: bool = False
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    value_embeds_layers: List[int] = field(default_factory=list)  # 0-based layer indices (empty = default alternating)
    dense_ve_enabled: bool = False
    # MoE value embeddings (mixture-of-experts value residuals)
    moe_ve_enabled: bool = False
    moe_ve_per_head_table: bool = False
    moe_ve_network_shared_table: bool = False
    moe_ve_setting: str = "1_0_0"  # shared_activated_experts, e.g. 1_2_4
    moe_ve_gate_nl: str = "sigmoid"
    moe_ve_gate_type: str = "linear"
    moe_ve_conv_kernel_size: int = 4
    moe_ve_router_input: str = "value"
    moe_ve_balance_lr: float = 0.001
    moe_balance_mode: str = "bias"  # bias (loss-free router bias)
    moe_ve_bias_scope: str = "slot"
    moe_ve_bias_update: str = "deepseek_moe"
    moe_ve_bias_min_visits: int = 0  # 0 = use num_experts
    moe_ve_bias_powerlaw_n: float = 1.4
    moe_ve_maxvio_window: int = 0
    moe_ve_slot_mapping: str = "none"
    moe_ve_slot_index_mode: str = "token"
    moe_ve_bigram_slot_factor: int = 1
    moe_ve_slot_vocab_size: int = 0
    moe_ve_slot_dedicated_size: int = 0
    moe_ve_slot_map_path: str = ""
    bigram_engram_enabled: bool = False
    bigram_engram_layers: List[int] = field(default_factory=list)
    bigram_engram_vocab_factor: int = 6
    bigram_engram_gate_channels: int = 32
    bigram_engram_init_lambda: float = 0.1
    stem_layers: Optional[List[int]] = None
    stem_embedding_dim: Optional[int] = None
    stem_multiple_of: int = 256
    stem_ffn_dim_multiplier: Optional[float] = None


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def _parse_moe_setting(setting: str) -> tuple[bool, int, int]:
    if not setting:
        return True, 0, 0
    parts = setting.strip().split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid MoE setting '{setting}'. Expected format like '1_2_4'.")
    shared_flag = int(parts[0])
    activated = int(parts[1])
    experts = int(parts[2])
    if shared_flag not in (0, 1):
        raise ValueError(f"Invalid MoE shared flag '{shared_flag}', expected 0 or 1.")
    return bool(shared_flag), activated, experts


def _build_fa2_attn_mask(Tq, Tk, prefix_len, window, device):
    """Build a causal (and optional sliding window) mask for SDPA."""
    idx_q = torch.arange(Tq, device=device).unsqueeze(1)
    idx_k = torch.arange(Tk, device=device).unsqueeze(0)
    pos_q = idx_q + prefix_len
    mask = idx_k <= pos_q
    if window >= 0:
        mask &= idx_k >= (pos_q - window + 1)
    return mask


_BIGRAM_HASH_MULTIPLIER_1 = 36313
_BIGRAM_HASH_MULTIPLIER_2 = 27191


def resolve_bigram_engram_layers(n_layer: int, layer_ids: list[int]) -> list[int]:
    if not layer_ids:
        parity = (n_layer - 1) % 2
        return [layer_idx for layer_idx in range(n_layer) if layer_idx % 2 == parity]
    resolved = sorted(set(int(layer_id) for layer_id in layer_ids))
    for layer_id in resolved:
        if layer_id < 0 or layer_id >= n_layer:
            raise ValueError(f"Invalid bigram Engram layer id {layer_id} for n_layer={n_layer}")
    return resolved


def compute_bigram_hash(
    idx: torch.Tensor,
    bigram_vocab_size: int,
    prev_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    if idx.ndim != 2:
        raise ValueError("compute_bigram_hash expects idx with shape [batch, sequence]")
    if bigram_vocab_size < 2:
        raise ValueError("bigram_vocab_size must be >= 2")

    batch, sequence = idx.shape
    hashes = torch.empty_like(idx, dtype=torch.long)
    if sequence == 0:
        return hashes

    mod = bigram_vocab_size - 1
    token_ids = idx.to(dtype=torch.long)

    if prev_tokens is None:
        hashes[:, 0] = mod
    else:
        prev_tokens = prev_tokens.to(device=idx.device, dtype=torch.long).view(-1)
        if prev_tokens.numel() == 1 and batch != 1:
            prev_tokens = prev_tokens.expand(batch)
        if prev_tokens.numel() != batch:
            raise ValueError(
                f"prev_tokens batch mismatch: got {prev_tokens.numel()} for batch size {batch}"
            )
        hashes[:, 0] = torch.bitwise_xor(
            _BIGRAM_HASH_MULTIPLIER_1 * token_ids[:, 0],
            _BIGRAM_HASH_MULTIPLIER_2 * prev_tokens,
        ) % mod

    if sequence > 1:
        hashes[:, 1:] = torch.bitwise_xor(
            _BIGRAM_HASH_MULTIPLIER_1 * token_ids[:, 1:],
            _BIGRAM_HASH_MULTIPLIER_2 * token_ids[:, :-1],
        ) % mod
    return hashes


class BigramHashEmbedding(nn.Module):
    def __init__(self, vocab_size: int, vocab_factor: int, n_embd: int):
        super().__init__()
        if vocab_factor < 1:
            raise ValueError("bigram_engram_vocab_factor must be >= 1")
        self.bigram_vocab_size = vocab_size * vocab_factor
        self.embedding = nn.Embedding(self.bigram_vocab_size, n_embd)
        self.embedding.weight.engram_is_embedding = True

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.zeros_(self.embedding.weight)

    def forward(self, idx: torch.Tensor, prev_tokens: torch.Tensor | None = None) -> torch.Tensor:
        return self.embedding(compute_bigram_hash(idx, self.bigram_vocab_size, prev_tokens=prev_tokens))


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx, use_ve_gate: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        gate_out_dim = self.n_kv_head
        self.ve_gate = nn.Linear(self.ve_gate_channels, gate_out_dim, bias=False) if use_ve_gate else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache, v_pre=None):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D), which matches the FA3-compatible attention API.
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        if v_pre is None:
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        else:
            v = v_pre

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Unified attention backend (PyTorch SDPA by default, FA3 opt-in on Hopper)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x, input_ids=None):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

    def init_weights(self, s):
        torch.nn.init.uniform_(self.c_fc.weight, -s, s)
        torch.nn.init.zeros_(self.c_proj.weight)


class Block(nn.Module):
    def __init__(self, config, layer_idx, use_ve_gate: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(config, layer_idx, use_ve_gate=use_ve_gate)
        self.mlp = MLP(config)

    def forward(self, x, ve, input_ids, cos_sin, window_size, kv_cache, v_pre=None):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, v_pre=v_pre)
        x = x + self.mlp(norm(x), input_ids=input_ids)
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        self.use_dense_value_embedding = config.dense_ve_enabled
        self.use_moe_value_embedding = config.moe_ve_enabled
        self.use_moe_bigram_slot_index = self.use_moe_value_embedding and config.moe_ve_slot_index_mode == "bigram"
        self.use_bigram_engram = config.bigram_engram_enabled or bool(config.bigram_engram_layers)
        if self.use_dense_value_embedding and self.use_moe_value_embedding:
            raise ValueError("dense_ve_enabled and moe_ve_enabled cannot both be true")
        value_embed_layers = config.value_embeds_layers if (self.use_dense_value_embedding or self.use_moe_value_embedding) else []
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([
                Block(config, layer_idx, use_ve_gate=(layer_idx in value_embed_layers))
                for layer_idx in range(config.n_layer)
            ]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.embed_value_dense = None
        self.embed_value_moe = None
        self.bigram_engram_layers = []
        self.bigram_engram_layer_set = set()
        self.bigram_engram_gate_channels = 0
        self.bigram_engram_head_dim = head_dim
        self.bigram_engram_embedding = None
        self.bigram_engram_gates = nn.ModuleDict()
        self.bigram_lambdas = None
        self._bigram_prev_tokens = None
        if self.use_dense_value_embedding:
            self.embed_value_dense = ValueEmbeddings(
                vocab_size=padded_vocab_size,
                kv_dim=kv_dim,
                layer_ids=value_embed_layers,
            )
        if self.use_moe_value_embedding:
            moe_ve_shared, moe_ve_num_activated, moe_ve_num_experts = _parse_moe_setting(config.moe_ve_setting)
            self.embed_value_moe = MoEValueEmbeddings(
                vocab_size=padded_vocab_size,
                n_embd=config.n_embd,
                kv_dim=kv_dim,
                n_kv_head=config.n_kv_head,
                layer_ids=value_embed_layers,
                num_experts=moe_ve_num_experts,
                shared=moe_ve_shared,
                num_activated=moe_ve_num_activated,
                balance_lr=config.moe_ve_balance_lr,
                balance_mode=config.moe_balance_mode,
                bias_scope=config.moe_ve_bias_scope,
                bias_update=config.moe_ve_bias_update,
                bias_min_visits=config.moe_ve_bias_min_visits,
                bias_powerlaw_n=config.moe_ve_bias_powerlaw_n,
                maxvio_window=config.moe_ve_maxvio_window,
                gate_nl=config.moe_ve_gate_nl,
                gate_type=config.moe_ve_gate_type,
                gate_kernel_size=config.moe_ve_conv_kernel_size,
                router_input_mode=config.moe_ve_router_input,
                per_head_table=config.moe_ve_per_head_table,
                network_shared_table=config.moe_ve_network_shared_table,
                slot_mapping=config.moe_ve_slot_mapping,
                slot_index_mode=config.moe_ve_slot_index_mode,
                slot_vocab_size=config.moe_ve_slot_vocab_size,
                slot_dedicated_size=config.moe_ve_slot_dedicated_size,
                slot_map_path=config.moe_ve_slot_map_path,
            )
        if self.use_bigram_engram:
            if config.bigram_engram_gate_channels <= 0:
                raise ValueError("bigram_engram_gate_channels must be > 0")
            if config.bigram_engram_gate_channels > config.n_embd:
                raise ValueError("bigram_engram_gate_channels cannot exceed n_embd")
            resolved_layers = resolve_bigram_engram_layers(config.n_layer, config.bigram_engram_layers)
            config.bigram_engram_layers = resolved_layers
            self.bigram_engram_layers = resolved_layers
            self.bigram_engram_layer_set = set(resolved_layers)
            self.bigram_engram_gate_channels = config.bigram_engram_gate_channels
            self.bigram_engram_embedding = BigramHashEmbedding(
                vocab_size=config.vocab_size,
                vocab_factor=config.bigram_engram_vocab_factor,
                n_embd=config.n_embd,
            )
            self.bigram_engram_gates = nn.ModuleDict({
                str(layer_idx): nn.Linear(config.bigram_engram_gate_channels, config.n_head, bias=False)
                for layer_idx in resolved_layers
            })
            self.bigram_lambdas = nn.Parameter(torch.empty(config.n_layer))
            self.bigram_lambdas.x0_like_is_scalar = True

        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # Cache rotary positions beyond the training context length.
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            if hasattr(block.mlp, "init_weights"):
                block.mlp.init_weights(s)
            else:
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        if self.use_dense_value_embedding:
            self.embed_value_dense.init_weights(s)
        if self.use_moe_value_embedding:
            self.embed_value_moe.init_weights(s)
        if self.use_bigram_engram:
            self.bigram_engram_embedding.init_weights()
            torch.nn.init.zeros_(self.bigram_lambdas)
            for layer_idx in self.bigram_engram_layers:
                self.bigram_lambdas[layer_idx] = self.config.bigram_engram_init_lambda
                torch.nn.init.zeros_(self.bigram_engram_gates[str(layer_idx)].weight)

        # Value embedding gates start neutral: sigmoid(0) * 2 -> 1.0
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            if self.use_dense_value_embedding:
                self.embed_value_dense.to(dtype=torch.bfloat16)
            if self.use_moe_value_embedding:
                self.embed_value_moe.to(dtype=torch.bfloat16)
            if self.use_bigram_engram:
                self.bigram_engram_embedding.to(dtype=torch.bfloat16)
                self.bigram_engram_embedding.embedding.weight.engram_is_embedding = True
            for block in self.transformer.h:
                if hasattr(block.mlp, "stem_embedding"):
                    block.mlp.stem_embedding.to(dtype=torch.bfloat16)
                    block.mlp.stem_embedding.weight.stem_is_embedding = True

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embedding_params = sum(
            p.numel() for p in self.parameters() if getattr(p, "value_embedding_is_table", False)
        )
        engram_embedding_params = sum(
            p.numel() for p in self.parameters() if getattr(p, "engram_is_embedding", False)
        )
        x0_like_scalar_params = sum(
            p.numel() for p in self.parameters() if getattr(p, "x0_like_is_scalar", False)
        )
        stem_embedding_params = sum(
            p.numel() for p in self.parameters() if getattr(p, "stem_is_embedding", False)
        )
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + value_embedding_params
            + engram_embedding_params
            + x0_like_scalar_params
            + stem_embedding_params
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return all of the parameters, same as Chinchilla paper.
        Kaplan et al. did not include embedding parameters and said that this led to cleaner scaling laws.
        But Kaplan et al. also had a bug in their results (as pointed out by Chinchilla).
        My own experiments in nanochat confirm the Chinchilla approach gives the much cleaner scaling law.
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper <- good).
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper <- bad)
        """
        nparams = sum(p.numel() for p in self.parameters())
        return nparams

    def setup_optimizers(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
        extra_adam_lr=None,
        value_embedding_lr=None,
        engram_embedding_lr=None,
        stem_embedding_lr=None,
        value_embedding_adam_betas=None,
        value_embedding_adam_eps=None,
        value_embedding_weight_decay=None,
        engram_embedding_adam_betas=None,
        engram_embedding_adam_eps=None,
        engram_embedding_weight_decay=None,
    ):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into groups
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        extra_x0_scalar_params = [p for p in self.parameters() if getattr(p, "x0_like_is_scalar", False)]
        x0_params = [self.x0_lambdas] + extra_x0_scalar_params
        value_embedding_params = [p for p in self.parameters() if getattr(p, "value_embedding_is_table", False)]
        engram_embedding_params = [p for p in self.parameters() if getattr(p, "engram_is_embedding", False)]
        stem_embedding_params = [p for p in self.parameters() if getattr(p, "stem_is_embedding", False)]
        excluded = {
            id(p)
            for p in (
                embedding_params
                + lm_head_params
                + resid_params
                + x0_params
                + value_embedding_params
                + engram_embedding_params
                + stem_embedding_params
            )
        }
        muon_params = []
        adam_extra_params = []
        for name, p in self.named_parameters():
            if id(p) in excluded:
                continue
            if p.ndim == 2:
                muon_params.append(p)
            else:
                adam_extra_params.append(p)
        group_ids = {
            id(p)
            for p in (
                embedding_params
                + lm_head_params
                + resid_params
                + x0_params
                + value_embedding_params
                + engram_embedding_params
                + stem_embedding_params
                + muon_params
                + adam_extra_params
            )
        }
        assert len(list(self.parameters())) == len(group_ids)
        # Create the AdamW optimizer for the embedding, lm_head, and per-layer scalars
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        if value_embedding_lr is None:
            value_embedding_lr = embedding_lr
        if engram_embedding_lr is None:
            engram_embedding_lr = embedding_lr
        if stem_embedding_lr is None:
            stem_embedding_lr = embedding_lr
        if extra_adam_lr is None:
            extra_adam_lr = matrix_lr
        # Per-group VE Adam overrides (None = inherit shared adamw defaults).
        ve_betas = value_embedding_adam_betas if value_embedding_adam_betas is not None else adam_betas
        ve_eps = value_embedding_adam_eps if value_embedding_adam_eps is not None else 1e-10
        ve_wd = value_embedding_weight_decay if value_embedding_weight_decay is not None else 0.0
        ve_group = dict(
            params=value_embedding_params,
            lr=value_embedding_lr * dmodel_lr_scale,
            betas=ve_betas,
            eps=ve_eps,
            weight_decay=ve_wd,
        )
        engram_betas = engram_embedding_adam_betas if engram_embedding_adam_betas is not None else adam_betas
        engram_eps = engram_embedding_adam_eps if engram_embedding_adam_eps is not None else 1e-10
        engram_wd = engram_embedding_weight_decay if engram_embedding_weight_decay is not None else 0.0
        engram_group = dict(
            params=engram_embedding_params,
            lr=engram_embedding_lr * dmodel_lr_scale,
            betas=engram_betas,
            eps=engram_eps,
            weight_decay=engram_wd,
        )
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            ve_group,  # value-embedding table; per-group betas/eps/wd allowed
            engram_group if engram_embedding_params else None,
            dict(params=stem_embedding_params, lr=stem_embedding_lr * dmodel_lr_scale) if stem_embedding_params else None,
            dict(params=resid_params, lr=scalar_lr * 0.01), # these are a lot more sensitive because they accumulate in the residual stream
            dict(params=x0_params, lr=scalar_lr, betas=(0.96, 0.95)),
            dict(params=adam_extra_params, lr=extra_adam_lr * dmodel_lr_scale) if adam_extra_params else None,
        ]
        adam_groups = [g for g in adam_groups if g is not None]
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0) # NOTE: weight decay is hardcoded to 0.0 for AdamW, only used in Muon
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(muon_params, **muon_kwargs)
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def reset_moe_ve_eval_maxvio(self):
        if self.embed_value_moe is not None and hasattr(self.embed_value_moe, "reset_eval_maxvio"):
            self.embed_value_moe.reset_eval_maxvio()

    def finish_moe_ve_eval_maxvio(self):
        if self.embed_value_moe is not None and hasattr(self.embed_value_moe, "finish_eval_maxvio"):
            return self.embed_value_moe.finish_eval_maxvio()
        return None

    def _resolve_bigram_prev_tokens(self, idx: torch.Tensor, kv_cache) -> torch.Tensor | None:
        if kv_cache is None or kv_cache.get_pos() == 0 or self._bigram_prev_tokens is None:
            return None
        prev_tokens = self._bigram_prev_tokens
        if prev_tokens.device != idx.device:
            prev_tokens = prev_tokens.to(idx.device)
        prev_tokens = prev_tokens.view(-1)
        if prev_tokens.numel() == 1 and idx.size(0) != 1:
            prev_tokens = prev_tokens.expand(idx.size(0))
        if prev_tokens.numel() != idx.size(0):
            raise RuntimeError(
                f"Cached previous tokens batch mismatch: {prev_tokens.numel()} vs {idx.size(0)}"
            )
        return prev_tokens

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', return_aux_loss: bool = False):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        x0_bigram_heads = None
        if self.use_bigram_engram:
            prev_tokens = self._resolve_bigram_prev_tokens(idx, kv_cache)
            x0_bigram = self.bigram_engram_embedding(idx, prev_tokens=prev_tokens)
            x0_bigram_heads = x0_bigram.view(B, T, self.config.n_head, self.bigram_engram_head_dim)
        aux_loss = x.new_zeros(())
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if i in self.bigram_engram_layer_set:
                x_norm = norm(x)
                gate = 2 * torch.sigmoid(
                    self.bigram_engram_gates[str(i)](x_norm[..., :self.bigram_engram_gate_channels])
                )
                bigram_term = (gate.unsqueeze(-1) * x0_bigram_heads).view(B, T, -1)
                x = x + self.bigram_lambdas[i].to(dtype=bigram_term.dtype) * bigram_term
            ve_total = None
            v_pre = None
            if self.use_dense_value_embedding:
                ve_dense, ve_aux_loss = self.embed_value_dense(i, idx)
                aux_loss = aux_loss + ve_aux_loss
                if ve_dense is not None:
                    ve_total = ve_dense
            if self.use_moe_value_embedding:
                x_norm = norm(x)
                v_pre = block.attn.c_v(x_norm).view(B, T, block.attn.n_kv_head, block.attn.head_dim)
                moe_router_input = v_pre if self.config.moe_ve_router_input == "value" else x_norm
                moe_prev_tokens = self._resolve_bigram_prev_tokens(idx, kv_cache) if self.use_moe_bigram_slot_index else None
                ve_moe, ve_aux_loss = self.embed_value_moe(
                    i,
                    idx,
                    x,
                    router_input=moe_router_input,
                    prev_tokens=moe_prev_tokens,
                )
                aux_loss = aux_loss + ve_aux_loss
                if ve_moe is not None:
                    ve_total = ve_moe if ve_total is None else ve_total + ve_moe
            if v_pre is not None:
                x = block(x, ve_total, idx, cos_sin, self.window_sizes[i], kv_cache, v_pre=v_pre)
            else:
                x = block(x, ve_total, idx, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if (self.use_bigram_engram or self.use_moe_bigram_slot_index) and kv_cache is not None and T > 0:
            self._bigram_prev_tokens = idx[:, -1].detach().clone()

        if targets is not None:
            # training: given the targets, compute and return the loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            if return_aux_loss:
                return loss, aux_loss
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
