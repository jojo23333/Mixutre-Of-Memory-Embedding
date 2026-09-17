"""
Separate upstream-style transformer blocks for the STEM 350M family.

This module mirrors the STEM 350M backbone structure while staying isolated
from the existing nanochat GPT and `stem_faithful` paths.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_stem_hidden_dim(
    *,
    dim: int,
    multiple_of: int,
    ffn_dim_multiplier: Optional[float],
) -> int:
    hidden_dim = int(2 * (4 * dim) / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def resolve_stem_layers(n_layer: int, stem_layers: Optional[list[int]]) -> list[int]:
    if stem_layers is None:
        resolved = list(range(1, n_layer))
    else:
        resolved = sorted(set(int(layer_idx) for layer_idx in stem_layers))
    return [layer_idx for layer_idx in resolved if 0 <= layer_idx < n_layer]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


def _build_kv_cache_mask(query_len: int, key_len: int, prefix_len: int, device: torch.device) -> torch.Tensor:
    query_idx = torch.arange(query_len, device=device).unsqueeze(1)
    key_idx = torch.arange(key_len, device=device).unsqueeze(0)
    return key_idx <= (prefix_len + query_idx)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device: Optional[torch.device] = None):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32)[: (dim // 2)] / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    cos, sin = freqs.cos(), freqs.sin()
    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int) -> torch.Tensor:
    ndim = x.ndim
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1
        for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, seq_dim: int, freqs_cis: torch.Tensor):
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_, seq_dim).float()
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(nn.Module):
    def __init__(self, theta: float, head_dim: int, max_seqlen: int):
        super().__init__()
        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim=head_dim, end=max_seqlen, theta=theta),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(
            dim=self.head_dim,
            end=self.max_seqlen,
            theta=self.theta,
            device=self.freqs_cis.device,
        )

    def forward(self, *, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        if seqlen is None:
            raise ValueError("RotaryEmbedding.forward requires seqlen or tok_idx")
        return self.freqs_cis[:seqlen]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        x_norm = x_float * torch.rsqrt((x_float * x_float).mean(-1, keepdim=True) + self.eps)
        return (x_norm * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


class StemGPT350MAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        layer_idx: int,
        use_ve_gate: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, n_kv_heads, bias=False) if use_ve_gate else None

    def project_value(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        kv_cache=None,
        ve: Optional[torch.Tensor] = None,
        v_pre: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.project_value(x) if v_pre is None else v_pre
        if ve is not None:
            if self.ve_gate is None:
                raise ValueError(f"Missing VE gate for layer {self.layer_idx}")
            ve = ve.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            xv = xv + gate.unsqueeze(-1) * ve
        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            prefix_len = kv_cache.get_pos()
            k_cache[:, prefix_len : prefix_len + seq_len, :, :] = xk
            v_cache[:, prefix_len : prefix_len + seq_len, :, :] = xv
            xk = k_cache[:, : prefix_len + seq_len, :, :]
            xv = v_cache[:, : prefix_len + seq_len, :, :]
        else:
            prefix_len = 0

        xk = repeat_kv(xk, self.heads_per_group)
        xv = repeat_kv(xv, self.heads_per_group)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if kv_cache is None:
            output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        else:
            attn_mask = _build_kv_cache_mask(
                query_len=seq_len,
                key_len=prefix_len + seq_len,
                prefix_len=prefix_len,
                device=x.device,
            )
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(seq_len)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.head_dim)
        return self.wo(output)

    def reset_parameters(self, init_std=None, factor: float = 1.0):
        init_std = init_std or (self.dim ** -0.5)
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(
                linear.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if self.ve_gate is not None:
            nn.init.zeros_(self.ve_gate.weight)


class StemGPT350MFeedForward(nn.Module):
    def __init__(self, *, dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = compute_stem_hidden_dim(
            dim=dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def reset_parameters(self, init_std=None, factor: float = 1.0):
        in_init_std = init_std or (self.dim ** -0.5)
        out_init_std = (init_std or (self.hidden_dim ** -0.5)) / factor
        for linear in (self.w1, self.w3):
            nn.init.trunc_normal_(
                linear.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )


class StemGPT350MStemFeedForward(nn.Module):
    def __init__(self, *, dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = compute_stem_hidden_dim(
            dim=dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * y)

    def reset_parameters(self, init_std=None, factor: float = 1.0):
        in_init_std = init_std or (self.dim ** -0.5)
        out_init_std = (init_std or (self.hidden_dim ** -0.5)) / factor
        nn.init.trunc_normal_(
            self.w1.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )


class StemGPT350MBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        norm_eps: float,
        layer_idx: int,
        use_stem: bool,
        use_ve_gate: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_stem = use_stem
        self.attention = StemGPT350MAttention(
            dim=dim,
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=rope_theta,
            layer_idx=layer_idx,
            use_ve_gate=use_ve_gate,
        )
        if use_stem:
            self.feed_forward = StemGPT350MStemFeedForward(
                dim=dim,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
            )
        else:
            self.feed_forward = StemGPT350MFeedForward(
                dim=dim,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
            )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        stem_y: Optional[torch.Tensor] = None,
        kv_cache=None,
        ve: Optional[torch.Tensor] = None,
        attn_input: Optional[torch.Tensor] = None,
        v_pre: Optional[torch.Tensor] = None,
    ):
        if attn_input is None:
            attn_input = self.attention_norm(x)
        h = x + self.attention(attn_input, freq_cis, kv_cache=kv_cache, ve=ve, v_pre=v_pre)
        if self.use_stem:
            if stem_y is None:
                raise ValueError(f"Missing stem embeddings for STEM layer {self.layer_idx}")
            out = h + self.feed_forward(self.ffn_norm(h), stem_y)
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def reset_parameters(self, init_std=None, factor: float = 1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()
