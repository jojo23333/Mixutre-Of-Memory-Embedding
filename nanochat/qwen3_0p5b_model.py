"""
Qwen3-0.5B-style dense backbone wired into nanochat training.

This follows the Qwen3 decoder shape while using nanochat's tokenizer/data and
experiment infrastructure. The active recipe intentionally uses rope_theta=1e4
for the 2k-context setting rather than Qwen3's long-context 1e6 default.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.adamw import DistAdamW
from nanochat.common import get_dist_info, print0
from nanochat.flash_attention import flash_attn
from nanochat.gpt import GPTConfig
from nanochat.moe_ve_gpt import MoEValueEmbeddings, ValueEmbeddings
from nanochat.muon import DistMuon, Muon


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


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        x_norm = x_float * torch.rsqrt((x_float * x_float).mean(-1, keepdim=True) + self.eps)
        return (x_norm * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3StemMLP(nn.Module):
    """STEM replacement for the up-projection branch of Qwen3's SwiGLU FFN."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, stem_y: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * stem_y)


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        norm_eps: float,
        layer_idx: int,
        use_ve_gate: bool = False,
    ):
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(head_dim, eps=norm_eps)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=norm_eps)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, num_kv_heads, bias=False) if use_ve_gate else None

    def project_value(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        kv_cache=None,
        ve: torch.Tensor | None = None,
        v_pre: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        query_states = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value_states = self.project_value(x) if v_pre is None else v_pre

        if ve is not None:
            if self.ve_gate is None:
                raise ValueError(f"Missing VE gate for layer {self.layer_idx}")
            ve = ve.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            value_states = value_states + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        query_states = apply_rotary_emb(self.q_norm(query_states), cos, sin)
        key_states = apply_rotary_emb(self.k_norm(key_states), cos, sin)

        if kv_cache is None:
            attn_output = flash_attn.flash_attn_func(
                query_states,
                key_states,
                value_states,
                causal=True,
                window_size=(-1, 0),
            )
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            attn_output = flash_attn.flash_attn_with_kvcache(
                query_states,
                k_cache,
                v_cache,
                k=key_states,
                v=value_states,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=(-1, 0),
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(seq_len)

        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        norm_eps: float,
        layer_idx: int,
        use_stem: bool = False,
        use_ve_gate: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_stem = use_stem
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            norm_eps=norm_eps,
            layer_idx=layer_idx,
            use_ve_gate=use_ve_gate,
        )
        mlp_cls = Qwen3StemMLP if use_stem else Qwen3MLP
        self.mlp = mlp_cls(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.input_layernorm = Qwen3RMSNorm(hidden_size, eps=norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        kv_cache=None,
        ve: torch.Tensor | None = None,
        stem_y: torch.Tensor | None = None,
        attn_input: torch.Tensor | None = None,
        v_pre: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn_input is None:
            attn_input = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(attn_input, cos_sin, kv_cache=kv_cache, ve=ve, v_pre=v_pre)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.use_stem:
            if stem_y is None:
                raise ValueError(f"Missing stem embeddings for STEM layer {self.layer_idx}")
            hidden_states = residual + self.mlp(hidden_states, stem_y)
        else:
            hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Qwen3_0p5B(nn.Module):
    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        self.use_dense_value_embedding = config.dense_ve_enabled
        self.use_moe_value_embedding = config.moe_ve_enabled
        if self.use_dense_value_embedding and self.use_moe_value_embedding:
            raise ValueError("dense_ve_enabled and moe_ve_enabled cannot both be true")

        self.norm_eps = 1e-6
        self.rope_theta = 10000.0
        self.initializer_range = 0.02
        self.rotary_seq_len = config.sequence_len * 10
        self.weight_tying = bool(config.weight_tying)
        self.head_dim = config.head_dim if config.head_dim > 0 else config.n_embd // config.n_head
        self.intermediate_size = 3 * config.n_embd
        value_embed_layers = config.value_embeds_layers if (self.use_dense_value_embedding or self.use_moe_value_embedding) else []
        resolved_stem_layers = sorted(set(config.stem_layers or []))

        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE")
        if config.n_head % config.n_kv_head != 0:
            raise ValueError(f"n_head={config.n_head} must be divisible by n_kv_head={config.n_kv_head}")
        invalid_stem_layers = [
            layer_idx
            for layer_idx in resolved_stem_layers
            if layer_idx < 0 or layer_idx >= config.n_layer
        ]
        if invalid_stem_layers:
            raise ValueError(
                f"Invalid STEM layer ids {invalid_stem_layers} for n_layer={config.n_layer}"
            )
        if resolved_stem_layers:
            if config.stem_embedding_dim is None:
                config.stem_embedding_dim = self.intermediate_size
            elif config.stem_embedding_dim != self.intermediate_size:
                raise ValueError(
                    f"stem_embedding_dim ({config.stem_embedding_dim}) must match "
                    f"Qwen3 intermediate_size ({self.intermediate_size})"
                )
        config.stem_layers = resolved_stem_layers
        self.stem_layers = resolved_stem_layers
        self._layer_to_stem_idx = {
            layer_idx: stem_idx
            for stem_idx, layer_idx in enumerate(resolved_stem_layers)
        }

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [
                        Qwen3DecoderLayer(
                            hidden_size=config.n_embd,
                            intermediate_size=self.intermediate_size,
                            num_heads=config.n_head,
                            num_kv_heads=config.n_kv_head,
                            head_dim=self.head_dim,
                            norm_eps=self.norm_eps,
                            layer_idx=layer_idx,
                            use_stem=(layer_idx in resolved_stem_layers),
                            use_ve_gate=(layer_idx in value_embed_layers),
                        )
                        for layer_idx in range(config.n_layer)
                    ]
                ),
                "ln_f": Qwen3RMSNorm(config.n_embd, eps=self.norm_eps),
            }
        )
        self.lm_head = None if self.weight_tying else nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        self.stem_embeddings = nn.ModuleList(
            [
                nn.Embedding(padded_vocab_size, self.intermediate_size)
                for _ in resolved_stem_layers
            ]
        )
        self.embed_value_dense = None
        self.embed_value_moe = None
        kv_dim = config.n_kv_head * self.head_dim
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

        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.rope_theta ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().bfloat16()[None, :, None, :]
        sin = freqs.sin().bfloat16()[None, :, None, :]
        return cos, sin

    @torch.no_grad()
    def init_weights(self):
        self.cos, self.sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.head_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            elif isinstance(module, Qwen3RMSNorm):
                module.reset_parameters()

        if self.use_dense_value_embedding:
            self.embed_value_dense.init_weights(self.initializer_range)
        if self.use_moe_value_embedding:
            self.embed_value_moe.init_weights(self.initializer_range)

        for embedding in self.stem_embeddings:
            embedding.weight.stem_is_embedding = True

        for layer in self.transformer.h:
            if layer.self_attn.ve_gate is not None:
                torch.nn.init.zeros_(layer.self_attn.ve_gate.weight)

        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for embedding in self.stem_embeddings:
                embedding.to(dtype=torch.bfloat16)
                embedding.weight.stem_is_embedding = True
            if self.use_dense_value_embedding:
                self.embed_value_dense.to(dtype=torch.bfloat16)
            if self.use_moe_value_embedding:
                self.embed_value_moe.to(dtype=torch.bfloat16)

    def get_device(self):
        return self.transformer.wte.weight.device

    def num_scaling_params(self):
        return sum(p.numel() for p in self.parameters())

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embedding_params = sum(p.numel() for p in self.parameters() if getattr(p, "value_embedding_is_table", False))
        stem_embedding_params = sum(p.numel() for p in self.parameters() if getattr(p, "stem_is_embedding", False))
        nparams_exclude = self.transformer.wte.weight.numel() + value_embedding_params + stem_embedding_params
        h = self.config.n_head
        t = self.config.sequence_len
        attn_flops = self.config.n_layer * 12 * h * self.head_dim * t
        return 6 * (nparams - nparams_exclude) + attn_flops

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
        del scalar_lr, engram_embedding_lr
        del engram_embedding_adam_betas, engram_embedding_adam_eps, engram_embedding_weight_decay
        model_dim = self.config.n_embd
        ddp, _, _, _ = get_dist_info()
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = [] if self.lm_head is None else list(self.lm_head.parameters())
        value_embedding_params = [p for p in self.parameters() if getattr(p, "value_embedding_is_table", False)]
        stem_embedding_params = [p for p in self.parameters() if getattr(p, "stem_is_embedding", False)]
        excluded = {
            id(p)
            for p in embedding_params + lm_head_params + value_embedding_params + stem_embedding_params
        }
        muon_params = []
        adam_extra_params = []
        for p in self.parameters():
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
                + value_embedding_params
                + stem_embedding_params
                + muon_params
                + adam_extra_params
            )
        }
        assert len(list(self.parameters())) == len(group_ids)

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        token_embedding_lr = unembedding_lr if self.weight_tying else embedding_lr
        if self.weight_tying:
            print0("Qwen3 tied token embeddings use unembedding_lr to avoid over-updating the output classifier.")
        if value_embedding_lr is None:
            value_embedding_lr = embedding_lr
        if stem_embedding_lr is None:
            stem_embedding_lr = embedding_lr
        if extra_adam_lr is None:
            extra_adam_lr = matrix_lr
        ve_betas = value_embedding_adam_betas if value_embedding_adam_betas is not None else adam_betas
        ve_eps = value_embedding_adam_eps if value_embedding_adam_eps is not None else 1e-10
        ve_wd = value_embedding_weight_decay if value_embedding_weight_decay is not None else 0.0

        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale) if lm_head_params else None,
            dict(params=embedding_params, lr=token_embedding_lr * dmodel_lr_scale),
            dict(
                params=value_embedding_params,
                lr=value_embedding_lr * dmodel_lr_scale,
                betas=ve_betas,
                eps=ve_eps,
                weight_decay=ve_wd,
            ) if value_embedding_params else None,
            dict(
                params=stem_embedding_params,
                lr=stem_embedding_lr * dmodel_lr_scale,
            ) if stem_embedding_params else None,
            dict(params=adam_extra_params, lr=extra_adam_lr * dmodel_lr_scale) if adam_extra_params else None,
        ]
        adam_groups = [group for group in adam_groups if group is not None]
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0)
        device = self.get_device()
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=(device.type == "cuda"))
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
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

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean", return_aux_loss: bool = False):
        batch_size, seq_len = idx.shape
        pos = 0 if kv_cache is None else kv_cache.get_pos()
        if pos + seq_len > self.cos.size(1):
            raise ValueError(f"Sequence position exceeds rotary cache: {pos + seq_len} > {self.cos.size(1)}")
        cos_sin = self.cos[:, pos : pos + seq_len], self.sin[:, pos : pos + seq_len]

        hidden_states = self.transformer.wte(idx)
        aux_loss = hidden_states.new_zeros(())
        for layer_idx, layer in enumerate(self.transformer.h):
            stem_y = None
            if layer_idx in self._layer_to_stem_idx:
                stem_embedding = self.stem_embeddings[self._layer_to_stem_idx[layer_idx]]
                stem_y = stem_embedding(idx)
            ve_total = None
            attn_input = None
            v_pre = None
            if self.use_dense_value_embedding:
                ve_dense, ve_aux_loss = self.embed_value_dense(layer_idx, idx)
                aux_loss = aux_loss + ve_aux_loss
                if ve_dense is not None:
                    ve_total = ve_dense
            if self.use_moe_value_embedding:
                attn_input = layer.input_layernorm(hidden_states)
                v_pre = layer.self_attn.project_value(attn_input)
                moe_router_input = v_pre if self.config.moe_ve_router_input == "value" else attn_input
                ve_moe, ve_aux_loss = self.embed_value_moe(
                    layer_idx,
                    idx,
                    hidden_states,
                    router_input=moe_router_input,
                )
                aux_loss = aux_loss + ve_aux_loss
                if ve_moe is not None:
                    ve_total = ve_moe if ve_total is None else ve_total + ve_moe
            hidden_states = layer(
                hidden_states,
                cos_sin,
                kv_cache=kv_cache,
                ve=ve_total,
                stem_y=stem_y,
                attn_input=attn_input,
                v_pre=v_pre,
            )

        hidden_states = self.transformer.ln_f(hidden_states)
        if self.lm_head is None:
            logits = F.linear(hidden_states, self.transformer.wte.weight)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits[..., : self.config.vocab_size].float()

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            if return_aux_loss:
                return loss, aux_loss
            return loss
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)[:, -1, :]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()
