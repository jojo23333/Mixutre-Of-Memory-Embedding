"""
Separate STEM 350M language model path wired to nanochat training.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.adamw import DistAdamW
from nanochat.common import get_dist_info, print0
from nanochat.gpt import BigramHashEmbedding, GPTConfig, resolve_bigram_engram_layers
from nanochat.moe_ve_gpt import MoEValueEmbeddings, ValueEmbeddings
from nanochat.muon import DistMuon, Muon
from nanochat.stemgpt_350m_transformer import (
    RMSNorm,
    RotaryEmbedding,
    StemGPT350MBlock,
    compute_stem_hidden_dim,
    resolve_stem_layers,
)


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


class StemGPT350M(nn.Module):
    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        self.use_dense_value_embedding = config.dense_ve_enabled
        self.use_moe_value_embedding = config.moe_ve_enabled
        self.use_bigram_engram = config.bigram_engram_enabled or bool(config.bigram_engram_layers)
        if sum([self.use_dense_value_embedding, self.use_moe_value_embedding, self.use_bigram_engram]) > 1:
            raise ValueError("dense_ve_enabled, moe_ve_enabled, and bigram_engram_enabled are mutually exclusive")
        self.norm_eps = 1e-5
        self.rope_theta = 10000.0
        self.init_base_std = None
        self.rotary_seq_len = config.sequence_len * 10
        self.weight_tying = bool(config.weight_tying)
        value_embed_layers = config.value_embeds_layers if (self.use_dense_value_embedding or self.use_moe_value_embedding) else []

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        if config.n_embd % config.n_head != 0:
            raise ValueError(f"n_embd={config.n_embd} must be divisible by n_head={config.n_head}")
        if config.n_head % config.n_kv_head != 0:
            raise ValueError(f"n_head={config.n_head} must be divisible by n_kv_head={config.n_kv_head}")

        resolved_layers = resolve_stem_layers(config.n_layer, config.stem_layers)
        hidden_dim = compute_stem_hidden_dim(
            dim=config.n_embd,
            multiple_of=config.stem_multiple_of,
            ffn_dim_multiplier=config.stem_ffn_dim_multiplier,
        )
        if config.stem_embedding_dim is None:
            config.stem_embedding_dim = hidden_dim
        elif config.stem_embedding_dim != hidden_dim:
            raise ValueError(
                f"stem_embedding_dim ({config.stem_embedding_dim}) must match STEM hidden dim ({hidden_dim})"
            )
        config.stem_layers = resolved_layers
        self.stem_layers = resolved_layers
        self._layer_to_stem_idx = {
            layer_idx: stem_idx
            for stem_idx, layer_idx in enumerate(resolved_layers)
        }

        head_dim = config.n_embd // config.n_head
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [
                        StemGPT350MBlock(
                            dim=config.n_embd,
                            head_dim=head_dim,
                            n_heads=config.n_head,
                            n_kv_heads=config.n_kv_head,
                            rope_theta=self.rope_theta,
                            multiple_of=config.stem_multiple_of,
                            ffn_dim_multiplier=config.stem_ffn_dim_multiplier,
                            norm_eps=self.norm_eps,
                            layer_idx=layer_idx,
                            use_stem=(layer_idx in resolved_layers),
                            use_ve_gate=(layer_idx in value_embed_layers),
                        )
                        for layer_idx in range(config.n_layer)
                    ]
                ),
                "ln_f": RMSNorm(config.n_embd, eps=self.norm_eps),
            }
        )
        self.lm_head = None if self.weight_tying else nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        self.stem_embeddings = nn.ModuleList(
            [
                nn.Embedding(padded_vocab_size, config.stem_embedding_dim)
                for _ in resolved_layers
            ]
        )
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
                kv_dim=config.n_kv_head * head_dim,
                layer_ids=value_embed_layers,
        )
        if self.use_moe_value_embedding:
            moe_ve_shared, moe_ve_num_activated, moe_ve_num_experts = _parse_moe_setting(config.moe_ve_setting)
            moe_ve_k = getattr(config, "moe_ve_k", -1)
            moe_ve_num_activated = moe_ve_k if moe_ve_k > 0 else moe_ve_num_activated
            self.embed_value_moe = MoEValueEmbeddings(
                vocab_size=padded_vocab_size,
                n_embd=config.n_embd,
                kv_dim=config.n_kv_head * head_dim,
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
            resolved_engram_layers = resolve_bigram_engram_layers(config.n_layer, config.bigram_engram_layers)
            config.bigram_engram_layers = resolved_engram_layers
            self.bigram_engram_layers = resolved_engram_layers
            self.bigram_engram_layer_set = set(resolved_engram_layers)
            self.bigram_engram_gate_channels = config.bigram_engram_gate_channels
            self.bigram_engram_embedding = BigramHashEmbedding(
                vocab_size=config.vocab_size,
                vocab_factor=config.bigram_engram_vocab_factor,
                n_embd=config.n_embd,
            )
            self.bigram_engram_gates = nn.ModuleDict({
                str(layer_idx): nn.Linear(config.bigram_engram_gate_channels, config.n_head, bias=False)
                for layer_idx in resolved_engram_layers
            })
            self.bigram_lambdas = nn.Parameter(torch.empty(config.n_layer))
            self.bigram_lambdas.x0_like_is_scalar = True
        for embedding in self.stem_embeddings:
            embedding.weight.stem_is_embedding = True

        self.rope_embeddings = RotaryEmbedding(
            theta=self.rope_theta,
            head_dim=head_dim,
            max_seqlen=self.rotary_seq_len,
        )

    @torch.no_grad()
    def init_weights(self):
        self.rope_embeddings.reset_parameters()
        init_std = self.init_base_std or (self.config.n_embd ** -0.5)
        torch.nn.init.trunc_normal_(
            self.transformer.wte.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if self.lm_head is not None:
            torch.nn.init.trunc_normal_(
                self.lm_head.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
        self.transformer.ln_f.reset_parameters()
        for embedding in self.stem_embeddings:
            torch.nn.init.trunc_normal_(
                embedding.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
            embedding.weight.stem_is_embedding = True

        for layer in self.transformer.h:
            layer.reset_parameters(self.init_base_std, 1.0)
        if self.use_dense_value_embedding:
            self.embed_value_dense.init_weights(init_std)
        if self.use_moe_value_embedding:
            self.embed_value_moe.init_weights(init_std)
        if self.use_bigram_engram:
            self.bigram_engram_embedding.init_weights()
            torch.nn.init.zeros_(self.bigram_lambdas)
            for layer_idx in self.bigram_engram_layers:
                self.bigram_lambdas[layer_idx] = self.config.bigram_engram_init_lambda
                torch.nn.init.zeros_(self.bigram_engram_gates[str(layer_idx)].weight)

        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for embedding in self.stem_embeddings:
                embedding.to(dtype=torch.bfloat16)
                embedding.weight.stem_is_embedding = True
            if self.use_dense_value_embedding:
                self.embed_value_dense.to(dtype=torch.bfloat16)
            if self.use_moe_value_embedding:
                self.embed_value_moe.to(dtype=torch.bfloat16)
            if self.use_bigram_engram:
                self.bigram_engram_embedding.to(dtype=torch.bfloat16)
                self.bigram_engram_embedding.embedding.weight.engram_is_embedding = True

    def get_device(self):
        return self.transformer.wte.weight.device

    def num_scaling_params(self):
        return sum(p.numel() for p in self.parameters())

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embedding_params = sum(p.numel() for p in self.parameters() if getattr(p, "value_embedding_is_table", False))
        engram_embedding_params = sum(p.numel() for p in self.parameters() if getattr(p, "engram_is_embedding", False))
        stem_embedding_params = sum(p.numel() for p in self.parameters() if getattr(p, "stem_is_embedding", False))
        nparams_exclude = self.transformer.wte.weight.numel() + value_embedding_params + engram_embedding_params + stem_embedding_params
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = self.config.n_layer * 12 * h * q * t
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
        del scalar_lr
        model_dim = self.config.n_embd
        ddp, _, _, _ = get_dist_info()
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = [] if self.lm_head is None else list(self.lm_head.parameters())
        value_embedding_params = [p for p in self.parameters() if getattr(p, "value_embedding_is_table", False)]
        engram_embedding_params = [p for p in self.parameters() if getattr(p, "engram_is_embedding", False)]
        stem_embedding_params = [p for p in self.parameters() if getattr(p, "stem_is_embedding", False)]
        excluded = {
            id(p)
            for p in (
                embedding_params
                + lm_head_params
                + value_embedding_params
                + engram_embedding_params
                + stem_embedding_params
            )
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
        ve_betas = value_embedding_adam_betas if value_embedding_adam_betas is not None else adam_betas
        ve_eps = value_embedding_adam_eps if value_embedding_adam_eps is not None else 1e-10
        ve_wd = value_embedding_weight_decay if value_embedding_weight_decay is not None else 0.0
        engram_betas = engram_embedding_adam_betas if engram_embedding_adam_betas is not None else adam_betas
        engram_eps = engram_embedding_adam_eps if engram_embedding_adam_eps is not None else 1e-10
        engram_wd = engram_embedding_weight_decay if engram_embedding_weight_decay is not None else 0.0
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(
                params=value_embedding_params,
                lr=value_embedding_lr * dmodel_lr_scale,
                betas=ve_betas,
                eps=ve_eps,
                weight_decay=ve_wd,
            ) if value_embedding_params else None,
            dict(
                params=engram_embedding_params,
                lr=engram_embedding_lr * dmodel_lr_scale,
                betas=engram_betas,
                eps=engram_eps,
                weight_decay=engram_wd,
            ) if engram_embedding_params else None,
            dict(params=stem_embedding_params, lr=stem_embedding_lr * dmodel_lr_scale) if stem_embedding_params else None,
            dict(params=adam_extra_params, lr=extra_adam_lr * dmodel_lr_scale) if adam_extra_params else None,
        ]
        adam_groups = [group for group in adam_groups if group is not None]
        group_ids = {
            id(p)
            for p in (
                embedding_params
                + lm_head_params
                + value_embedding_params
                + engram_embedding_params
                + stem_embedding_params
                + muon_params
                + adam_extra_params
            )
        }
        assert len(list(self.parameters())) == len(group_ids)
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

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean", return_aux_loss: bool = False):
        batch_size, seq_len = idx.shape
        pos = 0 if kv_cache is None else kv_cache.get_pos()
        if pos + seq_len > self.rope_embeddings.max_seqlen:
            raise ValueError(
                f"Sequence position exceeds rotary cache: {pos + seq_len} > {self.rope_embeddings.max_seqlen}"
            )
        tok_idx = torch.arange(pos, pos + seq_len, device=idx.device)
        freq_cis = self.rope_embeddings(tok_idx=tok_idx)

        hidden = self.transformer.wte(idx)
        bigram_engram_heads = None
        if self.use_bigram_engram:
            prev_tokens = self._resolve_bigram_prev_tokens(idx, kv_cache)
            bigram_engram = self.bigram_engram_embedding(idx, prev_tokens=prev_tokens)
            bigram_engram_heads = bigram_engram.view(
                batch_size,
                seq_len,
                self.config.n_head,
                self.bigram_engram_head_dim,
            )
        aux_loss = hidden.new_zeros(())
        for layer_idx, layer in enumerate(self.transformer.h):
            stem_y = None
            if layer_idx in self._layer_to_stem_idx:
                stem_embedding = self.stem_embeddings[self._layer_to_stem_idx[layer_idx]]
                stem_y = stem_embedding(idx)
            if layer_idx in self.bigram_engram_layer_set:
                hidden_norm = layer.attention_norm(hidden)
                gate = 2 * torch.sigmoid(
                    self.bigram_engram_gates[str(layer_idx)](
                        hidden_norm[..., : self.bigram_engram_gate_channels]
                    )
                )
                bigram_term = (gate.unsqueeze(-1) * bigram_engram_heads).view(batch_size, seq_len, -1)
                hidden = hidden + self.bigram_lambdas[layer_idx].to(dtype=bigram_term.dtype) * bigram_term
            ve_total = None
            attn_input = None
            v_pre = None
            if self.use_dense_value_embedding:
                ve_dense, ve_aux_loss = self.embed_value_dense(layer_idx, idx)
                aux_loss = aux_loss + ve_aux_loss
                if ve_dense is not None:
                    ve_total = ve_dense
            if self.use_moe_value_embedding:
                attn_input = layer.attention_norm(hidden)
                v_pre = layer.attention.project_value(attn_input)
                moe_router_input = v_pre if self.config.moe_ve_router_input == "value" else attn_input
                ve_moe, ve_aux_loss = self.embed_value_moe(
                    layer_idx,
                    idx,
                    hidden,
                    router_input=moe_router_input,
                )
                aux_loss = aux_loss + ve_aux_loss
                if ve_moe is not None:
                    ve_total = ve_moe if ve_total is None else ve_total + ve_moe
            hidden = layer(
                hidden,
                freq_cis,
                stem_y=stem_y,
                kv_cache=kv_cache,
                ve=ve_total,
                attn_input=attn_input,
                v_pre=v_pre,
            )

        hidden = self.transformer.ln_f(hidden)
        if self.lm_head is None:
            logits = F.linear(hidden, self.transformer.wte.weight)
        else:
            logits = self.lm_head(hidden)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()

        if self.use_bigram_engram and kv_cache is not None and seq_len > 0:
            self._bigram_prev_tokens = idx[:, -1].detach().clone()

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
