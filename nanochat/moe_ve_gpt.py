"""
MoE embeddings for nanochat.

- ValueEmbeddings: dense value embedding tables (no routing).
- MoEValueEmbeddings: routed/shared experts that return a value-embedding tensor.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from nanochat.common import get_dist_info
from nanochat.slot_map_io import load_slot_map_artifact


_BIGRAM_HASH_MULTIPLIER_1 = 36313
_BIGRAM_HASH_MULTIPLIER_2 = 27191


def compute_moe_bigram_hash(
    idx: torch.Tensor,
    bigram_vocab_size: int,
    prev_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    if idx.ndim != 2:
        raise ValueError("compute_moe_bigram_hash expects idx with shape [batch, sequence]")
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


def _make_value_embedding(num_embeddings: int, embedding_dim: int, embedding_tag: str) -> nn.Embedding:
    embedding = nn.Embedding(num_embeddings, embedding_dim)
    setattr(embedding.weight, embedding_tag, True)
    return embedding


def _validate_value_embedding(
    embedding: nn.Embedding,
    *,
    num_embeddings: int,
    embedding_dim: int,
    name: str,
) -> None:
    if embedding.num_embeddings != num_embeddings or embedding.embedding_dim != embedding_dim:
        raise ValueError(
            f"{name} shape mismatch: expected ({num_embeddings}, {embedding_dim}), "
            f"got ({embedding.num_embeddings}, {embedding.embedding_dim})"
        )


def _init_embedding_once(embedding: nn.Embedding, s: float, initialized_embeddings: set[int] | None) -> None:
    if initialized_embeddings is not None:
        key = id(embedding.weight)
        if key in initialized_embeddings:
            return
        initialized_embeddings.add(key)
    torch.nn.init.uniform_(embedding.weight, -s, s)


def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k: int | None = 2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class MoEValueRouter(nn.Module):
    def __init__(
        self,
        n_embd: int,
        num_experts: int,
        slot_vocab_size: int,
        num_head: int = 1,
        gate_type: str = "linear",
        gate_kernel_size: int = 4,
        input_mode: str = "value",
        bias_per_head: bool = True,
    ):
        super().__init__()
        if gate_type not in {"linear", "conv1d"}:
            raise ValueError("gate_type must be one of: linear, conv1d")
        if gate_kernel_size < 1:
            raise ValueError("gate_kernel_size must be >= 1")
        self.gate_type = gate_type
        self.gate_kernel_size = gate_kernel_size
        self.num_head = num_head
        self.num_experts = num_experts
        self.input_mode = input_mode
        if input_mode not in {"value", "hidden"}:
            raise ValueError("input_mode must be one of: value, hidden")
        output_experts = num_experts if input_mode == "value" else num_head * num_experts
        if gate_type == "linear":
            self.weight = nn.Parameter(torch.empty((output_experts, n_embd)))
            self.bias = nn.Parameter(torch.zeros(output_experts))
        elif gate_type == "conv1d":
            self.gate = nn.Conv1d(n_embd, output_experts, kernel_size=gate_kernel_size, bias=True, padding=0)
        bias_shape = (
            (slot_vocab_size, num_head, num_experts)
            if bias_per_head and num_head > 1
            else (slot_vocab_size, num_experts)
        )
        self.register_buffer("router_bias", torch.zeros(bias_shape), persistent=False)

    def init_weights(self):
        if self.gate_type == "linear":
            torch.nn.init.zeros_(self.weight)
            torch.nn.init.zeros_(self.bias)
        elif self.gate_type == "conv1d":
            torch.nn.init.zeros_(self.gate.weight)
            torch.nn.init.zeros_(self.gate.bias)
        self.router_bias.zero_()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == 4:
            B, T, H, C = hidden_states.size()
            if self.input_mode != "value":
                raise ValueError("MoEValueRouter expected 3D hidden-state input when input_mode='hidden'")
            if H != self.num_head:
                raise ValueError(f"MoEValueRouter expected num_head={self.num_head}, got {H}")
            if self.gate_type == "conv1d":
                # (B, T, H, C) -> (B*H, C, T)
                x = hidden_states.permute(0, 2, 3, 1).reshape(B * H, C, T)
                x = F.pad(x, (self.gate_kernel_size - 1, 0))
                logits = self.gate(x)
                # (B*H, num_experts, T) -> (B, T, H, num_experts)
                logits = logits.permute(0, 2, 1).reshape(B, T, H, self.num_experts)
                return logits
            flat = hidden_states.reshape(B * T * H, C)
            logits = F.linear(flat.float(), self.weight.float(), self.bias.float())
            return logits.view(B, T, H, self.num_experts)
        if self.gate_type == "conv1d":
            B, T, C = hidden_states.size()
            # hidden_states: (B, T, C) -> (B, C, T)
            x = hidden_states.transpose(1, 2)
            x = F.pad(x, (self.gate_kernel_size - 1, 0))
            logits = self.gate(x)
            # logits: (B, num_experts, T) -> (B, T, num_experts)
            logits = logits.transpose(1, 2)
            if self.input_mode == "hidden" and self.num_head > 1:
                return logits.view(B, T, self.num_head, self.num_experts)
            return logits
        flat = hidden_states.view(-1, hidden_states.size(-1))
        logits = F.linear(flat.float(), self.weight.float(), self.bias.float())
        if self.input_mode == "hidden" and self.num_head > 1:
            return logits.view(*hidden_states.shape[:-1], self.num_head, self.num_experts)
        return logits.view(*hidden_states.shape[:-1], -1)


class _MoEEmbeddingBase(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        n_embd: int,
        router_dim: int | None = None,
        embed_dim: int,
        num_head: int = 1,
        num_experts: int,
        shared: bool,
        num_activated: int,
        balance_lr: float,
        balance_mode: str = "bias",
        bias_scope: str = "slot",
        bias_update: str = "deepseek_moe",
        bias_min_visits: int = 32,
        bias_powerlaw_n: float = 1.4,
        maxvio_window: int = 0,
        gate_nl: str,
        slot_mapping: str = "none",
        slot_vocab_size: int = 0,
        slot_dedicated_size: int = 0,
        slot_map_path: str = "",
        embedding_tag: str = "value_embedding_is_table",
        per_head_table: bool = False,
        gate_type: str = "linear",
        gate_kernel_size: int = 4,
        router_input_mode: str = "value",
        slot_index_mode: str = "token",
        shared_embedding_module: nn.Embedding | None = None,
        routed_embedding_module: nn.Embedding | None = None,
    ):
        super().__init__()
        if num_head <= 0:
            raise ValueError("num_head must be >= 1")
        if num_experts < 0 or num_activated < 0:
            raise ValueError("num_experts/num_activated must be >= 0")
        if num_experts == 0 and not shared:
            raise ValueError("At least one expert must be configured")
        if num_experts == 0 and num_activated > 0:
            raise ValueError("num_activated requires num_experts > 0")
        if num_activated > num_experts:
            raise ValueError("num_activated cannot exceed num_experts")
        if gate_nl not in {"softmax", "softmax-norm", "softmax-norm-detach", "sigmoid", "sigmoid-norm"}:
            raise ValueError(
                "gate_nl must be one of: softmax, softmax-norm, softmax-norm-detach, sigmoid, sigmoid-norm"
            )
        if balance_mode not in {"bias", "aux"}:
            raise ValueError("balance_mode must be one of: bias, aux")
        if bias_scope not in {"slot", "slot_shared_head"}:
            raise ValueError("bias_scope must be one of: slot, slot_shared_head")
        if bias_update not in {"none", "deepseek_moe", "trust_region"}:
            raise ValueError("bias_update must be one of: none, deepseek_moe, trust_region")
        if bias_update == "trust_region" and bias_scope not in {"slot", "slot_shared_head"}:
            raise ValueError("bias_update='trust_region' requires bias_scope=slot or slot_shared_head")
        if bias_min_visits < 0:
            raise ValueError("bias_min_visits must be >= 0")
        if bias_powerlaw_n < 1.0:
            raise ValueError("bias_powerlaw_n must be >= 1.0")
        if maxvio_window < 0:
            raise ValueError("maxvio_window must be >= 0")
        if slot_mapping not in {"none", "mod", "headmod", "table"}:
            raise ValueError("slot_mapping must be one of: none, mod, headmod, table")
        if slot_index_mode not in {"token", "bigram"}:
            raise ValueError("slot_index_mode must be one of: token, bigram")
        if slot_index_mode == "bigram" and slot_mapping != "none":
            raise ValueError("slot_index_mode='bigram' requires slot_mapping='none'")
        if slot_vocab_size <= 0:
            slot_vocab_size = vocab_size
        if slot_index_mode == "token" and slot_vocab_size > vocab_size:
            raise ValueError("slot_vocab_size cannot exceed vocab_size")
        if slot_mapping == "none" and slot_index_mode == "token" and slot_vocab_size != vocab_size:
            raise ValueError("slot_vocab_size must equal vocab_size when slot_mapping='none'")
        if slot_mapping == "headmod":
            if slot_dedicated_size <= 0:
                raise ValueError("slot_dedicated_size must be > 0 when slot_mapping='headmod'")
            if slot_dedicated_size >= slot_vocab_size:
                raise ValueError("slot_dedicated_size must be smaller than slot_vocab_size when slot_mapping='headmod'")
        else:
            slot_dedicated_size = 0
        if slot_mapping == "table" and not slot_map_path:
            raise ValueError("slot_map_path must be provided when slot_mapping='table'")
        if slot_mapping != "table":
            slot_map_path = ""

        # Routed experts are gated; shared experts are always active.
        self.num_head = num_head
        self.num_experts = num_experts
        self.num_activated = num_activated
        self.embed_dim = embed_dim
        self.balance_lr = balance_lr
        self.balance_mode = balance_mode
        self.bias_scope = bias_scope
        self.bias_update = bias_update
        self.bias_min_visits = num_experts if bias_min_visits == 0 else bias_min_visits
        self.bias_powerlaw_n = bias_powerlaw_n
        self.trust_region_low_limit, self.trust_region_high_limit = self._compute_trust_region_share_limits()
        self.maxvio_window = maxvio_window
        self.gate_nl = gate_nl
        self.slot_mapping = slot_mapping
        self.slot_vocab_size = slot_vocab_size
        self.slot_dedicated_size = slot_dedicated_size
        self.slot_map_path = slot_map_path
        self.slot_index_mode = slot_index_mode
        self.input_vocab_size = vocab_size
        self.per_head_table = per_head_table and num_head > 1

        self.shared_embedding = None
        if shared:
            shared_rows = slot_vocab_size * num_head if self.per_head_table else slot_vocab_size
            if shared_embedding_module is None:
                self.shared_embedding = _make_value_embedding(shared_rows, embed_dim, embedding_tag)
            else:
                _validate_value_embedding(
                    shared_embedding_module,
                    num_embeddings=shared_rows,
                    embedding_dim=embed_dim,
                    name="shared_embedding_module",
                )
                setattr(shared_embedding_module.weight, embedding_tag, True)
                object.__setattr__(self, "shared_embedding", shared_embedding_module)
        elif shared_embedding_module is not None:
            raise ValueError("shared_embedding_module requires shared=True")

        self.routed_embedding = None
        self.router = None
        if num_experts > 0:
            table_rows = slot_vocab_size * num_experts
            if self.per_head_table:
                table_rows *= num_head
            if routed_embedding_module is None:
                self.routed_embedding = _make_value_embedding(table_rows, embed_dim, embedding_tag)
            else:
                _validate_value_embedding(
                    routed_embedding_module,
                    num_embeddings=table_rows,
                    embedding_dim=embed_dim,
                    name="routed_embedding_module",
                )
                setattr(routed_embedding_module.weight, embedding_tag, True)
                object.__setattr__(self, "routed_embedding", routed_embedding_module)
            gate_dim = n_embd if router_dim is None else router_dim
            self.router = MoEValueRouter(
                n_embd=gate_dim,
                num_experts=num_experts,
                slot_vocab_size=slot_vocab_size,
                num_head=num_head,
                gate_type=gate_type,
                gate_kernel_size=gate_kernel_size,
                input_mode=router_input_mode,
                bias_per_head=bias_scope == "slot",
            )

        # Tracks routed expert load for logging/debugging (shared experts excluded).
        self.register_buffer("last_load", torch.zeros(max(1, num_experts)), persistent=False)
        balance_prefix = (slot_vocab_size, num_head) if bias_scope == "slot" and num_head > 1 else (slot_vocab_size,)
        self.register_buffer(
            "maxvio_counts",
            torch.zeros(*balance_prefix, max(1, num_experts), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("maxvio_steps", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("maxvio_windows_completed", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("last_maxvio", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_maxvio_eligible_slots", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("slot_map", torch.empty(0, dtype=torch.long), persistent=False)
        self.ddp, self.ddp_rank, _, self.ddp_world_size = get_dist_info()
        self.collect_eval_maxvio = False

    def _map_tokens_to_slots(self, idx: torch.Tensor) -> torch.Tensor:
        if self.slot_mapping == "none":
            return idx
        if self.slot_mapping == "mod":
            return torch.remainder(idx, self.slot_vocab_size)
        if self.slot_mapping == "headmod":
            head = self.slot_dedicated_size
            tail_slots = self.slot_vocab_size - head
            tail_source = (idx - head).clamp_min(0)
            tail_slots_idx = head + torch.remainder(tail_source, tail_slots)
            return torch.where(idx < head, idx, tail_slots_idx)
        if self.slot_mapping == "table":
            if self.slot_map.numel() == 0:
                raise RuntimeError("slot_map buffer is empty for table mapping")
            return self.slot_map[idx]
        raise RuntimeError(f"Unsupported slot mapping: {self.slot_mapping}")

    def _reload_slot_map(self):
        if self.slot_mapping != "table":
            return
        slot_map, _ = load_slot_map_artifact(
            self.slot_map_path,
            expected_vocab_size=self.input_vocab_size,
            runtime_slot_vocab_size=self.slot_vocab_size,
        )
        self.slot_map = slot_map.to(device=self.last_load.device)

    def init_weights(self, s: float, initialized_embeddings: set[int] | None = None):
        if self.shared_embedding is not None:
            _init_embedding_once(self.shared_embedding, s, initialized_embeddings)
        if self.routed_embedding is not None:
            _init_embedding_once(self.routed_embedding, s, initialized_embeddings)
        if self.router is not None:
            self.router.init_weights()
        self._reload_slot_map()
        self.last_load.zero_()
        self.maxvio_counts.zero_()
        self.maxvio_steps.zero_()
        self.maxvio_windows_completed.zero_()
        self.last_maxvio.zero_()
        self.last_maxvio_eligible_slots.zero_()

    def _route_tokens(self, logits: torch.Tensor, router_bias: torch.Tensor | None = None):
        if self.num_experts == 0 or self.num_activated == 0:
            return None, None

        topk = min(self.num_activated, self.num_experts)
        if self.gate_nl.startswith("softmax"):
            base_scores = F.softmax(logits, dim=-1)
        else:
            base_scores = torch.sigmoid(logits)

        select_scores = base_scores if router_bias is None else base_scores + router_bias
        _, topk_idx = torch.topk(select_scores, k=topk, dim=-1, sorted=False)

        topk_weight = base_scores.gather(-1, topk_idx)
        if self.gate_nl in {"softmax-norm", "softmax-norm-detach", "sigmoid-norm"}:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-10
            if self.gate_nl == "softmax-norm-detach":
                denominator = denominator.detach()
            topk_weight = topk_weight / denominator

        return topk_idx, topk_weight

    def _route_all_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        if self.gate_nl.startswith("softmax"):
            return F.softmax(logits, dim=-1)

        base_scores = torch.sigmoid(logits)
        if self.gate_nl == "sigmoid-norm":
            denominator = base_scores.sum(dim=-1, keepdim=True) + 1e-10
            return base_scores / denominator
        return base_scores

    def _compute_slot_expert_counts(self, slot_idx, topk_idx):
        if topk_idx.ndim == slot_idx.ndim + 2:
            if topk_idx.ndim != 4:
                raise ValueError("Head-specific MoE routing counts expect topk_idx shape [B, T, H, K]")
            head_count = topk_idx.size(-2)
            slot_view = slot_idx.unsqueeze(-1).expand(*topk_idx.shape[:-1])
            head_view = torch.arange(head_count, device=topk_idx.device).view(1, 1, head_count)
            head_view = head_view.expand(*topk_idx.shape[:-1])
            flat_slots = slot_view.unsqueeze(-1).expand_as(topk_idx).reshape(-1)
            flat_heads = head_view.unsqueeze(-1).expand_as(topk_idx).reshape(-1)
            flat_experts = topk_idx.reshape(-1)
            if flat_experts.numel() == 0:
                return None
            flat_ids = (flat_slots * head_count + flat_heads) * self.num_experts + flat_experts
            counts = torch.zeros(
                self.slot_vocab_size * head_count * self.num_experts,
                device=flat_ids.device,
                dtype=torch.float32,
            )
            counts.scatter_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.float32))
            counts = counts.view(self.slot_vocab_size, head_count, self.num_experts)
            if self.bias_scope == "slot_shared_head":
                return counts.sum(dim=1)
            return counts

        slot_view = slot_idx
        while slot_view.ndim < topk_idx.ndim - 1:
            slot_view = slot_view.unsqueeze(2)
        slot_view = slot_view.expand(*topk_idx.shape[:-1])
        flat_slots = slot_view.unsqueeze(-1).expand_as(topk_idx).reshape(-1)
        flat_experts = topk_idx.reshape(-1)
        if flat_experts.numel() == 0:
            return None
        flat_ids = flat_slots * self.num_experts + flat_experts
        counts = torch.zeros(self.slot_vocab_size * self.num_experts, device=flat_ids.device, dtype=torch.float32)
        counts.scatter_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.float32))
        return counts.view(self.slot_vocab_size, self.num_experts)

    def _compute_full_routing_counts(self, slot_idx, head_count: int):
        flat_slots = slot_idx.reshape(-1)
        if flat_slots.numel() == 0:
            return None
        slot_counts = torch.zeros(self.slot_vocab_size, device=flat_slots.device, dtype=torch.float32)
        slot_counts.scatter_add_(0, flat_slots, torch.ones_like(flat_slots, dtype=torch.float32))
        if head_count > 1:
            counts = slot_counts.view(-1, 1, 1).expand(-1, head_count, self.num_experts).clone()
            if self.bias_scope == "slot_shared_head":
                return counts.sum(dim=1)
            return counts
        return slot_counts.unsqueeze(-1).expand(-1, self.num_experts).clone()

    def _compute_slot_balance_metrics(self, counts: torch.Tensor, eligible: torch.Tensor):
        if not eligible.any():
            return None
        eligible_counts = counts[eligible]
        slot_totals = eligible_counts.sum(dim=-1)
        mean_load = slot_totals / self.num_experts
        max_load = eligible_counts.max(dim=-1).values
        slot_maxvio = (max_load - mean_load) / mean_load.clamp_min(1e-6)
        slot_probs = eligible_counts / slot_totals.unsqueeze(-1)
        target = 1.0 / self.num_experts
        slot_absolute_deviation = torch.abs(slot_probs - target).amax(dim=-1)
        return {
            "maxvio": slot_maxvio.mean(),
            "absolute_load_deviation": slot_absolute_deviation.mean(),
            "eligible_slots": int(eligible.sum().item()),
        }

    def _compute_trust_region_share_limits(self) -> tuple[float, float]:
        weights = [math.pow(self.bias_powerlaw_n, -rank) for rank in range(self.num_experts)]
        denom = sum(weights)
        probs = [weight / denom for weight in weights]
        return probs[-1], probs[0]

    def _apply_trust_region_bias_update(self, counts: torch.Tensor):
        if self.router is None or self.balance_lr <= 0:
            return
        slot_totals = counts.sum(dim=-1, keepdim=True)
        eligible = slot_totals >= self.bias_min_visits
        bias = self.router.router_bias
        lower_bound = slot_totals * self.trust_region_low_limit
        upper_bound = slot_totals * self.trust_region_high_limit
        below = eligible & (counts < lower_bound)
        above = eligible & (counts > upper_bound)
        in_band = eligible & ~(below | above)
        bias.add_(below.to(dtype=bias.dtype), alpha=self.balance_lr)
        bias.add_(above.to(dtype=bias.dtype), alpha=-self.balance_lr)
        shrink = torch.minimum(bias.abs(), torch.full_like(bias, self.balance_lr)) * torch.sign(bias)
        bias.sub_(in_band.to(dtype=bias.dtype) * shrink)

    def _finalize_window_maxvio(self):
        slot_totals = self.maxvio_counts.sum(dim=-1)
        eligible = slot_totals >= self.num_experts
        metrics = self._compute_slot_balance_metrics(self.maxvio_counts, eligible)
        self.last_maxvio_eligible_slots.fill_(0 if metrics is None else metrics["eligible_slots"])
        if metrics is not None:
            self.last_maxvio.copy_(metrics["maxvio"])
        else:
            self.last_maxvio.zero_()
        self.maxvio_windows_completed += 1
        self.maxvio_counts.zero_()
        self.maxvio_steps.zero_()

    def _update_balance(self, slot_idx, topk_idx):
        if not self.training or self.num_experts <= 0 or topk_idx is None or self.bias_update == "none":
            return
        with torch.no_grad():
            counts = self._compute_slot_expert_counts(slot_idx, topk_idx)
            if counts is None:
                return
            if self.ddp and dist.is_available() and dist.is_initialized():
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            if self.router is not None and self.balance_mode == "bias" and self.balance_lr > 0:
                if self.bias_update == "trust_region":
                    self._apply_trust_region_bias_update(counts)
                else:
                    slot_totals = counts.sum(dim=-1, keepdim=True)
                    observed = slot_totals > 0
                    mean_load = slot_totals / self.num_experts
                    error = torch.where(observed, mean_load - counts, torch.zeros_like(counts))
                    self.router.router_bias += self.balance_lr * torch.sign(error)
            total = counts.sum()
            reduce_dims = tuple(range(counts.ndim - 1))
            self.last_load.copy_(counts.sum(dim=reduce_dims) / (total + 1e-6))
            if self.maxvio_window > 0:
                self.maxvio_counts += counts
                self.maxvio_steps += 1
                if int(self.maxvio_steps.item()) >= self.maxvio_window:
                    self._finalize_window_maxvio()

    def reset_eval_maxvio(self):
        self.collect_eval_maxvio = True
        self.maxvio_counts.zero_()

    def finish_eval_maxvio(self):
        self.collect_eval_maxvio = False
        counts = self.maxvio_counts.clone()
        if self.ddp and dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        self.maxvio_counts.zero_()
        slot_totals = counts.sum(dim=-1)
        visited = slot_totals > 0
        metrics = self._compute_slot_balance_metrics(counts, visited)
        if metrics is None:
            return None
        return {
            "maxvio": metrics["maxvio"].item(),
            "absolute_load_deviation": metrics["absolute_load_deviation"].item(),
            "visited_slots": metrics["eligible_slots"],
        }

    def forward(
        self,
        idx: torch.Tensor,
        x: torch.Tensor,
        router_input: torch.Tensor | None = None,
        prev_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if idx is None or x is None:
            raise ValueError("MoEValueEmbeddings requires idx and x")
        B, T = idx.size()
        slot_source_idx = idx
        if self.slot_index_mode == "bigram":
            slot_source_idx = compute_moe_bigram_hash(idx, self.slot_vocab_size, prev_tokens=prev_tokens)
        slot_idx = self._map_tokens_to_slots(slot_source_idx)

        out = None
        aux_loss = x.new_zeros(())
        if self.shared_embedding is not None:
            if self.per_head_table:
                head_ids = torch.arange(self.num_head, device=slot_idx.device).view(1, 1, self.num_head)
                shared_idx = slot_idx.unsqueeze(-1) * self.num_head + head_ids
                shared = self.shared_embedding(shared_idx)
            else:
                shared = self.shared_embedding(slot_idx).view(B, T, 1, self.embed_dim).repeat(1, 1, self.num_head, 1)
            out = shared

        if self.routed_embedding is not None and self.router is not None and self.num_activated > 0:
            gate_input = x if router_input is None else router_input
            if gate_input.ndim == 3 and self.num_head > 1 and self.router.input_mode != "hidden":
                raise ValueError("MoE router_input must be 4D when num_head > 1")
            logits = self.router(gate_input)
            # disable bias for now
            # logits = logits + self.router.router_bias.to(dtype=logits.dtype)
            router_bias = None
            if self.balance_mode == "bias" and self.training and self.bias_update != "none":
                router_bias = self.router.router_bias[slot_idx].to(dtype=logits.dtype)
                if logits.ndim == 4 and router_bias.ndim == 3:
                    router_bias = router_bias.unsqueeze(2)
            full_routing = self.num_activated == self.num_experts
            if self.balance_mode == "aux" and self.training:
                flat_logits = logits.reshape(-1, self.num_experts)
                aux_loss = load_balancing_loss_func(
                    (flat_logits,),
                    num_experts=self.num_experts,
                    top_k=min(self.num_activated, self.num_experts),
                )
            if full_routing:
                weights = self._route_all_tokens(logits)
                head_count = logits.size(2) if logits.ndim == 4 else 1
                counts = None
                if self.balance_mode == "bias" and self.training and self.bias_update != "none":
                    counts = self._compute_full_routing_counts(slot_idx, head_count)
                    if counts is not None:
                        if self.ddp and dist.is_available() and dist.is_initialized():
                            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                        total = counts.sum()
                        reduce_dims = tuple(range(counts.ndim - 1))
                        self.last_load.copy_(counts.sum(dim=reduce_dims) / (total + 1e-6))
                        if self.maxvio_window > 0:
                            self.maxvio_counts += counts
                            self.maxvio_steps += 1
                            if int(self.maxvio_steps.item()) >= self.maxvio_window:
                                self._finalize_window_maxvio()
                elif self.collect_eval_maxvio:
                    with torch.no_grad():
                        counts = self._compute_full_routing_counts(slot_idx, head_count)
                        if counts is not None:
                            self.maxvio_counts += counts

                if weights.ndim == 3:
                    weights = weights.unsqueeze(2)
                if self.per_head_table:
                    head_ids = torch.arange(weights.size(2), device=slot_idx.device).view(1, 1, -1, 1)
                    base_idx = (slot_idx.unsqueeze(-1).unsqueeze(-1) * self.num_head + head_ids) * self.num_experts
                else:
                    base_idx = slot_idx.unsqueeze(-1).unsqueeze(-1) * self.num_experts  # (B, T, 1, 1)
                expert_ids = torch.arange(self.num_experts, device=slot_idx.device).view(1, 1, 1, self.num_experts)
                all_flat_idx = base_idx + expert_ids  # (B, T, 1, E)
                routed_all = F.embedding(all_flat_idx, self.routed_embedding.weight)
                if routed_all.size(2) != weights.size(2):
                    routed_all = routed_all.expand(-1, -1, weights.size(2), -1, -1)
                routed_out = (weights.unsqueeze(-1) * routed_all).sum(dim=3)
            else:
                topk_idx, weights = self._route_tokens(logits, router_bias)
                if self.balance_mode == "bias" and self.training:
                    self._update_balance(slot_idx, topk_idx)
                elif self.collect_eval_maxvio:
                    with torch.no_grad():
                        counts = self._compute_slot_expert_counts(slot_idx, topk_idx)
                        if counts is not None:
                            self.maxvio_counts += counts

                if topk_idx.ndim == 3:
                    topk_idx = topk_idx.unsqueeze(2)
                    weights = weights.unsqueeze(2)

                if self.per_head_table:
                    head_ids = torch.arange(topk_idx.size(2), device=slot_idx.device).view(1, 1, -1, 1)
                    base_idx = (slot_idx.unsqueeze(-1).unsqueeze(-1) * self.num_head + head_ids) * self.num_experts
                else:
                    base_idx = slot_idx.unsqueeze(-1).unsqueeze(-1) * self.num_experts  # (B, T, 1, 1)
                flat_idx = base_idx + topk_idx  # (B, T, H, K)
                routed_selected = F.embedding(flat_idx, self.routed_embedding.weight)  # (B, T, H, K, D)
                routed_out = (weights.unsqueeze(-1) * routed_selected).sum(dim=3)
            out = routed_out if out is None else out + routed_out

        if out is None:
            raise RuntimeError("MoE embeddings produced no output; check configuration.")
        # Keep routing math in fp32, but hand back activations in the caller's
        # dtype so value-residuals stay compatible with bf16 attention kernels.
        if out.dtype != x.dtype:
            out = out.to(dtype=x.dtype)
        return out.view(B, T, self.num_head * self.embed_dim), aux_loss


class ValueEmbeddings(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        kv_dim: int,
        layer_ids: list[int],
    ):
        super().__init__()
        if len(layer_ids) == 0:
            raise ValueError("At least one layer_id must be specified for ValueEmbeddings")
        self.layer_ids = layer_ids
        self.layers = nn.ModuleDict({
            str(layer_id): nn.Embedding(vocab_size, kv_dim)
            for layer_id in layer_ids
        })
        for embedding in self.layers.values():
            setattr(embedding.weight, "value_embedding_is_table", True)

    def init_weights(self, s: float):
        for embedding in self.layers.values():
            torch.nn.init.uniform_(embedding.weight, -s, s)

    def forward(self, layer_idx: int, idx: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        key = str(layer_idx)
        if key not in self.layers:
            return None, idx.new_zeros(())
        return self.layers[key](idx), idx.new_zeros(())


class MoEValueEmbeddings(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        n_embd: int,
        kv_dim: int,
        n_kv_head: int,
        layer_ids: list[int],
        num_experts: int,
        shared: bool,
        num_activated: int,
        balance_lr: float,
        balance_mode: str = "bias",
        bias_scope: str = "slot",
        bias_update: str = "deepseek_moe",
        bias_min_visits: int = 32,
        bias_powerlaw_n: float = 1.4,
        maxvio_window: int = 0,
        gate_nl: str,
        gate_type: str = "linear",
        gate_kernel_size: int = 4,
        router_input_mode: str = "value",
        per_head_table: bool = False,
        slot_mapping: str = "none",
        slot_vocab_size: int = 0,
        slot_dedicated_size: int = 0,
        slot_map_path: str = "",
        slot_index_mode: str = "token",
        network_shared_table: bool = False,
    ):
        super().__init__()
        if n_kv_head <= 0:
            raise ValueError("n_kv_head must be > 0")
        if kv_dim % n_kv_head != 0:
            raise ValueError("kv_dim must be divisible by n_kv_head")
        if len(layer_ids) == 0:
            raise ValueError("At least one layer_id must be specified for MoEValueEmbeddings")
        self.layer_ids = layer_ids
        self.layer_map = {layer_id: idx for idx, layer_id in enumerate(layer_ids)}
        self.n_kv_head = n_kv_head
        self.head_dim = kv_dim // n_kv_head
        self.network_shared_table = network_shared_table
        effective_slot_vocab_size = slot_vocab_size if slot_vocab_size > 0 else vocab_size
        self.network_shared_shared_embedding = None
        self.network_shared_routed_embedding = None
        if network_shared_table:
            if shared:
                shared_rows = (
                    effective_slot_vocab_size * n_kv_head
                    if per_head_table and n_kv_head > 1
                    else effective_slot_vocab_size
                )
                self.network_shared_shared_embedding = _make_value_embedding(
                    shared_rows,
                    self.head_dim,
                    "value_embedding_is_table",
                )
            if num_experts > 0:
                table_rows = effective_slot_vocab_size * num_experts
                if per_head_table and n_kv_head > 1:
                    table_rows *= n_kv_head
                self.network_shared_routed_embedding = _make_value_embedding(
                    table_rows,
                    self.head_dim,
                    "value_embedding_is_table",
                )
        self.layers = nn.ModuleList([
            _MoEEmbeddingBase(
                vocab_size=vocab_size,
                n_embd=n_embd,
                router_dim=None if router_input_mode == "hidden" else kv_dim // n_kv_head,
                embed_dim=kv_dim // n_kv_head,
                num_head=n_kv_head,
                num_experts=num_experts,
                shared=shared,
                num_activated=num_activated,
                balance_lr=balance_lr,
                balance_mode=balance_mode,
                bias_scope=bias_scope,
                bias_update=bias_update,
                bias_min_visits=bias_min_visits,
                bias_powerlaw_n=bias_powerlaw_n,
                maxvio_window=maxvio_window,
                gate_nl=gate_nl,
                router_input_mode=router_input_mode,
                per_head_table=per_head_table,
                gate_type=gate_type,
                gate_kernel_size=gate_kernel_size,
                slot_mapping=slot_mapping,
                slot_vocab_size=slot_vocab_size,
                slot_dedicated_size=slot_dedicated_size,
                slot_map_path=slot_map_path,
                slot_index_mode=slot_index_mode,
                shared_embedding_module=self.network_shared_shared_embedding,
                routed_embedding_module=self.network_shared_routed_embedding,
                embedding_tag="value_embedding_is_table",
            )
            for _ in layer_ids
        ])

    def init_weights(self, s: float):
        initialized_embeddings: set[int] = set()
        for layer in self.layers:
            layer.init_weights(s, initialized_embeddings=initialized_embeddings)

    def iter_layers(self):
        return iter(self.layers)

    def reset_eval_maxvio(self):
        for layer in self.layers:
            layer.reset_eval_maxvio()

    def finish_eval_maxvio(self):
        maxvios = []
        absolute_load_deviations = []
        visited_slots = []
        for layer in self.layers:
            stats = layer.finish_eval_maxvio()
            if stats is None:
                continue
            maxvios.append(stats["maxvio"])
            absolute_load_deviations.append(stats["absolute_load_deviation"])
            visited_slots.append(stats["visited_slots"])
        if not maxvios:
            return None
        return {
            "maxvio_mean": sum(maxvios) / len(maxvios),
            "absolute_load_deviation_mean": sum(absolute_load_deviations) / len(absolute_load_deviations),
            "visited_slots_mean": sum(visited_slots) / len(visited_slots),
            "layers": len(maxvios),
        }

    def forward(
        self,
        layer_idx: int,
        idx: torch.Tensor,
        x: torch.Tensor,
        router_input: torch.Tensor | None = None,
        prev_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        pos = self.layer_map.get(layer_idx)
        if pos is None:
            return None, x.new_zeros(())
        ve, aux_loss = self.layers[pos](idx, x, router_input=router_input, prev_tokens=prev_tokens)
        return ve, aux_loss
