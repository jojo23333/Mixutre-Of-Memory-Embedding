import pytest
import torch

from nanochat.moe_ve_gpt import MoEValueRouter, _MoEEmbeddingBase


def _build_test_layer() -> _MoEEmbeddingBase:
    layer = _MoEEmbeddingBase(
        vocab_size=8,
        n_embd=1,
        embed_dim=2,
        num_head=1,
        num_experts=3,
        shared=False,
        num_activated=1,
        balance_lr=0.0,
        balance_mode="bias",
        gate_nl="softmax",
        slot_mapping="none",
        embedding_tag="value_embedding_is_table",
        gate_type="linear",
        router_input_mode="value",
    )
    with torch.no_grad():
        layer.routed_embedding.weight.zero_()
        layer.routed_embedding.weight[3].fill_(1.0)
        layer.routed_embedding.weight[4].fill_(2.0)
        layer.routed_embedding.weight[5].fill_(3.0)
        layer.router.weight.zero_()
        layer.router.bias.copy_(torch.tensor([3.0, 2.0, 1.0]))
    return layer


def test_moe_bias_balance_updates_are_slot_conditioned():
    layer = _MoEEmbeddingBase(
        vocab_size=4,
        n_embd=1,
        embed_dim=2,
        num_head=1,
        num_experts=3,
        shared=False,
        num_activated=1,
        balance_lr=0.5,
        balance_mode="bias",
        gate_nl="softmax",
        slot_mapping="none",
        embedding_tag="value_embedding_is_table",
        gate_type="linear",
        router_input_mode="value",
    )
    layer.train()
    slot_idx = torch.tensor([[0, 1]], dtype=torch.long)
    topk_idx = torch.tensor([[[0], [1]]], dtype=torch.long)

    layer._update_balance(slot_idx, topk_idx)

    expected_slot0 = torch.tensor([-0.5, 0.5, 0.5], dtype=layer.router.router_bias.dtype)
    expected_slot1 = torch.tensor([0.5, -0.5, 0.5], dtype=layer.router.router_bias.dtype)
    torch.testing.assert_close(layer.router.router_bias[0], expected_slot0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(layer.router.router_bias[1], expected_slot1, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        layer.router.router_bias[2:],
        torch.zeros_like(layer.router.router_bias[2:]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        layer.last_load,
        torch.tensor([0.5, 0.5, 0.0], dtype=layer.last_load.dtype),
        rtol=0.0,
        atol=1e-6,
    )




def test_eval_maxvio_uses_all_visited_slots_average():
    layer = _build_test_layer()
    layer.eval()
    layer.reset_eval_maxvio()

    idx = torch.tensor([[1, 2]], dtype=torch.long)
    x = torch.ones(1, 2, 1)
    layer(idx, x)

    stats = layer.finish_eval_maxvio()

    assert stats is not None
    assert stats["visited_slots"] == 2
    torch.testing.assert_close(
        torch.tensor(stats["maxvio"]),
        torch.tensor(2.0),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.tensor(stats["absolute_load_deviation"]),
        torch.tensor(2.0 / 3.0),
        rtol=0.0,
        atol=1e-6,
    )


def test_window_maxvio_uses_overload_ratio_definition():
    layer = _build_test_layer()
    layer.train()
    layer.maxvio_window = 1

    slot_idx = torch.tensor([[1, 2]], dtype=torch.long)
    topk_idx = torch.tensor([[[0], [0]]], dtype=torch.long)
    layer._update_balance(slot_idx, topk_idx)

    assert int(layer.maxvio_windows_completed.item()) == 1
    assert int(layer.last_maxvio_eligible_slots.item()) == 0
    torch.testing.assert_close(layer.last_maxvio, torch.zeros_like(layer.last_maxvio), rtol=0.0, atol=0.0)

    layer.maxvio_window = 1
    slot_idx = torch.tensor([[1, 1, 1]], dtype=torch.long)
    topk_idx = torch.tensor([[[0], [0], [0]]], dtype=torch.long)
    layer._update_balance(slot_idx, topk_idx)

    assert int(layer.maxvio_windows_completed.item()) == 2
    assert int(layer.last_maxvio_eligible_slots.item()) == 1
    torch.testing.assert_close(
        layer.last_maxvio,
        torch.tensor(2.0, dtype=layer.last_maxvio.dtype),
        rtol=0.0,
        atol=1e-6,
    )


def test_moe_full_routing_path_matches_dense_weighted_sum():
    layer = _MoEEmbeddingBase(
        vocab_size=4,
        n_embd=1,
        embed_dim=2,
        num_head=1,
        num_experts=3,
        shared=False,
        num_activated=3,
        balance_lr=0.5,
        balance_mode="bias",
        gate_nl="softmax",
        slot_mapping="none",
        embedding_tag="value_embedding_is_table",
        gate_type="linear",
        router_input_mode="value",
    )
    with torch.no_grad():
        layer.routed_embedding.weight.zero_()
        layer.routed_embedding.weight[3].copy_(torch.tensor([1.0, 10.0]))
        layer.routed_embedding.weight[4].copy_(torch.tensor([2.0, 20.0]))
        layer.routed_embedding.weight[5].copy_(torch.tensor([4.0, 40.0]))
        layer.router.weight.zero_()
        layer.router.bias.copy_(torch.tensor([3.0, 2.0, 1.0]))

    layer.train()
    idx = torch.tensor([[1]], dtype=torch.long)
    x = torch.ones(1, 1, 1)
    out, aux_loss = layer(idx, x)

    probs = torch.softmax(torch.tensor([3.0, 2.0, 1.0]), dim=0)
    expected = (
        probs[0] * torch.tensor([1.0, 10.0])
        + probs[1] * torch.tensor([2.0, 20.0])
        + probs[2] * torch.tensor([4.0, 40.0])
    )

    torch.testing.assert_close(aux_loss, torch.zeros_like(aux_loss), rtol=0.0, atol=0.0)
    torch.testing.assert_close(out[0, 0], expected.to(dtype=out.dtype), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(
        layer.last_load,
        torch.full_like(layer.last_load, 1.0 / 3.0),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        layer.router.router_bias,
        torch.zeros_like(layer.router.router_bias),
        rtol=0.0,
        atol=0.0,
    )


def test_moe_conv1d_router_is_causal():
    router = MoEValueRouter(
        n_embd=1,
        num_experts=1,
        slot_vocab_size=8,
        num_head=1,
        gate_type="conv1d",
        gate_kernel_size=3,
        input_mode="hidden",
    )
    with torch.no_grad():
        router.gate.weight.fill_(1.0)
        router.gate.bias.zero_()

    hidden = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    logits = router(hidden)

    expected = torch.tensor([[[1.0], [3.0], [6.0], [9.0]]])
    torch.testing.assert_close(logits, expected, rtol=0.0, atol=1e-6)


def test_route_tokens_keeps_headwise_shape_and_bias_broadcast():
    layer = _MoEEmbeddingBase(
        vocab_size=8,
        n_embd=1,
        embed_dim=2,
        num_head=2,
        num_experts=3,
        shared=False,
        num_activated=1,
        balance_lr=0.0,
        balance_mode="bias",
        gate_nl="softmax-norm-detach",
        slot_mapping="none",
        embedding_tag="value_embedding_is_table",
        gate_type="linear",
        router_input_mode="value",
    )
    logits = torch.randn(1, 2, 2, 3)
    router_bias = torch.randn(1, 2, 1, 3)
    topk_idx, topk_weight = layer._route_tokens(logits, router_bias)
    assert topk_idx.shape == (1, 2, 2, 1)
    assert topk_weight.shape == (1, 2, 2, 1)




def test_none_bias_update_skips_balance_compute_and_updates():
    layer = _MoEEmbeddingBase(
        vocab_size=4,
        n_embd=1,
        embed_dim=2,
        num_head=1,
        num_experts=3,
        shared=False,
        num_activated=1,
        balance_lr=0.5,
        balance_mode="bias",
        bias_update="none",
        gate_nl="sigmoid-norm",
        slot_mapping="none",
        embedding_tag="value_embedding_is_table",
        gate_type="linear",
        router_input_mode="value",
    )
    layer.train()
    slot_idx = torch.tensor([[0, 1]], dtype=torch.long)
    topk_idx = torch.tensor([[[0], [1]]], dtype=torch.long)

    layer._update_balance(slot_idx, topk_idx)

    torch.testing.assert_close(
        layer.router.router_bias,
        torch.zeros_like(layer.router.router_bias),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        layer.last_load,
        torch.zeros_like(layer.last_load),
        rtol=0.0,
        atol=0.0,
    )


def test_trust_region_bias_update_uses_window_thresholds_and_decay():
    layer = _MoEEmbeddingBase(
        vocab_size=4,
        n_embd=1,
        embed_dim=2,
        num_head=1,
        num_experts=3,
        shared=False,
        num_activated=1,
        balance_lr=0.1,
        balance_mode="bias",
        bias_scope="slot",
        bias_update="trust_region",
        bias_min_visits=5,
        bias_powerlaw_n=2.0,
        gate_nl="sigmoid-norm",
        slot_mapping="none",
        embedding_tag="value_embedding_is_table",
        gate_type="linear",
        router_input_mode="value",
    )
    with torch.no_grad():
        layer.router.router_bias.zero_()
        layer.router.router_bias[1].copy_(torch.tensor([0.3, -0.3, 0.2], dtype=layer.router.router_bias.dtype))
        layer.router.router_bias[2].copy_(torch.tensor([0.4, 0.0, -0.4], dtype=layer.router.router_bias.dtype))

    counts = torch.zeros(4, 3, dtype=torch.float32)
    counts[1] = torch.tensor([7.0, 2.0, 1.0])
    counts[2] = torch.tensor([1.0, 1.0, 0.0])

    layer._apply_trust_region_bias_update(counts)
    torch.testing.assert_close(layer.router.router_bias[1], torch.tensor([0.2, -0.2, 0.3]), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(layer.router.router_bias[2], torch.tensor([0.4, 0.0, -0.4]), rtol=0.0, atol=0.0)


def test_softmax_norm_detach_keeps_top1_gate_gradient():
    layer = _MoEEmbeddingBase(
        vocab_size=8,
        n_embd=1,
        embed_dim=2,
        num_head=1,
        num_experts=3,
        shared=False,
        num_activated=1,
        balance_lr=0.0,
        balance_mode="bias",
        gate_nl="softmax-norm-detach",
        slot_mapping="none",
        embedding_tag="value_embedding_is_table",
        gate_type="linear",
        router_input_mode="value",
    )
    logits = torch.tensor([[[2.0, 1.0, 0.0]]], requires_grad=True)
    _, topk_weight = layer._route_tokens(logits)
    topk_weight.sum().backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) > 0
