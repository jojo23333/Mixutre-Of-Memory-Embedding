import torch

from nanochat.engine import KVCache
from nanochat.gpt import GPTConfig
from nanochat.muon import Muon
from nanochat.qwen3_0p5b_model import Qwen3_0p5B


def _make_config(**overrides):
    config = dict(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        head_dim=16,
        weight_tying=True,
    )
    config.update(overrides)
    return GPTConfig(**config)


def test_qwen3_0p5b_uses_explicit_head_dim_and_2k_rope_theta():
    cfg = GPTConfig(
        sequence_len=2048,
        vocab_size=32768,
        n_layer=28,
        n_head=16,
        n_kv_head=8,
        n_embd=1024,
        head_dim=128,
        weight_tying=True,
    )
    with torch.device("meta"):
        model = Qwen3_0p5B(cfg)

    assert model.head_dim == 128
    assert model.rope_theta == 10000.0
    assert model.lm_head is None
    assert sum(p.numel() for p in model.parameters()) == 474_021_888


def test_qwen3_0p5b_forward_pass():
    model = Qwen3_0p5B(_make_config())
    model.init_weights()

    idx = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    loss = model(idx, targets)

    assert torch.isfinite(loss)


def test_qwen3_0p5b_kv_cache_uses_explicit_head_dim():
    cfg = _make_config()
    model = Qwen3_0p5B(cfg)
    model.init_weights()
    kv_cache = KVCache(
        batch_size=1,
        num_heads=cfg.n_kv_head,
        seq_len=8,
        head_dim=cfg.head_dim,
        num_layers=cfg.n_layer,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    idx = torch.randint(0, 128, (1, 4))

    logits = model(idx, kv_cache=kv_cache)

    assert logits.shape == (1, 4, 128)
    assert kv_cache.get_pos() == 4


def test_qwen3_0p5b_dense_value_embeddings_use_adamw():
    model = Qwen3_0p5B(
        _make_config(
            dense_ve_enabled=True,
            value_embeds_layers=[1],
        )
    )
    model.init_weights()
    adamw_optimizer, muon_optimizer = model.setup_optimizers()

    assert isinstance(muon_optimizer, Muon)
    adam_param_ids = {id(param) for group in adamw_optimizer.param_groups for param in group["params"]}
    muon_param_ids = {id(param) for group in muon_optimizer.param_groups for param in group["params"]}
    value_table_param_ids = {
        id(param)
        for param in model.parameters()
        if getattr(param, "value_embedding_is_table", False)
    }

    assert value_table_param_ids
    assert value_table_param_ids <= adam_param_ids
    assert value_table_param_ids.isdisjoint(muon_param_ids)
    assert model.transformer.h[1].self_attn.ve_gate is not None
    assert model.transformer.h[0].self_attn.ve_gate is None

    idx = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    loss, aux_loss = model(idx, targets, return_aux_loss=True)

    assert torch.isfinite(loss)
    assert torch.isfinite(aux_loss)


def test_qwen3_0p5b_tied_embeddings_use_unembedding_lr():
    model = Qwen3_0p5B(_make_config(weight_tying=True))
    model.init_weights()
    adamw_optimizer, _ = model.setup_optimizers(
        unembedding_lr=0.004,
        embedding_lr=0.3,
    )

    embedding_param = model.transformer.wte.weight
    embedding_group = next(
        group
        for group in adamw_optimizer.param_groups
        if any(param is embedding_param for param in group["params"])
    )
    expected_lr = 0.004 * (model.config.n_embd / 768) ** -0.5

    assert embedding_group["lr"] == expected_lr


def test_qwen3_0p5b_moe_value_embeddings_use_adamw():
    model = Qwen3_0p5B(
        _make_config(
            moe_ve_enabled=True,
            moe_ve_setting="0_1_2",
            moe_ve_gate_nl="sigmoid-norm",
            moe_ve_router_input="hidden",
            moe_ve_bias_scope="slot",
            moe_ve_bias_update="trust_region",
            moe_ve_bias_min_visits=0,
            moe_ve_bias_powerlaw_n=1.2,
            moe_ve_slot_mapping="none",
            value_embeds_layers=[1],
        )
    )
    model.init_weights()
    adamw_optimizer, muon_optimizer = model.setup_optimizers()

    adam_param_ids = {id(param) for group in adamw_optimizer.param_groups for param in group["params"]}
    muon_param_ids = {id(param) for group in muon_optimizer.param_groups for param in group["params"]}
    value_table_param_ids = {
        id(param)
        for param in model.parameters()
        if getattr(param, "value_embedding_is_table", False)
    }

    assert value_table_param_ids
    assert value_table_param_ids <= adam_param_ids
    assert value_table_param_ids.isdisjoint(muon_param_ids)

    idx = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    loss, aux_loss = model(idx, targets, return_aux_loss=True)

    assert torch.isfinite(loss)
    assert torch.isfinite(aux_loss)
