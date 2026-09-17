import torch

from nanochat.gpt import GPTConfig
from nanochat.muon import Muon
from nanochat.stemgpt_350m_model import StemGPT350M


def _make_config(**overrides):
    config = dict(
        sequence_len=16,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
        stem_multiple_of=32,
        stem_layers=None,
    )
    config.update(overrides)
    return GPTConfig(**config)


def test_default_stem_layers_match_upstream_pattern():
    with torch.device("meta"):
        model = StemGPT350M(_make_config())
    assert model.stem_layers == [1, 2, 3]


def test_dense_and_half_stem_forward_passes():
    dense_model = StemGPT350M(_make_config(stem_layers=[]))
    dense_model.init_weights()
    stem_model = StemGPT350M(_make_config(stem_layers=[1, 3]))
    stem_model.init_weights()

    idx = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))

    dense_loss = dense_model(idx, targets)
    stem_loss = stem_model(idx, targets)

    assert torch.isfinite(dense_loss)
    assert torch.isfinite(stem_loss)


def test_setup_optimizers_keeps_stem_embeddings_out_of_muon():
    model = StemGPT350M(_make_config(stem_layers=[1, 3]))
    model.init_weights()
    adamw_optimizer, muon_optimizer = model.setup_optimizers()

    assert isinstance(muon_optimizer, Muon)

    adam_param_ids = {
        id(param)
        for group in adamw_optimizer.param_groups
        for param in group["params"]
    }
    muon_param_ids = {
        id(param)
        for group in muon_optimizer.param_groups
        for param in group["params"]
    }
    stem_param_ids = {
        id(param)
        for param in model.parameters()
        if getattr(param, "stem_is_embedding", False)
    }

    assert stem_param_ids
    assert stem_param_ids <= adam_param_ids
    assert stem_param_ids.isdisjoint(muon_param_ids)


def test_moe_value_embeddings_route_through_stemgpt_attention():
    model = StemGPT350M(
        _make_config(
            stem_layers=[],
            moe_ve_enabled=True,
            moe_ve_setting="0_2_6",
            moe_ve_gate_nl="sigmoid-norm",
            moe_ve_router_input="hidden",
            moe_ve_bias_scope="slot",
            moe_ve_bias_update="trust_region",
            moe_ve_bias_min_visits=0,
            moe_ve_bias_powerlaw_n=1.2,
            moe_ve_slot_mapping="none",
            moe_ve_slot_vocab_size=128,
            value_embeds_layers=[1, 3],
        )
    )
    model.init_weights()
    adamw_optimizer, muon_optimizer = model.setup_optimizers()

    adam_param_ids = {
        id(param)
        for group in adamw_optimizer.param_groups
        for param in group["params"]
    }
    muon_param_ids = {
        id(param)
        for group in muon_optimizer.param_groups
        for param in group["params"]
    }
    value_table_param_ids = {
        id(param)
        for param in model.parameters()
        if getattr(param, "value_embedding_is_table", False)
    }

    assert value_table_param_ids
    assert value_table_param_ids <= adam_param_ids
    assert value_table_param_ids.isdisjoint(muon_param_ids)
    assert model.transformer.h[1].attention.ve_gate is not None
    assert model.transformer.h[0].attention.ve_gate is None

    idx = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    loss, aux_loss = model(idx, targets, return_aux_loss=True)

    assert torch.isfinite(loss)
    assert torch.isfinite(aux_loss)


def test_dense_value_embeddings_route_through_stemgpt_attention():
    model = StemGPT350M(
        _make_config(
            stem_layers=[],
            dense_ve_enabled=True,
            moe_ve_enabled=False,
            value_embeds_layers=[1, 3],
        )
    )
    model.init_weights()
    adamw_optimizer, muon_optimizer = model.setup_optimizers()

    adam_param_ids = {
        id(param)
        for group in adamw_optimizer.param_groups
        for param in group["params"]
    }
    muon_param_ids = {
        id(param)
        for group in muon_optimizer.param_groups
        for param in group["params"]
    }
    value_table_param_ids = {
        id(param)
        for param in model.parameters()
        if getattr(param, "value_embedding_is_table", False)
    }

    assert value_table_param_ids
    assert value_table_param_ids <= adam_param_ids
    assert value_table_param_ids.isdisjoint(muon_param_ids)
    assert model.transformer.h[1].attention.ve_gate is not None
    assert model.transformer.h[0].attention.ve_gate is None

    idx = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    loss, aux_loss = model(idx, targets, return_aux_loss=True)

    assert torch.isfinite(loss)
    assert torch.isfinite(aux_loss)


def test_weight_tying_reduces_125m_recipe_to_expected_scale():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=32768,
        n_layer=30,
        n_head=9,
        n_kv_head=3,
        n_embd=576,
        weight_tying=True,
        stem_layers=[],
    )
    with torch.device("meta"):
        model = StemGPT350M(cfg)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params == 125_077_824
