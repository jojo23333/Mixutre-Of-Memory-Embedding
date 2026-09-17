import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.qwen3_0p5b_model import Qwen3_0p5B


def _count_params(cfg):
    with torch.device("meta"):
        model = GPT(cfg)
    return sum(p.numel() for p in model.parameters())


def _count_value_embedding_params(cfg):
    with torch.device("meta"):
        model = GPT(cfg)
    return sum(p.numel() for p in model.parameters() if getattr(p, "value_embedding_is_table", False))


def test_dense_value_embeddings_match_0_2_6_table_footprint():
    common = dict(
        sequence_len=2048,
        vocab_size=32768,
        n_layer=12,
        n_head=6,
        n_kv_head=6,
        n_embd=768,
        window_pattern="L",
        value_embeds_layers=[1, 3, 5, 7, 9, 11],
    )
    moe_cfg = GPTConfig(
        **common,
        moe_ve_enabled=True,
        moe_ve_setting="0_2_6",
        moe_ve_gate_nl="softmax",
        moe_ve_router_input="hidden",
        moe_ve_slot_mapping="none",
        moe_ve_slot_vocab_size=32768,
    )
    dense_cfg = GPTConfig(
        **common,
        dense_ve_enabled=True,
    )

    assert _count_value_embedding_params(moe_cfg) == _count_value_embedding_params(dense_cfg)
    assert _count_params(dense_cfg) < _count_params(moe_cfg)


def test_moe_network_shared_table_reuses_memory_without_sharing_routers():
    common = dict(
        sequence_len=16,
        vocab_size=64,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        value_embeds_layers=[1, 3],
        moe_ve_enabled=True,
        moe_ve_setting="0_1_2",
        moe_ve_gate_nl="sigmoid-norm",
        moe_ve_router_input="hidden",
        moe_ve_slot_mapping="none",
        moe_ve_slot_vocab_size=64,
    )
    default_model = GPT(GPTConfig(**common))
    shared_model = GPT(GPTConfig(**common, moe_ve_network_shared_table=True))

    assert _count_value_embedding_params(shared_model.config) * 2 == _count_value_embedding_params(default_model.config)
    assert shared_model.embed_value_moe.layers[0].routed_embedding is shared_model.embed_value_moe.layers[1].routed_embedding
    assert shared_model.embed_value_moe.layers[0].router is not shared_model.embed_value_moe.layers[1].router
    assert (
        shared_model.embed_value_moe.layers[0].router.router_bias.data_ptr()
        != shared_model.embed_value_moe.layers[1].router.router_bias.data_ptr()
    )


def _count_qwen_value_embedding_params(cfg):
    with torch.device("meta"):
        model = Qwen3_0p5B(cfg)
    return sum(p.numel() for p in model.parameters() if getattr(p, "value_embedding_is_table", False))


def test_qwen3_dense_value_embeddings_match_0_2_8_table_footprint():
    common = dict(
        sequence_len=2048,
        vocab_size=32768,
        n_layer=28,
        n_head=16,
        n_kv_head=8,
        n_embd=1024,
        head_dim=128,
        weight_tying=True,
        value_embeds_layers=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
    )
    moe_cfg = GPTConfig(
        **common,
        moe_ve_enabled=True,
        moe_ve_setting="0_2_8",
        moe_ve_gate_nl="sigmoid-norm",
        moe_ve_router_input="hidden",
        moe_ve_slot_mapping="none",
        moe_ve_slot_vocab_size=32768,
    )
    dense_cfg = GPTConfig(
        **common,
        dense_ve_enabled=True,
    )

    assert _count_qwen_value_embedding_params(moe_cfg) == _count_qwen_value_embedding_params(dense_cfg)
