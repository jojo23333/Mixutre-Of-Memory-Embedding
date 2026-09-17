import pytest
import torch

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import flash_attn, HAS_FA3
from nanochat.gpt import GPT, GPTConfig


@pytest.mark.skipif(not HAS_FA3, reason="requires Flash Attention 3 on Hopper")
def test_moe_value_embeddings_keep_fa3_qkv_dtypes_aligned():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=256,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=256,
        moe_ve_enabled=True,
        moe_ve_setting="0_2_4",
        moe_ve_gate_nl="softmax",
        moe_ve_router_input="hidden",
        moe_ve_slot_mapping="none",
        moe_ve_slot_vocab_size=256,
        value_embeds_layers=[0],
    )

    seen_dtypes = []
    orig_impl = fa_module._override_impl
    orig_flash = flash_attn.flash_attn_func

    def wrapped_flash(q, k, v, **kwargs):
        seen_dtypes.append((q.dtype, k.dtype, v.dtype))
        return orig_flash(q, k, v, **kwargs)

    fa_module._override_impl = "fa3"
    flash_attn.flash_attn_func = wrapped_flash
    try:
        with torch.device("cuda"):
            model = GPT(cfg)
        model = model.cuda()
        model.init_weights()
        idx = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_len), device="cuda")
        targets = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_len), device="cuda")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(idx, targets)
        assert torch.isfinite(loss)
    finally:
        flash_attn.flash_attn_func = orig_flash
        fa_module._override_impl = orig_impl

    assert seen_dtypes, "expected the FA3 attention path to run"
    for q_dtype, k_dtype, v_dtype in seen_dtypes:
        assert q_dtype == k_dtype == v_dtype == torch.bfloat16
