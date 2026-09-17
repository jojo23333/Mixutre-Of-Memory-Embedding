
import torch
import pytest


from nanochat.bigram_engram_gpt import (
    BigramEngramGPT,
    compute_bigram_hash,
    resolve_bigram_engram_layers,
)
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig


def test_compute_bigram_hash_uses_reserved_index_and_prev_tokens():
    idx = torch.tensor([[5, 7, 11], [1, 2, 3]], dtype=torch.long)
    bigram_vocab_size = 101
    mod = bigram_vocab_size - 1

    hashes = compute_bigram_hash(idx, bigram_vocab_size)
    assert torch.equal(hashes[:, 0], torch.full((2,), mod, dtype=torch.long))
    assert hashes[0, 1].item() == ((36313 * 7) ^ (27191 * 5)) % mod
    assert hashes[0, 2].item() == ((36313 * 11) ^ (27191 * 7)) % mod

    prev_tokens = torch.tensor([13, 17], dtype=torch.long)
    first_hashes = compute_bigram_hash(idx[:, :1], bigram_vocab_size, prev_tokens=prev_tokens)
    assert first_hashes[0, 0].item() == ((36313 * 5) ^ (27191 * 13)) % mod
    assert first_hashes[1, 0].item() == ((36313 * 1) ^ (27191 * 17)) % mod


def test_bigram_engram_default_layers_match_d12_odd_schedule():
    assert resolve_bigram_engram_layers(12, []) == [1, 3, 5, 7, 9, 11]


def test_bigram_engram_dict_size_scales_hash_table():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        bigram_engram_vocab_factor=6,
    )
    model = BigramEngramGPT(cfg)
    assert model.bigram_engram_embedding.bigram_vocab_size == 384


def test_bigram_engram_init_matches_baseline():
    common = dict(
        sequence_len=16,
        vocab_size=64,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
    )
    baseline = GPT(GPTConfig(**common))
    engram = BigramEngramGPT(GPTConfig(**common))

    torch.manual_seed(123)
    baseline.init_weights()
    torch.manual_seed(123)
    engram.init_weights()

    idx = torch.randint(0, 64, (2, 16))
    with torch.inference_mode():
        baseline_logits = baseline(idx)
        engram_logits = engram(idx)
    torch.testing.assert_close(baseline_logits, engram_logits, rtol=0.0, atol=0.0)


def test_bigram_engram_rejects_value_embeddings():
    with pytest.raises(ValueError, match="does not support value embeddings"):
        BigramEngramGPT(
            GPTConfig(
                sequence_len=16,
                vocab_size=64,
                n_layer=4,
                n_head=2,
                n_kv_head=2,
                n_embd=128,
                moe_ve_enabled=True,
            )
        )


def test_bigram_engram_wrapper_matches_base_gpt_plugin():
    common = dict(
        sequence_len=16,
        vocab_size=64,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        bigram_engram_enabled=True,
        bigram_engram_layers=[1, 3],
    )
    base = GPT(GPTConfig(**common))
    wrapped = BigramEngramGPT(
        GPTConfig(
            sequence_len=16,
            vocab_size=64,
            n_layer=4,
            n_head=2,
            n_kv_head=2,
            n_embd=128,
            bigram_engram_layers=[1, 3],
        )
    )

    torch.manual_seed(123)
    base.init_weights()
    torch.manual_seed(123)
    wrapped.init_weights()

    idx = torch.randint(0, 64, (2, 16))
    with torch.inference_mode():
        base_logits = base(idx)
        wrapped_logits = wrapped(idx)
    torch.testing.assert_close(base_logits, wrapped_logits, rtol=0.0, atol=0.0)


def test_bigram_engram_kv_cache_matches_full_forward():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
    )
    model = BigramEngramGPT(cfg)
    torch.manual_seed(123)
    model.init_weights()
    model.eval()

    prompt = torch.tensor([[4, 9, 2, 7]], dtype=torch.long)
    next_tokens = torch.tensor([[11], [13]], dtype=torch.long)
    full = torch.cat((prompt.expand(2, -1), next_tokens), dim=1)

    with torch.inference_mode():
        full_logits = model(full)[:, -1, :]

        kv_cache_prefill = KVCache(
            batch_size=1,
            num_heads=cfg.n_kv_head,
            seq_len=cfg.sequence_len,
            head_dim=cfg.n_embd // cfg.n_head,
            num_layers=cfg.n_layer,
            device="cpu",
            dtype=torch.float32,
        )
        _ = model(prompt, kv_cache=kv_cache_prefill)

        kv_cache_decode = KVCache(
            batch_size=2,
            num_heads=cfg.n_kv_head,
            seq_len=cfg.sequence_len,
            head_dim=cfg.n_embd // cfg.n_head,
            num_layers=cfg.n_layer,
            device="cpu",
            dtype=torch.float32,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        decode_logits = model(next_tokens, kv_cache=kv_cache_decode)[:, -1, :]

    torch.testing.assert_close(decode_logits, full_logits, rtol=1e-5, atol=1e-7)


def test_bigram_engram_optimizer_group_uses_tuned_embedding_recipe():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=12,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
    )
    model = BigramEngramGPT(cfg)
    model.init_weights()
    adamw_optimizer, _ = model.setup_optimizers(
        embedding_lr=0.3,
        engram_embedding_lr=0.2,
        engram_embedding_adam_betas=(0.8, 0.995),
        engram_embedding_weight_decay=0.001,
    )

    target_param = model.bigram_engram_embedding.embedding.weight
    target_group = None
    for group in adamw_optimizer.param_groups:
        if any(param is target_param for param in group["params"]):
            target_group = group
            break

    assert target_group is not None
    dmodel_lr_scale = (cfg.n_embd / 768) ** -0.5
    assert target_group["lr"] == pytest.approx(0.2 * dmodel_lr_scale)
    assert target_group["betas"] == (0.8, 0.995)
    assert target_group["weight_decay"] == pytest.approx(0.001)
