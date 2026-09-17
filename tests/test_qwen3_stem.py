import torch

from nanochat.gpt import GPTConfig
from nanochat.muon import Muon
from nanochat.qwen3_0p5b_model import Qwen3MLP, Qwen3StemMLP, Qwen3_0p5B


def _make_config(**overrides):
    config = dict(
        sequence_len=16,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        head_dim=16,
        weight_tying=True,
        stem_layers=[1, 3],
        stem_embedding_dim=192,
    )
    config.update(overrides)
    return GPTConfig(**config)


def test_qwen3_stem_replaces_only_selected_swiglu_branches():
    model = Qwen3_0p5B(_make_config())

    assert model.stem_layers == [1, 3]
    assert len(model.stem_embeddings) == 2
    assert isinstance(model.transformer.h[0].mlp, Qwen3MLP)
    assert isinstance(model.transformer.h[1].mlp, Qwen3StemMLP)
    assert isinstance(model.transformer.h[2].mlp, Qwen3MLP)
    assert isinstance(model.transformer.h[3].mlp, Qwen3StemMLP)


def test_qwen3_stem_forward_backward_and_optimizer_partition():
    model = Qwen3_0p5B(_make_config())
    model.init_weights()
    adamw_optimizer, muon_optimizer = model.setup_optimizers(stem_embedding_lr=0.3)

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

    idx = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    loss = model(idx, targets)
    loss.backward()

    assert torch.isfinite(loss)
    for embedding in model.stem_embeddings:
        assert embedding.weight.grad is not None
        assert torch.isfinite(embedding.weight.grad).all()


def test_qwen3_half_stem_full_parameter_audit():
    cfg = GPTConfig(
        sequence_len=2048,
        vocab_size=32768,
        n_layer=28,
        n_head=16,
        n_kv_head=8,
        n_embd=1024,
        head_dim=128,
        weight_tying=True,
        stem_layers=list(range(1, 28, 2)),
        stem_embedding_dim=3072,
    )
    with torch.device("meta"):
        model = Qwen3_0p5B(cfg)

    stem_params = sum(param.numel() for param in model.stem_embeddings.parameters())
    total_params = sum(param.numel() for param in model.parameters())

    assert stem_params == 1_409_286_144
    assert total_params == 1_839_267_840
