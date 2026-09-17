"""
Thin wrapper for the bigram Engram GPT variant.

The actual Engram plugin logic lives in nanochat.gpt.GPT so it stays attached
to the base transformer implementation, just like the value-embedding plugins.
"""

from __future__ import annotations

from nanochat.gpt import GPT, GPTConfig, compute_bigram_hash, resolve_bigram_engram_layers


class BigramEngramGPT(GPT):
    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        if config.dense_ve_enabled or config.moe_ve_enabled:
            raise ValueError("bigram_engram_gpt does not support value embeddings")
        config.bigram_engram_enabled = True
        super().__init__(config, pad_vocab_size_to=pad_vocab_size_to)
