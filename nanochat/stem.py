"""STEM layer selection and feed-forward width helpers."""

from __future__ import annotations

from typing import List, Optional



def compute_stem_hidden_dim(
    *,
    dim: int,
    multiple_of: int,
    ffn_dim_multiplier: Optional[float],
) -> int:
    hidden_dim = int(2 * (4 * dim) / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def resolve_stem_layers(n_layer: int, stem_layers: Optional[List[int]] = None) -> List[int]:
    if stem_layers is None:
        # Paper-style default: replace one third of FFNs at uniform intervals.
        start = 1 if n_layer > 1 else 0
        resolved = list(range(start, n_layer, 3))
        if not resolved:
            resolved = [n_layer - 1]
    else:
        resolved = sorted(set(int(layer_id) for layer_id in stem_layers))
    resolved = [layer_id for layer_id in resolved if 0 <= layer_id < n_layer]
    return resolved
