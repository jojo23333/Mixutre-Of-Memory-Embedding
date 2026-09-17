"""
Unified attention interface with SDPA as the default runtime backend.

Exports an FA3-compatible attention interface. Set `NANOCHAT_FLASH_IMPL=fa3` to require FA3 or `NANOCHAT_FLASH_IMPL=auto` to select it when supported; `sdpa` is the default.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper+ GPU)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 9:  # Hopper is sm90
            return None
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        module = get_kernel('varunneal/flash-attention-3').flash_attn_interface
        module_dir = Path(module.__file__).parent
        # The downloaded FA3 package uses absolute imports such as
        # `from flash_attn_config import CONFIG`, so its directory must be on
        # sys.path when torch.compile replays the graph.
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))
        return module
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for tests/debugging: set to 'fa3', 'sdpa', or None to defer to env.
_override_impl = None


def configured_flash_impl():
    """Return the requested attention backend: sdpa, fa3, or auto."""
    requested = _override_impl
    if requested is None:
        requested = os.environ.get("NANOCHAT_FLASH_IMPL", "sdpa")
    requested = requested.strip().lower()
    if requested == "":
        requested = "sdpa"
    if requested not in {"sdpa", "fa3", "auto"}:
        raise ValueError("NANOCHAT_FLASH_IMPL must be one of: sdpa, fa3, auto")
    return requested


def _use_fa3():
    """Determine whether to use FA3 based on the configured backend."""
    requested = configured_flash_impl()
    if requested == 'fa3':
        assert HAS_FA3, "Cannot select FA3: not available on this hardware"
        return True
    if requested == 'auto':
        return HAS_FA3
    return False


def active_flash_impl():
    """Return the backend actually used for attention calls."""
    return "fa3" if _use_fa3() else "sdpa"


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask
    device = q.device
    if Tq == Tk:
        # Causal + sliding window
        mask = torch.tril(torch.ones(Tq, Tk, device=device, dtype=torch.bool))
        if window > 0 and window < Tq:
            row_idx = torch.arange(Tq, device=device).unsqueeze(1)
            col_idx = torch.arange(Tk, device=device).unsqueeze(0)
            mask = mask & ((row_idx - col_idx) <= window)
    else:
        # Chunk inference: attend to prefix + causal within chunk
        prefix_len = Tk - Tq
        mask = torch.zeros(Tq, Tk, device=device, dtype=torch.bool)
        mask[:, :prefix_len] = True
        mask[:, prefix_len:] = torch.tril(torch.ones(Tq, Tq, device=device, dtype=torch.bool))

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if _use_fa3():
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if _use_fa3():
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (FA3-compatible API backed by SDPA or FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
