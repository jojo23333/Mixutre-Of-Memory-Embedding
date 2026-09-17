"""Load and validate the released JSON slot maps."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch


def visible_token_text(token_str: str) -> str:
    if token_str == "":
        return "<empty>"
    pieces = []
    for ch in token_str:
        code = ord(ch)
        if ch == " ":
            pieces.append("[sp]")
        elif ch == "\n":
            pieces.append("[nl]")
        elif ch == "\r":
            pieces.append("[cr]")
        elif ch == "\t":
            pieces.append("[tab]")
        elif code < 32 or code == 127:
            pieces.append(f"[0x{code:02x}]")
        else:
            pieces.append(ch)
    return "".join(pieces)


def _normalize_histogram(histogram: dict[int | str, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(histogram.items(), key=lambda item: int(item[0]))}








def _validate_runtime_sizes(
    *,
    slot_map: torch.Tensor,
    metadata: dict[str, object],
    expected_vocab_size: int | None,
    runtime_slot_vocab_size: int | None,
    expected_slot_factor: int | None,
) -> None:
    if expected_vocab_size is not None and slot_map.numel() != expected_vocab_size:
        raise ValueError(
            f"slot_map vocab_size mismatch: file has {slot_map.numel()} tokens, expected {expected_vocab_size}"
        )
    file_slot_vocab_size = metadata.get("slot_vocab_size")
    if runtime_slot_vocab_size is not None:
        if slot_map.numel() == 0:
            raise ValueError("slot_map is empty")
        if slot_map.min().item() < 0 or slot_map.max().item() >= runtime_slot_vocab_size:
            raise ValueError("slot_map values must lie in [0, runtime_slot_vocab_size)")
        if file_slot_vocab_size is not None and int(file_slot_vocab_size) > runtime_slot_vocab_size:
            raise ValueError(
                f"slot_vocab_size mismatch: file declares {file_slot_vocab_size}, runtime has {runtime_slot_vocab_size}"
            )
    if expected_slot_factor is not None:
        file_slot_factor = metadata.get("slot_factor")
        if file_slot_factor is not None and int(file_slot_factor) != expected_slot_factor:
            raise ValueError(
                f"slot_factor mismatch: file declares {file_slot_factor}, expected {expected_slot_factor}"
            )




def _materialize_json_slot_map(
    document: dict[str, object],
    *,
    tokenizer=None,
    expected_vocab_size: int | None = None,
    runtime_slot_vocab_size: int | None = None,
    expected_slot_factor: int | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    if document.get("format") != "nanochat_slot_map_v1":
        raise ValueError(f"Unsupported slot-map format: {document.get('format')}")
    vocab_size = int(document["vocab_size"])
    slot_vocab_size = int(document["slot_vocab_size"])
    slots = document["slots"]
    if len(slots) != slot_vocab_size:
        raise ValueError(f"slot count mismatch: found {len(slots)} slots, expected {slot_vocab_size}")

    slot_map = torch.full((vocab_size,), -1, dtype=torch.long)
    seen = torch.zeros(vocab_size, dtype=torch.bool)
    slot_size_histogram: dict[int, int] = {}

    tokenizer_info = document.get("tokenizer", {})
    if tokenizer is not None:
        if int(tokenizer_info.get("vocab_size", tokenizer.get_vocab_size())) != tokenizer.get_vocab_size():
            raise ValueError("slot-map tokenizer vocab_size does not match the active tokenizer")
        if int(tokenizer_info.get("bos_token_id", tokenizer.get_bos_token_id())) != tokenizer.get_bos_token_id():
            raise ValueError("slot-map tokenizer bos_token_id does not match the active tokenizer")

    for expected_slot_id, slot in enumerate(slots):
        slot_id = int(slot["slot_id"])
        if slot_id != expected_slot_id:
            raise ValueError(f"slot order mismatch: expected slot_id={expected_slot_id}, found {slot_id}")
        token_ids = [int(token_id) for token_id in slot["token_ids"]]
        token_texts = slot.get("token_texts", [])
        token_texts_visible = slot.get("token_texts_visible", [])
        sample_counts = slot.get("sample_counts")
        if token_texts and len(token_texts) != len(token_ids):
            raise ValueError(f"slot_id={slot_id} token_text length mismatch")
        if token_texts_visible and len(token_texts_visible) != len(token_ids):
            raise ValueError(f"slot_id={slot_id} token_texts_visible length mismatch")
        if sample_counts is not None and len(sample_counts) != len(token_ids):
            raise ValueError(f"slot_id={slot_id} sample_counts length mismatch")
        slot_size_histogram[len(token_ids)] = slot_size_histogram.get(len(token_ids), 0) + 1
        for index, token_id in enumerate(token_ids):
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(f"slot_id={slot_id} contains out-of-range token_id={token_id}")
            if seen[token_id]:
                raise ValueError(f"token_id={token_id} appears in multiple slots")
            seen[token_id] = True
            slot_map[token_id] = slot_id
            if tokenizer is not None:
                actual_text = tokenizer.id_to_token(token_id)
                if token_texts and token_texts[index] != actual_text:
                    raise ValueError(
                        f"slot_id={slot_id} token_id={token_id} text mismatch: {token_texts[index]!r} != {actual_text!r}"
                    )
                actual_visible = visible_token_text(actual_text)
                if token_texts_visible and token_texts_visible[index] != actual_visible:
                    raise ValueError(
                        "slot_id={slot_id} token_id={token_id} visible text mismatch".format(
                            slot_id=slot_id,
                            token_id=token_id,
                        )
                    )

    if not bool(torch.all(seen)):
        missing = int((~seen).sum().item())
        raise ValueError(f"slot-map does not cover the full vocabulary; missing {missing} token ids")

    metadata = {key: value for key, value in document.items() if key != "slots"}
    metadata["slot_size_histogram"] = _normalize_histogram(slot_size_histogram)
    _validate_runtime_sizes(
        slot_map=slot_map,
        metadata=metadata,
        expected_vocab_size=expected_vocab_size,
        runtime_slot_vocab_size=runtime_slot_vocab_size,
        expected_slot_factor=expected_slot_factor,
    )
    return slot_map, metadata


@lru_cache(maxsize=16)
def _load_slot_map_cached(path_str: str) -> tuple[torch.Tensor, dict[str, object], dict[str, object] | None]:
    path = Path(path_str)
    if path.suffix.lower() == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
        slot_map, metadata = _materialize_json_slot_map(document)
        return slot_map, metadata, document
    raise ValueError("Slot maps must use the released JSON format")


def load_slot_map_artifact(
    path: str | Path,
    *,
    expected_vocab_size: int | None = None,
    runtime_slot_vocab_size: int | None = None,
    expected_slot_factor: int | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    slot_map, metadata, _ = _load_slot_map_cached(str(Path(path)))
    _validate_runtime_sizes(
        slot_map=slot_map,
        metadata=metadata,
        expected_vocab_size=expected_vocab_size,
        runtime_slot_vocab_size=runtime_slot_vocab_size,
        expected_slot_factor=expected_slot_factor,
    )
    return slot_map, dict(metadata)


def validate_slot_map_with_tokenizer(
    path: str | Path,
    *,
    tokenizer,
    expected_vocab_size: int | None = None,
    runtime_slot_vocab_size: int | None = None,
    expected_slot_factor: int | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    slot_map, metadata, document = _load_slot_map_cached(str(Path(path)))
    if document is not None:
        slot_map, metadata = _materialize_json_slot_map(
            document,
            tokenizer=tokenizer,
            expected_vocab_size=expected_vocab_size,
            runtime_slot_vocab_size=runtime_slot_vocab_size,
            expected_slot_factor=expected_slot_factor,
        )
    else:
        _validate_runtime_sizes(
            slot_map=slot_map,
            metadata=metadata,
            expected_vocab_size=expected_vocab_size,
            runtime_slot_vocab_size=runtime_slot_vocab_size,
            expected_slot_factor=expected_slot_factor,
        )
    metadata = dict(metadata)
    metadata["tokenizer_validated"] = True
    return slot_map, metadata
