# -*- coding: utf-8 -*-
"""Utility helpers: reproducible seeding, device selection, and subword token cleaning.

These helpers intentionally avoid importing heavy dependencies (``torch``,
``transformers``) at module import time so that the lightweight parts of the
library remain usable, and testable, in environments where the deep-learning
stack is not installed.
"""
from __future__ import annotations

import os
import random
from typing import List, Sequence, Tuple

import numpy as np

__all__ = [
    "set_seed",
    "resolve_device",
    "clean_token",
    "merge_subwords",
]

# Byte-level BPE markers used by the RoBERTa / GraphCodeBERT tokenizer.
SPACE_MARKER = "Ġ"  # "Ġ" -> start-of-word marker
NEWLINE_MARKER = "Ċ"  # "Ċ" -> newline marker


def set_seed(seed: int = 42) -> int:
    """Seed Python, NumPy and (if available) PyTorch for reproducible runs.

    Parameters
    ----------
    seed:
        The integer seed to apply. Defaults to ``42`` to match the values used
        throughout the accompanying paper.

    Returns
    -------
    int
        The seed that was applied, for convenient logging.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is an optional, heavy dependency.
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - exercised only without torch
        pass
    return seed


def resolve_device(device: str | None = None) -> str:
    """Return a concrete device string (``"cpu"`` or ``"cuda"``).

    When ``device`` is ``None`` the best available device is selected.
    """
    if device is not None:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - exercised only without torch
        return "cpu"


def clean_token(token: str) -> str:
    """Strip byte-level BPE markers from a single tokenizer token."""
    return token.replace(SPACE_MARKER, "").replace(NEWLINE_MARKER, "")


def merge_subwords(
    tokens: Sequence[str], values: Sequence[float]
) -> Tuple[List[str], List[float]]:
    """Merge byte-level BPE subword tokens into whole words.

    Subword fragments belonging to the same word are concatenated and their
    associated scalar ``values`` (e.g. saliency scores) are averaged. A new word
    begins whenever a token carries the start-of-word marker.

    Parameters
    ----------
    tokens:
        Raw tokenizer tokens (may contain ``Ġ``/``Ċ`` markers).
    values:
        A per-token scalar aligned with ``tokens``.

    Returns
    -------
    (words, merged_values)
        Whole words and their averaged values.
    """
    if len(tokens) != len(values):
        raise ValueError(
            f"tokens and values must be the same length, got "
            f"{len(tokens)} and {len(values)}"
        )

    words: List[str] = []
    merged: List[float] = []
    current_word = ""
    current_sum = 0.0
    count = 0

    for token, value in zip(tokens, values):
        if token.startswith(SPACE_MARKER):
            if current_word:
                words.append(current_word)
                merged.append(current_sum / count)
            current_word = clean_token(token)
            current_sum = float(value)
            count = 1
        else:
            current_word += clean_token(token)
            current_sum += float(value)
            count += 1

    if current_word:
        words.append(current_word)
        merged.append(current_sum / count)

    return words, merged
