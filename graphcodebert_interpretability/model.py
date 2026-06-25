# -*- coding: utf-8 -*-
"""Lazy, cached loading of the GraphCodeBERT model and tokenizer.

``torch`` and ``transformers`` are imported lazily inside :func:`load_model` so
that the rest of the library (datasets, baselines, projection math, report
formatting) can be imported and unit-tested without the heavy deep-learning
stack installed.
"""
from __future__ import annotations

from typing import Tuple

from .utils import resolve_device

__all__ = ["DEFAULT_MODEL", "load_model", "clear_cache"]

DEFAULT_MODEL = "microsoft/graphcodebert-base"

# Cache keyed by (model_name, device) so repeated calls reuse the same weights.
_CACHE: dict = {}


def load_model(
    model_name: str = DEFAULT_MODEL, device: str | None = None
) -> Tuple["object", "object"]:
    """Load (and cache) a GraphCodeBERT model and its tokenizer.

    Parameters
    ----------
    model_name:
        Any Hugging Face model identifier compatible with ``RobertaModel``.
        Defaults to ``microsoft/graphcodebert-base``. Other code models such as
        ``microsoft/codebert-base`` or ``microsoft/unixcoder-base`` also work,
        which makes cross-model interpretability comparisons possible.
    device:
        ``"cpu"``, ``"cuda"`` or ``None`` (auto-select the best available).

    Returns
    -------
    (model, tokenizer)
        The model is set to evaluation mode and moved to ``device``.
    """
    device = resolve_device(device)
    key = (model_name, device)
    if key in _CACHE:
        return _CACHE[key]

    try:
        from transformers import RobertaModel, RobertaTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise ImportError(
            "Loading a model requires the optional dependencies 'torch' and "
            "'transformers'. Install them with: pip install "
            "graphcodebert-interpretability[model]"
        ) from exc

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    _CACHE[key] = (model, tokenizer)
    return model, tokenizer


def clear_cache() -> None:
    """Drop all cached models (useful to free memory between experiments)."""
    _CACHE.clear()
