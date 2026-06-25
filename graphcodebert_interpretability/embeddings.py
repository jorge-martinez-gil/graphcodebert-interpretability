# -*- coding: utf-8 -*-
"""Extract token-level and sequence-level embeddings from GraphCodeBERT.

``torch`` is imported lazily inside the functions so this module can be imported
without the deep-learning stack present.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .model import DEFAULT_MODEL, load_model

__all__ = [
    "token_embeddings",
    "sequence_embedding",
    "embed",
    "MAX_LENGTH",
]

MAX_LENGTH = 512


def _ensure_model(model=None, tokenizer=None, model_name: str = DEFAULT_MODEL):
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_name)
    return model, tokenizer


def token_embeddings(
    code: str,
    model=None,
    tokenizer=None,
    model_name: str = DEFAULT_MODEL,
) -> Tuple[List[str], np.ndarray]:
    """Return per-token embeddings for a code snippet.

    Parameters
    ----------
    code:
        The source code to encode.
    model, tokenizer:
        Optionally supply a preloaded model/tokenizer to avoid repeated loads.
    model_name:
        Model identifier used when ``model``/``tokenizer`` are not supplied.

    Returns
    -------
    (tokens, embeddings)
        ``tokens`` is the list of tokenizer tokens; ``embeddings`` is a
        ``(num_tokens, hidden_size)`` array from the final hidden layer.
    """
    import torch

    model, tokenizer = _ensure_model(model, tokenizer, model_name)
    inputs = tokenizer(
        code,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
    return tokens, embeddings


def sequence_embedding(
    code: str,
    pooling: str = "mean",
    model=None,
    tokenizer=None,
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Return a single fixed-size embedding for a code snippet.

    Parameters
    ----------
    pooling:
        ``"mean"`` averages the token embeddings (default, matches the paper);
        ``"cls"`` uses the first token; ``"pooler"`` uses the model's pooled
        output.
    """
    import torch

    model, tokenizer = _ensure_model(model, tokenizer, model_name)
    inputs = tokenizer(
        code,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    if pooling == "mean":
        vector = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    elif pooling == "cls":
        vector = outputs.last_hidden_state[:, 0, :].squeeze(0)
    elif pooling == "pooler":
        vector = outputs.pooler_output.squeeze(0)
    else:  # pragma: no cover - simple guard
        raise ValueError(
            f"Unknown pooling {pooling!r}; use 'mean', 'cls' or 'pooler'."
        )
    return vector.cpu().numpy()


def embed(
    code: str,
    level: str = "sequence",
    pooling: str = "mean",
    model=None,
    tokenizer=None,
    model_name: str = DEFAULT_MODEL,
):
    """Convenience wrapper returning either token or sequence embeddings.

    ``level="sequence"`` returns a single vector; ``level="token"`` returns the
    ``(tokens, embeddings)`` pair.
    """
    if level == "sequence":
        return sequence_embedding(
            code, pooling=pooling, model=model, tokenizer=tokenizer,
            model_name=model_name,
        )
    if level == "token":
        return token_embeddings(
            code, model=model, tokenizer=tokenizer, model_name=model_name
        )
    raise ValueError(f"Unknown level {level!r}; use 'sequence' or 'token'.")
