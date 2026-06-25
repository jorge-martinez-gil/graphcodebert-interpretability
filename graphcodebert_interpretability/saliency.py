# -*- coding: utf-8 -*-
"""Gradient-based saliency maps for GraphCodeBERT.

Saliency is computed as the magnitude of the gradient of a scalar objective
(the mean of the final hidden state) with respect to the input word embeddings,
summed over the embedding dimension. ``torch`` is imported lazily.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .model import DEFAULT_MODEL, load_model
from .utils import merge_subwords

__all__ = ["SaliencyResult", "saliency"]


@dataclass
class SaliencyResult:
    """Per-word saliency scores for a single snippet."""

    words: List[str]
    scores: np.ndarray = field(repr=False)
    raw_tokens: List[str] = field(default_factory=list, repr=False)
    raw_scores: np.ndarray = field(default=None, repr=False)
    name: str = "Snippet"

    def normalized(self) -> np.ndarray:
        """Return saliency scaled to ``[0, 1]`` by its maximum."""
        scores = np.asarray(self.scores, dtype=np.float64)
        peak = scores.max() if scores.size else 0.0
        if peak <= 0:
            return scores
        return scores / peak


def saliency(
    code: str,
    name: str = "Snippet",
    merge: bool = True,
    model=None,
    tokenizer=None,
    model_name: str = DEFAULT_MODEL,
) -> SaliencyResult:
    """Compute a gradient saliency map for a code snippet.

    Parameters
    ----------
    code:
        Source code to analyse (any language GraphCodeBERT tokenizes).
    merge:
        When ``True`` (default) subword fragments are merged into whole words
        and their saliencies averaged.

    Returns
    -------
    SaliencyResult
        Words (or raw tokens) and their saliency scores.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_name)

    inputs = tokenizer(
        code, return_tensors="pt", max_length=512, truncation=True, padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    inputs_embeds = model.embeddings.word_embeddings(inputs["input_ids"])
    inputs_embeds.retain_grad()

    outputs = model(
        inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"]
    )
    objective = outputs.last_hidden_state.mean()
    model.zero_grad()
    objective.backward()

    raw_scores = (
        inputs_embeds.grad.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
    )
    raw_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))

    if merge:
        words, merged = merge_subwords(raw_tokens, raw_scores.tolist())
        scores = np.asarray(merged, dtype=np.float64)
    else:
        words = raw_tokens
        scores = raw_scores

    return SaliencyResult(
        words=words,
        scores=scores,
        raw_tokens=raw_tokens,
        raw_scores=raw_scores,
        name=name,
    )
