"""Shared pytest fixtures and markers.

Tests that need the GraphCodeBERT weights (torch + transformers + a download)
are marked ``model`` and skipped automatically when torch is unavailable, so the
fast suite runs anywhere, including CI without the heavy stack.
"""
import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

requires_model = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch/transformers not installed (model-backed test).",
)


@pytest.fixture
def sorting_snippets():
    from graphcodebert_interpretability import load_dataset

    return load_dataset("sorting")
