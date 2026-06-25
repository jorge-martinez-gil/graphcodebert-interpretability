"""Tests for dimensionality reduction (model-free, on synthetic embeddings)."""
import importlib.util

import numpy as np
import pytest

from graphcodebert_interpretability import project

UMAP_AVAILABLE = importlib.util.find_spec("umap") is not None


@pytest.fixture
def embeddings():
    return np.random.RandomState(0).randn(20, 16)


@pytest.mark.parametrize("method", ["pca", "tsne"])
def test_projection_shape(embeddings, method):
    coords = project(embeddings, method=method)
    assert coords.shape == (20, 2)


def test_pca_is_deterministic(embeddings):
    a = project(embeddings, method="pca", random_state=42)
    b = project(embeddings, method="pca", random_state=42)
    assert np.allclose(a, b)


def test_tsne_is_deterministic(embeddings):
    a = project(embeddings, method="tsne", random_state=42)
    b = project(embeddings, method="tsne", random_state=42)
    assert np.allclose(a, b)


def test_unknown_method_raises(embeddings):
    with pytest.raises(ValueError):
        project(embeddings, method="not-a-method")


@pytest.mark.skipif(not UMAP_AVAILABLE, reason="umap-learn not installed")
def test_umap_shape(embeddings):
    coords = project(embeddings, method="umap")
    assert coords.shape == (20, 2)


@pytest.mark.skipif(UMAP_AVAILABLE, reason="umap-learn is installed")
def test_umap_missing_raises_helpful_error(embeddings):
    with pytest.raises(ImportError, match="umap-learn"):
        project(embeddings, method="umap")
