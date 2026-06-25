"""Model-backed integration tests.

These require torch + transformers and download GraphCodeBERT on first run, so
they are skipped automatically when torch is unavailable (e.g. in fast CI). Run
them explicitly with::

    pytest -m model
"""
import importlib.util

import numpy as np
import pytest

requires_model = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch/transformers not installed (model-backed test).",
)


@requires_model
def test_compare_runs_and_is_symmetric():
    import graphcodebert_interpretability as gcbi

    a = "def add(a, b):\n    return a + b\n"
    b = "def sum2(x, y):\n    return x + y\n"
    r1 = gcbi.compare(a, b)
    r2 = gcbi.compare(b, a)
    assert 0.0 <= r1.score <= 1.0
    assert r1.score == pytest.approx(r2.score, abs=1e-4)


@requires_model
def test_identical_code_scores_near_one():
    import graphcodebert_interpretability as gcbi

    code = "def f(x):\n    return x * 2\n"
    assert gcbi.compare(code, code).score == pytest.approx(1.0, abs=1e-3)


@requires_model
def test_saliency_shapes():
    import graphcodebert_interpretability as gcbi

    result = gcbi.saliency("def f(x):\n    return x + 1\n")
    assert len(result.words) == len(result.scores)
    assert np.all(np.isfinite(result.scores))


@requires_model
def test_similarity_matrix_diagonal(sorting_snippets):
    import graphcodebert_interpretability as gcbi

    labels, matrix = gcbi.similarity_matrix(sorting_snippets)
    assert matrix.shape == (len(labels), len(labels))
    assert np.allclose(np.diag(matrix), 1.0, atol=1e-3)
