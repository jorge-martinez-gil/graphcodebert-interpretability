"""Tests for report formatting, HTML highlighting and utilities (model-free)."""
import os

import numpy as np
import pytest

from graphcodebert_interpretability import (
    generate_report,
    highlight_html,
    merge_subwords,
    pairwise_rows,
    set_seed,
    to_latex,
    to_markdown,
)
from graphcodebert_interpretability.similarity import SimilarityResult


# --- report formatting ------------------------------------------------------
def _rows():
    labels = ["A", "B", "C"]
    matrices = {
        "GraphCodeBERT": np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]),
        "AST": np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.4], [0.1, 0.4, 1.0]]),
    }
    return pairwise_rows(labels, matrices)


def test_pairwise_rows_count_and_keys():
    rows = _rows()
    assert len(rows) == 3  # C(3,2)
    assert rows[0]["pair"] == "A vs B"
    assert "GraphCodeBERT" in rows[0] and "AST" in rows[0]
    assert rows[0]["GraphCodeBERT"] == pytest.approx(0.5)


def test_to_markdown_has_header_and_rows():
    md = to_markdown(_rows())
    lines = md.splitlines()
    assert lines[0].startswith("| Pair |")
    assert "A vs B" in md
    assert len(lines) == 2 + 3  # header + sep + 3 pairs


def test_to_latex_is_booktabs():
    tex = to_latex(_rows(), caption="Cap", label="tab:x")
    assert r"\begin{table}" in tex
    assert r"\toprule" in tex and r"\bottomrule" in tex
    assert r"\caption{Cap}" in tex


def test_empty_rows_format_to_empty_string():
    assert to_markdown([]) == ""
    assert to_latex([]) == ""


def test_generate_report_writes_files(tmp_path):
    paths = generate_report(
        dataset="sorting", outdir=str(tmp_path), include_neural=False
    )
    for key in ("markdown", "latex", "figure"):
        assert os.path.exists(paths[key])
        assert os.path.getsize(paths[key]) > 0


# --- HTML highlighting ------------------------------------------------------
def test_highlight_html_contains_score_and_escapes():
    result = SimilarityResult(
        score=0.87,
        tokens_a=["Ġif", "Ġa", "Ġ<", "Ġb"],
        tokens_b=["Ġif", "Ġa", "Ġ<", "Ġb"],
        matrix=np.array([[0.9, 0.1, 0.1, 0.1]] * 4),
        name_a="A",
        name_b="B",
    )
    html = highlight_html(result)
    assert "0.87" in html
    assert "&lt;" in html  # the '<' token must be escaped
    assert "<span" in html


# --- utils ------------------------------------------------------------------
def test_set_seed_makes_numpy_reproducible():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_merge_subwords_merges_and_averages():
    tokens = ["Ġdef", "Ġfoo", "bar"]
    values = [1.0, 2.0, 4.0]
    words, merged = merge_subwords(tokens, values)
    assert words == ["def", "foobar"]
    assert merged == pytest.approx([1.0, 3.0])  # (2+4)/2 == 3


def test_merge_subwords_length_mismatch_raises():
    with pytest.raises(ValueError):
        merge_subwords(["a"], [1.0, 2.0])
