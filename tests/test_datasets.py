"""Tests for dataset loading utilities."""
import json

import pytest

from graphcodebert_interpretability import (
    SORTING_ALGORITHMS,
    list_datasets,
    load_dataset,
    load_jsonl,
    load_snippets_from_dir,
    slug,
)


def test_builtin_sorting_dataset():
    assert "sorting" in list_datasets()
    data = load_dataset("sorting")
    assert len(data) == 5
    assert "Bubble Sort" in data
    assert "def bubble_sort" in data["Bubble Sort"]


def test_load_dataset_returns_copy():
    data = load_dataset("sorting")
    data["Bubble Sort"] = "mutated"
    assert SORTING_ALGORITHMS["Bubble Sort"] != "mutated"


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        load_dataset("does-not-exist")


def test_slug():
    assert slug("Bubble Sort") == "Bubble_Sort"
    assert slug("Quick Sort", sep="-") == "Quick-Sort"


def test_load_snippets_from_dir(tmp_path):
    (tmp_path / "a.py").write_text("def a(): pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def b(): pass\n", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("nope", encoding="utf-8")
    snippets = load_snippets_from_dir(str(tmp_path))
    assert set(snippets) == {"a", "b"}


def test_load_snippets_from_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError):
        load_snippets_from_dir(str(tmp_path))


def test_load_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"name": "x", "code": "print(1)"}) + "\n")
        handle.write("\n")  # blank line tolerated
        handle.write(json.dumps({"name": "y", "code": "print(2)"}) + "\n")
    snippets = load_jsonl(str(path))
    assert snippets == {"x": "print(1)", "y": "print(2)"}
