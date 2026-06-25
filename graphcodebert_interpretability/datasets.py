# -*- coding: utf-8 -*-
"""Built-in and user-loadable code datasets.

This module is the *single source of truth* for the classical sorting-algorithm
snippets used throughout the accompanying paper. Previously these snippets were
copy-pasted into every analysis script; centralising them here removes that
duplication and lets users plug in their own corpora with the same loaders.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Dict, List

__all__ = [
    "SORTING_ALGORITHMS",
    "list_datasets",
    "load_dataset",
    "load_snippets_from_dir",
    "load_jsonl",
    "slug",
]

# ---------------------------------------------------------------------------
# Built-in dataset: classical sorting algorithms (display name -> source code)
# ---------------------------------------------------------------------------
SORTING_ALGORITHMS: "OrderedDict[str, str]" = OrderedDict(
    [
        (
            "Bubble Sort",
            """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
""",
        ),
        (
            "Selection Sort",
            """
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
""",
        ),
        (
            "Insertion Sort",
            """
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >=0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
""",
        ),
        (
            "Merge Sort",
            """
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
    return arr
""",
        ),
        (
            "Quick Sort",
            """
def partition(arr, low, high):
    i = (low-1)
    pivot = arr[high]

    for j in range(low, high):
        if arr[j] <= pivot:
            i = i+1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[high] = arr[high], arr[i+1]
    return (i+1)

def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi-1)
        quick_sort(arr, pi+1, high)
    return arr
""",
        ),
    ]
)

_BUILTIN: Dict[str, "OrderedDict[str, str]"] = {
    "sorting": SORTING_ALGORITHMS,
}


def list_datasets() -> List[str]:
    """Return the names of the built-in datasets."""
    return sorted(_BUILTIN)


def load_dataset(name: str = "sorting") -> "OrderedDict[str, str]":
    """Load a built-in dataset as an ordered ``{name: code}`` mapping.

    Parameters
    ----------
    name:
        The dataset key. See :func:`list_datasets`.
    """
    try:
        return OrderedDict(_BUILTIN[name])
    except KeyError as exc:  # pragma: no cover - simple guard
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {list_datasets()}"
        ) from exc


def load_snippets_from_dir(
    path: str, extensions: tuple = (".py",)
) -> "OrderedDict[str, str]":
    """Load every source file in a directory as ``{filename: code}``.

    Parameters
    ----------
    path:
        Directory containing source files.
    extensions:
        File extensions to include. Defaults to Python files.
    """
    snippets: "OrderedDict[str, str]" = OrderedDict()
    for filename in sorted(os.listdir(path)):
        if not filename.endswith(extensions):
            continue
        full = os.path.join(path, filename)
        if not os.path.isfile(full):
            continue
        with open(full, "r", encoding="utf-8") as handle:
            snippets[os.path.splitext(filename)[0]] = handle.read()
    if not snippets:
        raise ValueError(
            f"No files with extensions {extensions} found in {path!r}."
        )
    return snippets


def load_jsonl(
    path: str, name_key: str = "name", code_key: str = "code"
) -> "OrderedDict[str, str]":
    """Load snippets from a JSON Lines file.

    Each line must be a JSON object containing a name field and a code field.
    """
    snippets: "OrderedDict[str, str]" = OrderedDict()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                snippets[str(record[name_key])] = str(record[code_key])
            except KeyError as exc:  # pragma: no cover - simple guard
                raise KeyError(
                    f"Line {line_number} is missing key {exc}."
                ) from exc
    return snippets


def slug(name: str, sep: str = "_") -> str:
    """Convert a display name into a filesystem-friendly slug."""
    return sep.join(name.split())
