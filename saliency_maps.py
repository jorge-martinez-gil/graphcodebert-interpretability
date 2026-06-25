# -*- coding: utf-8 -*-
"""Gradient saliency maps (paper reproduction script)."""
import itertools
import os

from graphcodebert_interpretability import load_dataset, load_model, saliency, set_seed
from graphcodebert_interpretability.visualize import plot_saliency_pair

OUTPUT_DIR = "saliency_maps"

if __name__ == "__main__":
    set_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    snippets = load_dataset("sorting")
    model, tokenizer = load_model()
    for name_a, name_b in itertools.combinations(snippets, 2):
        sal_a = saliency(snippets[name_a], name=name_a, model=model, tokenizer=tokenizer)
        sal_b = saliency(snippets[name_b], name=name_b, model=model, tokenizer=tokenizer)
        out = os.path.join(OUTPUT_DIR, f"{name_a}_vs_{name_b}_saliency_map.png")
        plot_saliency_pair(
            sal_a.words, sal_a.normalized(), sal_b.words, sal_b.normalized(),
            name_a, name_b, out_path=out,
        )
