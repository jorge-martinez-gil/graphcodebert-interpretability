# -*- coding: utf-8 -*-
"""UMAP token-embedding projections (paper reproduction script)."""
import itertools
import os

from graphcodebert_interpretability import load_dataset, load_model, project_pair, set_seed
from graphcodebert_interpretability.visualize import scatter_projection

OUTPUT_DIR = "umap_pairwise_comparisons"

if __name__ == "__main__":
    set_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    snippets = load_dataset("sorting")
    model, tokenizer = load_model()
    for name_a, name_b in itertools.combinations(snippets, 2):
        result = project_pair(
            snippets[name_a], snippets[name_b], method="umap",
            name_a=name_a, name_b=name_b, model=model, tokenizer=tokenizer,
        )
        out = os.path.join(OUTPUT_DIR, f"{name_a}_vs_{name_b}_tokens_2d_umap.png")
        scatter_projection(result.coords, result.split_index, name_a, name_b, out_path=out)
