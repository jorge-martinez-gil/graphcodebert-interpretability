# -*- coding: utf-8 -*-
"""AST structural-similarity baseline (paper reproduction script)."""
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graphcodebert_interpretability import ast_similarity_matrix, load_dataset, slug

OUTPUT_DIR = "ast_tree_kernel_visualizations"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    snippets = load_dataset("sorting")
    labels, matrix = ast_similarity_matrix(snippets)
    index = {name: i for i, name in enumerate(labels)}
    for name_a, name_b in itertools.combinations(labels, 2):
        similarity = matrix[index[name_a], index[name_b]]
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.set_title(f"AST-Based Similarity: {name_a} vs {name_b}")
        ax.bar(["Similarity"], [similarity], color="green")
        ax.set_ylim(0, 1)
        ax.set_ylabel("AST-Based Similarity")
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"{slug(name_a)}_vs_{slug(name_b)}_ast_similarity.png")
        fig.savefig(out, format="png", dpi=300)
        plt.close(fig)
