# -*- coding: utf-8 -*-
"""TF-IDF lexical-similarity baseline (paper reproduction script)."""
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graphcodebert_interpretability import load_dataset, slug, tfidf_similarity_matrix

OUTPUT_DIR = "cosine_similarity_visualizations"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    snippets = load_dataset("sorting")
    labels, matrix = tfidf_similarity_matrix(snippets)
    index = {name: i for i, name in enumerate(labels)}
    for name_a, name_b in itertools.combinations(labels, 2):
        similarity = matrix[index[name_a], index[name_b]]
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.set_title(f"Cosine Similarity: {name_a} vs {name_b}")
        ax.bar(["Similarity"], [similarity], color="blue")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Cosine Similarity")
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"{slug(name_a)}_vs_{slug(name_b)}_similarity.png")
        fig.savefig(out, format="png", dpi=300)
        plt.close(fig)
