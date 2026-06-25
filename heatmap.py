# -*- coding: utf-8 -*-
"""Global pairwise similarity heatmap (paper reproduction script)."""
from graphcodebert_interpretability import load_dataset, set_seed, similarity_matrix
from graphcodebert_interpretability.visualize import plot_similarity_heatmap

if __name__ == "__main__":
    set_seed(42)
    snippets = load_dataset("sorting")
    labels, matrix = similarity_matrix(snippets)
    plot_similarity_heatmap(
        labels, matrix,
        title="Similarity between Sorting Algorithms using GraphCodeBERT",
        out_path="sorting_algorithms_similarity.png",
    )
