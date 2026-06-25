# -*- coding: utf-8 -*-
"""Ablation over embedding level and projection method (paper reproduction script)."""
import itertools
import os

from graphcodebert_interpretability import (
    cosine_similarity_matrix,
    load_dataset,
    load_model,
    project_pair,
    sequence_embedding,
    set_seed,
    slug,
)
from graphcodebert_interpretability.visualize import scatter_projection

OUTPUT_DIR = "ablation_study_results"
USE_TOKEN_EMBEDDINGS = True
METHOD = "pca"

if __name__ == "__main__":
    set_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    snippets = load_dataset("sorting")
    model, tokenizer = load_model()
    method_label = METHOD.upper() if METHOD == "pca" else "t-SNE"
    for name_a, name_b in itertools.combinations(snippets, 2):
        result = project_pair(
            snippets[name_a], snippets[name_b], method=METHOD,
            name_a=name_a, name_b=name_b, model=model, tokenizer=tokenizer,
        )
        out = os.path.join(OUTPUT_DIR, f"{slug(name_a)}_vs_{slug(name_b)}_{method_label}.png")
        scatter_projection(
            result.coords, result.split_index, name_a, name_b,
            title=f"{name_a} vs {name_b} ({method_label})", out_path=out,
        )
        emb_a = sequence_embedding(
            snippets[name_a], pooling="pooler", model=model, tokenizer=tokenizer
        )
        emb_b = sequence_embedding(
            snippets[name_b], pooling="pooler", model=model, tokenizer=tokenizer
        )
        similarity = cosine_similarity_matrix(emb_a, emb_b)[0, 0]
        print(f"Cosine similarity between {name_a} and {name_b}: {similarity:.4f}")
