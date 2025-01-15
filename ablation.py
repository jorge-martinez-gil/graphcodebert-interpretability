# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024] Augmenting the Interpretability of GraphCodeBERT for Code Similarity Tasks, arXiv preprint arXiv:2410.05275, 2024

@author: Jorge Martinez-Gil
"""

import os
from transformers import RobertaTokenizer, RobertaModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.spatial.distance import cosine

# Initialize GraphCodeBERT
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")

# Define the classical sorting algorithms
sorting_algorithms = {
    "Bubble_Sort": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
    """,
    "Selection_Sort": """
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
    """,
    "Insertion_Sort": """
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
    "Merge_Sort": """
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
    "Quick_Sort": """
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
    """
}

# Function to get token embeddings
def get_token_embeddings(code):
    inputs = tokenizer(code, return_tensors="pt", max_length=512, truncation=True, padding=True)
    outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state.squeeze().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
    return token_embeddings, tokens

# Function to get pooled text embeddings
def get_text_embeddings(code):
    inputs = tokenizer(code, return_tensors="pt", max_length=512, truncation=True, padding=True)
    outputs = model(**inputs)
    text_embedding = outputs.pooler_output.squeeze().detach().numpy()  # Single vector for the text
    return text_embedding

# Directory to save images
output_dir = "ablation_study_results"
os.makedirs(output_dir, exist_ok=True)

# Generate all possible pairs of sorting algorithms
algorithm_pairs = list(itertools.combinations(sorting_algorithms.keys(), 2))

# Ablation options
use_token_embeddings = True  # Toggle between token embeddings and text embeddings
use_pca = True  # Toggle between PCA and t-SNE

# Loop over each pair and generate the visualizations
for (algo1_name, algo2_name) in algorithm_pairs:
    algo1_code = sorting_algorithms[algo1_name]
    algo2_code = sorting_algorithms[algo2_name]
    
    # Get embeddings for both algorithms
    if use_token_embeddings:
        algo1_embeddings, algo1_tokens = get_token_embeddings(algo1_code)
        algo2_embeddings, algo2_tokens = get_token_embeddings(algo2_code)
        all_embeddings = np.concatenate((algo1_embeddings, algo2_embeddings), axis=0)
        labels = ["Token"] * len(algo1_tokens) + ["Token"] * len(algo2_tokens)
    else:
        algo1_embedding = get_text_embeddings(algo1_code)
        algo2_embedding = get_text_embeddings(algo2_code)
        all_embeddings = np.array([algo1_embedding, algo2_embedding])
        labels = ["Text", "Text"]

    # Dimensionality reduction
    if use_pca:
        reducer = PCA(n_components=2)
        method_name = "PCA"
    else:
        reducer = TSNE(n_components=2, perplexity=30, n_iter=300)
        method_name = "t-SNE"

    embeddings_2d = reducer.fit_transform(all_embeddings)

    # Plotting
    plt.figure(figsize=(10, 8), dpi=300)

    if use_token_embeddings:
        plt.scatter(embeddings_2d[:len(algo1_tokens), 0],
                    embeddings_2d[:len(algo1_tokens), 1],
                    color='red', s=50, label=algo1_name, alpha=0.8)
        plt.scatter(embeddings_2d[len(algo1_tokens):, 0],
                    embeddings_2d[len(algo1_tokens):, 1],
                    color='blue', s=50, label=algo2_name, alpha=0.8)
    else:
        plt.scatter(embeddings_2d[:, 0],
                    embeddings_2d[:, 1],
                    color=['red', 'blue'], s=100, alpha=0.8)
    
    plt.title(f"{algo1_name} vs {algo2_name} ({method_name})")
    plt.grid(False)
    plt.legend()
    output_file = os.path.join(output_dir, f"{algo1_name}_vs_{algo2_name}_{method_name}.png")
    plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print cosine similarity for text embeddings
    if not use_token_embeddings:
        similarity = 1 - cosine(algo1_embedding, algo2_embedding)
        print(f"Cosine similarity between {algo1_name} and {algo2_name}: {similarity:.4f}")

print("Ablation study completed. Results saved.")