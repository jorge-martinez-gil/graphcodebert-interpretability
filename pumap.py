# -*- coding: utf-8 -*-
"""
[Martinez-Gil2025] Martinez-Gil, J. (2025). Augmenting the Interpretability of GraphCodeBERT for Code Similarity Tasks. International Journal of Software Engineering and Knowledge Engineering, 35(05), 657-678.

@author: Jorge Martinez-Gil
"""

import os
from transformers import RobertaTokenizer, RobertaModel
import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np
import itertools  # Import itertools module

# Initialize GraphCodeBERT
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")

# Define the classical sorting algorithms
sorting_algorithms = {
    "Bubble Sort": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
    """,

    "Selection Sort": """
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
    """,

    "Insertion Sort": """
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

    "Merge Sort": """
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

    "Quick Sort": """
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

# Function to get token embeddings for a code snippet
def get_token_embeddings(code):
    inputs = tokenizer(code, return_tensors="pt", max_length=512, truncation=True, padding=True)
    outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state.squeeze().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
    return token_embeddings, tokens

# Directory to save images
output_dir = "umap_pairwise_comparisons"
os.makedirs(output_dir, exist_ok=True)

# Generate all possible pairs of sorting algorithms
algorithm_pairs = list(itertools.combinations(sorting_algorithms.keys(), 2))

# Loop over each pair and generate the UMAP visualizations
for (algo1_name, algo2_name) in algorithm_pairs:
    algo1_code = sorting_algorithms[algo1_name]
    algo2_code = sorting_algorithms[algo2_name]
    
    # Get token embeddings for both algorithms
    algo1_embeddings, algo1_tokens = get_token_embeddings(algo1_code)
    algo2_embeddings, algo2_tokens = get_token_embeddings(algo2_code)
    
    # Combine embeddings
    all_embeddings = np.concatenate((algo1_embeddings, algo2_embeddings), axis=0)
    
    # Perform UMAP dimensionality reduction to 2D
    umap_reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=42)
    embeddings_2d = umap_reducer.fit_transform(all_embeddings)
    
    # Plotting the UMAP results for tokens
    plt.figure(figsize=(10, 8), dpi=300)

    # Scatter plot for the first algorithm tokens
    plt.scatter(embeddings_2d[:len(algo1_tokens), 0],
                embeddings_2d[:len(algo1_tokens), 1],
                color='red', s=50, label=algo1_name, alpha=0.8)

    # Scatter plot for the second algorithm tokens
    plt.scatter(embeddings_2d[len(algo1_tokens):, 0],
                embeddings_2d[len(algo1_tokens):, 1],
                color='blue', s=50, label=algo2_name, alpha=0.8)

    # Make the visualization more professional
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.grid(False)
    plt.legend()

    # Save the figure as a high-quality PNG file
    output_file = os.path.join(output_dir, f"{algo1_name}_vs_{algo2_name}_tokens_2d_umap.png")
    plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
    
    # Close the plot
    plt.close()

print("UMAP pairwise comparison visualizations for token-level embeddings have been generated.")


