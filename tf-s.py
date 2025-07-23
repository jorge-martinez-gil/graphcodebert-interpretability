# -*- coding: utf-8 -*-
"""
[Martinez-Gil2025] Martinez-Gil, J. (2025). Augmenting the Interpretability of GraphCodeBERT for Code Similarity Tasks. International Journal of Software Engineering and Knowledge Engineering, 35(05), 657-678.

@author: Jorge Martinez-Gil
"""

import os
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

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

# Convert sorting algorithm code to TF-IDF features
vectorizer = TfidfVectorizer()
algorithm_names = list(sorting_algorithms.keys())
algorithm_codes = list(sorting_algorithms.values())
tfidf_matrix = vectorizer.fit_transform(algorithm_codes)

# Directory to save images
output_dir = "cosine_similarity_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Generate all possible pairs of sorting algorithms
algorithm_pairs = list(itertools.combinations(range(len(algorithm_names)), 2))

# Loop over each pair and compute cosine similarity
for idx1, idx2 in algorithm_pairs:
    algo1_name = algorithm_names[idx1]
    algo2_name = algorithm_names[idx2]

    # Compute cosine similarity between the two algorithms
    similarity = cosine_similarity(tfidf_matrix[idx1], tfidf_matrix[idx2])[0, 0]

    # Plotting similarity
    plt.figure(figsize=(6, 6), dpi=150)
    plt.title(f"Cosine Similarity: {algo1_name} vs {algo2_name}")
    plt.bar(["Similarity"], [similarity], color='blue')
    plt.ylim(0, 1)
    plt.ylabel("Cosine Similarity")
    plt.tight_layout()

    # Save the figure as a high-quality PNG file
    output_file = os.path.join(output_dir, f"{algo1_name}_vs_{algo2_name}_similarity.png")
    plt.savefig(output_file, format='png', dpi=300)
    plt.close()

print("All cosine similarity visualizations have been generated.")
