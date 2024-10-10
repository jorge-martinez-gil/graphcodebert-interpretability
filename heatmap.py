# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024] Augmenting the Interpretability of GraphCodeBERT for Code Similarity Tasks, arXiv preprint arXiv:2410.05275, 2024

@author: Jorge Martinez-Gil
"""

import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Initialize GraphCodeBERT
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")

# Sample classical sorting algorithms in Python as strings
sorting_algorithms = {
    "bubble_sort": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
    """,
    
    "selection_sort": """
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
    """,

    "insertion_sort": """
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

    "merge_sort": """
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

    "quick_sort": """
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

# Function to get the embeddings for code
def get_embedding(code):
    inputs = tokenizer(code, return_tensors="pt", max_length=512, truncation=True, padding=True)
    outputs = model(**inputs)
    # Flatten the embedding to 1D
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

# Get embeddings for each sorting algorithm
embeddings = {name: get_embedding(code) for name, code in sorting_algorithms.items()}

# Calculate cosine similarities
similarities = cosine_similarity([embeddings[name] for name in sorting_algorithms])

# Convert to a DataFrame for better visualization
similarity_df = pd.DataFrame(similarities, index=sorting_algorithms.keys(), columns=sorting_algorithms.keys())

# Plotting the heatmap
plt.figure(figsize=(10, 8), dpi=300)  # Set DPI for high-quality image
sns.heatmap(similarity_df, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Similarity between Sorting Algorithms using GraphCodeBERT")

# Save the figure as a high-quality PNG image
plt.savefig("sorting_algorithms_similarity.png", format='png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()


