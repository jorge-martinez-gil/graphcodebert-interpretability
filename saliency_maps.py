# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024] Augmenting the Interpretability of GraphCodeBERT for Code Similarity Tasks, arXiv preprint arXiv:2410.05275, 2024

@author: Jorge Martinez-Gil
"""

import torch
from transformers import RobertaTokenizer, RobertaModel
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools

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

# Function to get token embeddings and saliency scores
def get_saliency_map(code, model, tokenizer):
    inputs = tokenizer(code, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    # Get the embeddings for the input tokens
    inputs_embeds = model.embeddings.word_embeddings(inputs['input_ids'])
    inputs_embeds.retain_grad()  # Retain gradients for saliency computation
    
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=inputs['attention_mask'])
    
    # Generate saliency scores by computing the gradient of the embeddings
    loss = outputs.last_hidden_state.mean()  # Simplified objective
    loss.backward()

    # The gradient of the loss with respect to the input embeddings
    saliency = inputs_embeds.grad.abs().sum(dim=-1).squeeze().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())

    return tokens, saliency

# Helper function to clean up and join subword tokens and sum their saliency scores
def clean_and_join_tokens(tokens, saliency):
    words = []
    merged_saliency = []
    current_word = ""
    current_saliency = 0.0
    count = 0

    for i, token in enumerate(tokens):
        if token.startswith("Ġ"):  # Tokenizer-specific prefix for new words
            if current_word:  # Save the previous word
                words.append(current_word)
                merged_saliency.append(current_saliency / count)
            current_word = token[1:]  # Remove the prefix
            current_saliency = saliency[i]
            count = 1
        else:
            current_word += token
            current_saliency += saliency[i]
            count += 1

    if current_word:  # Add the last word
        words.append(current_word)
        merged_saliency.append(current_saliency / count)
    
    return words, merged_saliency

# Directory to save images
output_dir = "saliency_maps"
os.makedirs(output_dir, exist_ok=True)

# Generate all possible pairs of sorting algorithms
algorithm_pairs = list(itertools.combinations(sorting_algorithms.keys(), 2))

# Loop over each pair and generate the saliency maps
for (algo1_name, algo2_name) in algorithm_pairs:
    algo1_code = sorting_algorithms[algo1_name]
    algo2_code = sorting_algorithms[algo2_name]
    
    # Get saliency maps for both algorithms
    tokens1, saliency1 = get_saliency_map(algo1_code, model, tokenizer)
    tokens2, saliency2 = get_saliency_map(algo2_code, model, tokenizer)

    # Clean and join tokens, adjust saliency scores
    decoded_tokens1, adjusted_saliency1 = clean_and_join_tokens(tokens1, saliency1)
    decoded_tokens2, adjusted_saliency2 = clean_and_join_tokens(tokens2, saliency2)

    # Normalize saliency scores for better visualization
    adjusted_saliency1 = np.array(adjusted_saliency1) / np.max(adjusted_saliency1)
    adjusted_saliency2 = np.array(adjusted_saliency2) / np.max(adjusted_saliency2)

    # Plotting the saliency maps together
    plt.figure(figsize=(12, 4))

    # Plotting the saliency map for the first algorithm
    plt.bar(range(len(decoded_tokens1)), adjusted_saliency1, color='red', label=f"Saliency Map for {algo1_name}")

    # Plotting the saliency map for the second algorithm
    plt.bar(range(len(decoded_tokens2)), adjusted_saliency2, color='blue', label=f"Saliency Map for {algo2_name}")

    # Remove the X-axis labels
    plt.xticks(range(len(decoded_tokens1)), [''] * len(decoded_tokens1))

    plt.title(f"Saliency Maps: {algo1_name} vs {algo2_name}")
    plt.xlabel("Token")
    plt.ylabel("Saliency Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot as a PNG file
    output_file = os.path.join(output_dir, f"{algo1_name}_vs_{algo2_name}_saliency_map.png")
    plt.savefig(output_file, format='png', dpi=300)

    # Close the plot
    plt.close()

print("Saliency maps for pairwise algorithm comparisons have been generated.")

