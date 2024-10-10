# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024] Augmenting the Interpretability of GraphCodeBERT for Code Similarity Tasks, arXiv preprint arXiv:2410.05275, 2024

@author: Jorge Martinez-Gil
"""

import torch
from transformers import RobertaTokenizer, RobertaModel
import nltk
import numpy as np

nltk.download('punkt')

# Load GraphCodeBERT model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")

def get_token_embeddings(code_snippet):
    tokens = tokenizer.tokenize(code_snippet)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([token_ids])
    
    with torch.no_grad():
        outputs = model(tokens_tensor)
    
    return tokens, outputs[0][0]  # Return tokens and their embeddings

def cosine_similarity(embedding1, embedding2):
    # Ensure embeddings are 2D before comparison
    if embedding1.dim() == 1:
        embedding1 = embedding1.unsqueeze(0)
    if embedding2.dim() == 1:
        embedding2 = embedding2.unsqueeze(0)

    return torch.nn.functional.cosine_similarity(embedding1, embedding2)

def highlight_similarities(code1, code2, similarity_threshold=0.8):
    tokens1, embeddings1 = get_token_embeddings(code1)
    tokens2, embeddings2 = get_token_embeddings(code2)
    
    similarities = np.zeros((len(tokens1), len(tokens2)))
    
    for i, embedding1 in enumerate(embeddings1):
        for j, embedding2 in enumerate(embeddings2):  # Corrected variable name here
            sim_score = cosine_similarity(embedding1, embedding2).item()
            similarities[i, j] = sim_score
    
    highlighted_code1 = []
    highlighted_code2 = []
    
    for i, token1 in enumerate(tokens1):
        max_sim = np.max(similarities[i])
        color_intensity = int(255 * max_sim) if max_sim > similarity_threshold else 0
        highlighted_code1.append((token1, color_intensity))
    
    for j, token2 in enumerate(tokens2):
        max_sim = np.max(similarities[:, j])
        color_intensity = int(255 * max_sim) if max_sim > similarity_threshold else 0
        highlighted_code2.append((token2, color_intensity))
    
    return highlighted_code1, highlighted_code2

def calculate_final_similarity(similarities):
    # Calculate the average of the maximum similarities for each token in both snippets
    max_similarities_1 = np.max(similarities, axis=1)  # Max similarity for each token in snippet 1
    max_similarities_2 = np.max(similarities, axis=0)  # Max similarity for each token in snippet 2
    
    final_similarity = (np.mean(max_similarities_1) + np.mean(max_similarities_2)) / 2
    return final_similarity

def generate_html(code1, code2, highlighted_code1, highlighted_code2, output_file, final_similarity):
    # Open the file with UTF-8 encoding to avoid UnicodeEncodeError
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("""
        <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    color: #333;
                    margin: 0;
                    padding: 20px;
                }
                h2 {
                    color: #444;
                    border-bottom: 2px solid #ddd;
                    padding-bottom: 10px;
                }
                pre {
                    background-color: #272822;
                    color: #f8f8f2;
                    padding: 20px;
                    border-radius: 5px;
                    overflow-x: auto;
                    font-size: 16px;
                    line-height: 1.5;
                }
                .token {
                    display: inline-block;
                    padding: 2px 5px;
                    border-radius: 3px;
                }
            </style>
        </head>
        <body>
        """)

        def get_color(intensity):
            """Return a color based on intensity, using a gradient from blue to green to yellow to red."""
            if intensity < 64:
                return f"rgb(173, 216, 230)"  # Light blue
            elif intensity < 128:
                return f"rgb(144, 238, 144)"  # Light green
            elif intensity < 192:
                return f"rgb(255, 255, 102)"  # Yellow
            else:
                return f"rgb(255, 69, 0)"  # Red

        f.write("<h2>Source Code 1:</h2><pre style='font-family:monospace;'>")
        
        previous_token = ''
        for token, intensity in highlighted_code1:
            if token == 'Ċ':
                f.write("<br>")
            elif token == 'Ġ':
                f.write(" ")
            else:
                if previous_token and not previous_token.endswith(('Ċ', 'Ġ')) and not previous_token.isspace():
                    f.write(" ")
                
                token = token.lstrip('Ġ')
                background_color = get_color(intensity)
                color = "#000" if intensity < 192 else "#fff"  # Adjust text color based on intensity
                token = token.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
                f.write(f"<span class='token' style='background-color:{background_color}; color:{color};'>{token}</span>")
            
            previous_token = token
        
        f.write("</pre>")
        
        f.write("<h2>Source Code 2:</h2><pre style='font-family:monospace;'>")
        
        previous_token = ''
        for token, intensity in highlighted_code2:
            if token == 'Ċ':
                f.write("<br>")
            elif token == 'Ġ':
                f.write(" ")
            else:
                if previous_token and not previous_token.endswith(('Ċ', 'Ġ')) and not previous_token.isspace():
                    f.write(" ")
                
                token = token.lstrip('Ġ')
                background_color = get_color(intensity)
                color = "#000" if intensity < 192 else "#fff"  # Adjust text color based on intensity
                token = token.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
                f.write(f"<span class='token' style='background-color:{background_color}; color:{color};'>{token}</span>")
            
            previous_token = token
        
        f.write("</pre>")
        f.write(f"<h2>Final Similarity Score: {final_similarity:.2f}</h2>")
        f.write("</body></html>")


if __name__ == "__main__":
    code_snippet_1 = """
def process_numbers(numbers):
    even_numbers = []
    for num in numbers:
        if is_even(num):
            even_numbers.append(calculate_factorial(num))
    return even_numbers
    """

    code_snippet_2 = """
def filter_numbers(nums):
    odd_numbers = []
    for val in nums:
        if is_odd(val):
            odd_numbers.append(compute_factorial(val))
    return odd_numbers
    """

    # Get the highlighted tokens and their similarities
    tokens1, embeddings1 = get_token_embeddings(code_snippet_1)
    tokens2, embeddings2 = get_token_embeddings(code_snippet_2)
    similarities = np.zeros((len(tokens1), len(tokens2)))
    
    for i, embedding1 in enumerate(embeddings1):
        for j, embedding2 in enumerate(embeddings2):
            similarities[i, j] = cosine_similarity(embedding1, embedding2).item()
    
    highlighted_code_1, highlighted_code_2 = highlight_similarities(code_snippet_1, code_snippet_2)
    
    # Calculate the final similarity score
    final_similarity = calculate_final_similarity(similarities)
    
    # Generate the HTML file with the final similarity score
    generate_html(code_snippet_1, code_snippet_2, highlighted_code_1, highlighted_code_2, "code_similarity.html", final_similarity)

