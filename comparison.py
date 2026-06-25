# -*- coding: utf-8 -*-
"""Token-level similarity HTML report (paper reproduction script)."""
from graphcodebert_interpretability import compare, highlight_html, set_seed

CODE_A = """
def process_numbers(numbers):
    even_numbers = []
    for num in numbers:
        if is_even(num):
            even_numbers.append(calculate_factorial(num))
    return even_numbers
"""

CODE_B = """
def filter_numbers(nums):
    odd_numbers = []
    for val in nums:
        if is_odd(val):
            odd_numbers.append(compute_factorial(val))
    return odd_numbers
"""

if __name__ == "__main__":
    set_seed(42)
    result = compare(CODE_A, CODE_B, name_a="Source Code 1", name_b="Source Code 2")
    with open("code_similarity.html", "w", encoding="utf-8") as handle:
        handle.write(highlight_html(result, similarity_threshold=0.8))
    print(f"Final similarity score: {result.score:.2f}")
