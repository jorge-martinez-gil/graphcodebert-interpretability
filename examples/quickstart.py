# -*- coding: utf-8 -*-
"""Quickstart for the graphcodebert_interpretability library."""
import graphcodebert_interpretability as gcbi

gcbi.set_seed(42)

CODE_A = "def f(numbers):\n    return [factorial(n) for n in numbers if is_even(n)]\n"
CODE_B = "def g(nums):\n    return [factorial(v) for v in nums if is_odd(v)]\n"


def main() -> None:
    print("AST structural similarity :", round(gcbi.ast_similarity(CODE_A, CODE_B), 4))
    data = gcbi.load_dataset("sorting")
    print("Bundled dataset snippets  :", list(data))
    paths = gcbi.generate_report(
        dataset="sorting", outdir="benchmark_report", include_neural=False
    )
    print("Benchmark report written  :", paths["markdown"])
    try:
        result = gcbi.compare(CODE_A, CODE_B, name_a="A", name_b="B")
        print("GraphCodeBERT similarity  :", round(result.score, 4))
    except ImportError as exc:
        print(f"[skipped neural lenses] {exc}")


if __name__ == "__main__":
    main()
