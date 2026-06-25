# -*- coding: utf-8 -*-
"""Command-line interface for graphcodebert-interpretability.

Examples
--------
    gcbi compare a.py b.py --html out.html
    gcbi saliency a.py --out saliency.png
    gcbi project --dataset sorting --method pca --outdir figs/
    gcbi heatmap --dataset sorting --out heatmap.png
    gcbi report --dataset sorting --outdir benchmark_report
    gcbi reproduce --outdir .
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional

from . import __version__
from .datasets import load_dataset, load_snippets_from_dir, slug
from .model import DEFAULT_MODEL
from .utils import set_seed


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _resolve_snippets(args) -> Dict[str, str]:
    if getattr(args, "dir", None):
        return load_snippets_from_dir(args.dir)
    if getattr(args, "files", None):
        return {os.path.splitext(os.path.basename(f))[0]: _read(f) for f in args.files}
    return load_dataset(getattr(args, "dataset", "sorting"))


def _cmd_compare(args) -> int:
    from .similarity import compare, highlight_html

    result = compare(
        _read(args.code_a),
        _read(args.code_b),
        name_a=os.path.basename(args.code_a),
        name_b=os.path.basename(args.code_b),
        model_name=args.model,
    )
    print(f"Final similarity score: {result.score:.4f}")
    if args.html:
        with open(args.html, "w", encoding="utf-8") as handle:
            handle.write(highlight_html(result))
        print(f"Wrote HTML report: {args.html}")
    return 0


def _cmd_saliency(args) -> int:
    from .saliency import saliency
    from .visualize import plot_saliency

    result = saliency(
        _read(args.code), name=os.path.basename(args.code), model_name=args.model
    )
    out = args.out or "saliency.png"
    plot_saliency(result.words, result.normalized(), name=result.name, out_path=out)
    print(f"Wrote saliency map: {out}")
    return 0


def _cmd_project(args) -> int:
    import itertools

    from .projection import project_pair
    from .visualize import scatter_projection

    snippets = _resolve_snippets(args)
    os.makedirs(args.outdir, exist_ok=True)
    names = list(snippets)
    count = 0
    for a, b in itertools.combinations(names, 2):
        res = project_pair(
            snippets[a], snippets[b], method=args.method,
            name_a=a, name_b=b, model_name=args.model,
        )
        out = os.path.join(
            args.outdir, f"{slug(a)}_vs_{slug(b)}_tokens_2d_{args.method}.png"
        )
        scatter_projection(
            res.coords, res.split_index, a, b,
            title=f"{a} vs {b} ({args.method.upper()})", out_path=out,
        )
        count += 1
    print(f"Wrote {count} {args.method.upper()} projection(s) to {args.outdir}/")
    return 0


def _cmd_heatmap(args) -> int:
    from .similarity import similarity_matrix
    from .visualize import plot_similarity_heatmap

    snippets = _resolve_snippets(args)
    labels, matrix = similarity_matrix(snippets, model_name=args.model)
    out = args.out or "similarity_heatmap.png"
    plot_similarity_heatmap(
        labels, matrix,
        title="Code similarity (GraphCodeBERT)", out_path=out,
    )
    print(f"Wrote heatmap: {out}")
    return 0


def _cmd_report(args) -> int:
    from .report import generate_report

    snippets = None
    if getattr(args, "dir", None) or getattr(args, "files", None):
        snippets = _resolve_snippets(args)
    paths = generate_report(
        snippets=snippets,
        dataset=args.dataset,
        outdir=args.outdir,
        model_name=args.model,
        include_neural=not args.no_neural,
    )
    print("Report artefacts:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


def _cmd_reproduce(args) -> int:
    """Regenerate the paper's full figure set from the bundled dataset."""
    base = args.outdir
    for method in ("pca", "tsne", "umap"):
        sub = argparse.Namespace(
            dataset="sorting", files=None, dir=None, method=method,
            outdir=os.path.join(base, f"{method}_pairwise_comparisons"),
            model=args.model,
        )
        try:
            _cmd_project(sub)
        except ImportError as exc:
            print(f"[skip] {method}: {exc}")
    _cmd_heatmap(
        argparse.Namespace(
            dataset="sorting", files=None, dir=None,
            out=os.path.join(base, "sorting_algorithms_similarity.png"),
            model=args.model,
        )
    )
    _cmd_report(
        argparse.Namespace(
            dataset="sorting", files=None, dir=None,
            outdir=os.path.join(base, "benchmark_report"),
            model=args.model, no_neural=False,
        )
    )
    print("Reproduction complete.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gcbi",
        description="GraphCodeBERT interpretability toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"gcbi {__version__}")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_model_arg(p):
        p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id.")

    def add_corpus_args(p):
        group = p.add_mutually_exclusive_group()
        group.add_argument("--dataset", default="sorting", help="Built-in dataset.")
        group.add_argument("--dir", help="Directory of source files.")
        group.add_argument("--files", nargs="+", help="Explicit source files.")

    p_cmp = sub.add_parser("compare", help="Token-level similarity of two files.")
    p_cmp.add_argument("code_a")
    p_cmp.add_argument("code_b")
    p_cmp.add_argument("--html", help="Write an HTML highlight report here.")
    add_model_arg(p_cmp)
    p_cmp.set_defaults(func=_cmd_compare)

    p_sal = sub.add_parser("saliency", help="Gradient saliency map for one file.")
    p_sal.add_argument("code")
    p_sal.add_argument("--out", help="Output PNG path.")
    add_model_arg(p_sal)
    p_sal.set_defaults(func=_cmd_saliency)

    p_prj = sub.add_parser("project", help="2D projection of token embeddings.")
    add_corpus_args(p_prj)
    p_prj.add_argument(
        "--method", choices=("pca", "tsne", "umap"), default="pca"
    )
    p_prj.add_argument("--outdir", default="projections")
    add_model_arg(p_prj)
    p_prj.set_defaults(func=_cmd_project)

    p_hm = sub.add_parser("heatmap", help="Global similarity heatmap.")
    add_corpus_args(p_hm)
    p_hm.add_argument("--out", help="Output PNG path.")
    add_model_arg(p_hm)
    p_hm.set_defaults(func=_cmd_heatmap)

    p_rep = sub.add_parser("report", help="Benchmark report (MD + LaTeX + figure).")
    add_corpus_args(p_rep)
    p_rep.add_argument("--outdir", default="benchmark_report")
    p_rep.add_argument(
        "--no-neural", action="store_true",
        help="Skip the GraphCodeBERT lens (baselines only, no model needed).",
    )
    add_model_arg(p_rep)
    p_rep.set_defaults(func=_cmd_report)

    p_rpr = sub.add_parser("reproduce", help="Regenerate the paper figure set.")
    p_rpr.add_argument("--outdir", default=".")
    add_model_arg(p_rpr)
    p_rpr.set_defaults(func=_cmd_reproduce)

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    set_seed(args.seed)
    try:
        return args.func(args)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
