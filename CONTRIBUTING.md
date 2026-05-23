# Contributing

Thank you for your interest in improving this project. Contributions that improve **reproducibility**, **clarity**, and **research impact** are especially welcome.

## Ways to Contribute
- Report bugs in scripts, notebooks, or generated outputs.
- Request features (new analyses, visualizations, or usability improvements).
- Propose reproducibility enhancements (environment setup, deterministic runs, clearer instructions).

Please open a GitHub Issue first: <https://github.com/jorge-martinez-gil/graphcodebert-interpretability/issues>.

## Coding Style
- Follow **PEP 8**.
- Include clear docstrings for public functions and modules.
- Keep changes focused and aligned with the repository's interpretability scope.

## Reproducing Results Locally
From the repository root:

```bash
pip install -r requirements.txt
python comparison.py
python heatmap.py
python pca.py
python tsne.py
python pumap.py
python saliency_maps.py
python ablation.py
python ast-s.py
python tf-s.py
```

For full reproducibility details (runtime expectations, deterministic settings), see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
