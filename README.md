# Augmenting the Interpretability of GraphCodeBERT for Code Similarity Tasks

This repository contains the code and resources for the paper *"Augmenting the Interpretability of GraphCodeBERT for Code Similarity Tasks."*

This research has been recently published in the *International Journal of Software Engineering and Knowledge Engineering* and is available at [https://doi.org/10.1142/S0218194025500160](https://doi.org/10.1142/S0218194025500160).

[![DOI](https://img.shields.io/badge/DOI-10.1142%2FS0218194025500160-blue)](https://doi.org/10.1142/S0218194025500160)
[![arXiv](https://img.shields.io/badge/arXiv-2410.05275-b31b1b.svg)](https://arxiv.org/abs/2410.05275)
![License](https://img.shields.io/badge/license-MIT-green) ![GraphCodeBERT](https://img.shields.io/badge/GraphCodeBERT-interpretability-brightgreen)

![Comparison between Bubble and Insertion Sort](/pca_pairwise_comparisons/Bubble_Sort_vs_Insertion_Sort_tokens_2d_pca.png)


## 🌍 Overview
The goal of this research is to enhance the transparency and interpretability of the code similarity assessments provided by GraphCodeBERT, allowing developers to better understand why certain code fragments are deemed similar.

GraphCodeBERT is a powerful model for assessing code similarity based on the semantics of code fragments. However, it can be difficult to understand the inner workings of these assessments. Our approach introduces new methods for visualizing these relationships, making the model's decisions more transparent.

## 📂 Features
- **Code Similarity Assessment**: Identify similar code fragments using GraphCodeBERT's transformer model.
- **Visualization of Similarity**: Graphical outputs such as heatmaps and similarity matrices help explain why two code fragments are similar.


## 🛠️ Installation
To get started with this project, you'll need Python 3.7 or higher. Clone this repository and install the necessary dependencies.

```bash
git clone https://github.com/jorge-martinez-gil/graphcodebert-interpretability.git
cd graphcodebert-interpretability
pip install -r requirements.txt
```

## 📈 Visualizations
The visualizations help you understand how code fragments relate to each other. For example:
- **PCA, t-SNE and UMAP Plots**: Visualize high-dimensional code embeddings in a lower-dimensional space.
- **Saliency Maps**: Show which parts of the code were most important for the model's decision.

## 📚 Reference

If you use this work, please cite:

```bibtex
@article{martinezgil2025augmenting,
      author = {Martinez-Gil, Jorge},
      title = {Augmenting the Interpretability of GraphCodeBERT for Code Similarity Tasks},
      journal = {International Journal of Software Engineering and Knowledge Engineering},
      pages = {1-22},
      year = {2025},
      doi = {10.1142/S0218194025500160},
      URL = {https://doi.org/10.1142/S0218194025500160}
}
```
  
## 📄 License
This project is licensed under the MIT License.
