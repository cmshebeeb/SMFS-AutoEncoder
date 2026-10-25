# AC-SMFAE: Unsupervised Feature Selection Framework

An advanced unsupervised feature selection framework that integrates **Adaptive Sample-Graph Topology Construction, Feature Graph Regularization, L2,1-Norm Sparsity, and Contractive Autoencoder Penalty** to identify informative features in high-dimensional datasets.

## Overview

AC-SMFAE extends sparse manifold-based feature selection by combining sample-level and feature-level graph regularization with sparsity constraints and contractive representation learning.

The framework is evaluated on five benchmark datasets covering face-image clustering and cancer microarray analysis.

### Key Components

- **Adaptive Sample Graph:** Models relationships between samples to preserve local manifold structure.
- **Feature Graph Regularization:** Incorporates relationships among features to support structured feature selection.
- **L2,1-Norm Sparsity:** Encourages row-wise sparsity in feature weights to identify informative features.
- **Contractive Autoencoder Penalty:** Encourages robust representations by penalizing sensitivity to input perturbations.

## Repository Structure

```text
AC-SMFAE/
├── data/
│   └── preprocessed/
│       ├── ORL/
│       ├── Yale/
│       ├── Leukemia/
│       ├── Colon/
│       └── Lymphoma/
├── notebooks/
│   ├── 01_*.ipynb
│   ├── 02_*.ipynb
│   ├── ...
│   └── 11_*.ipynb
├── reports/
│   ├── accuracy_nmi.csv
│   ├── summary_tables/
│   └── comparison_plots/
└── README.md
```

*Note: The directory and notebook names above are illustrative. Adjust them to match the actual repository contents.*

## Datasets

| Dataset | Domain | Samples | Classes |
|---|---|---:|---:|
| ORL | Face Images | 400 | 40 |
| Yale | Face Images | 165 | 15 |
| Leukemia | Cancer Microarray | 72 | 2 |
| Colon | Cancer Microarray | 62 | 2 |
| Lymphoma | Cancer Microarray | 96 | 9 |

## Experimental Results

The following table summarizes the reported clustering results for the base-paper baseline, the base paper's proposed Model 2, and the proposed AC-SMFAE system.

**Metric format: ACC (%) / NMI (%)**

| Dataset | Base Paper Baseline | Base Paper Proposed Model 2 | Proposed AC-SMFAE | Key Observation |
|---|---:|---:|---:|---|
| ORL | 56.90 / 76.25 | 53.00 / 76.00 | **57.84 / 76.52** | Higher ACC and NMI than both reported comparisons |
| Yale | 38.00 / 33.00 | 49.00 / 53.00 | **48.18 / 55.63** | Higher NMI than Model 2, with slightly lower ACC |
| Leukemia | 66.67 / 4.02 | 77.00 / 19.00 | **73.26 / 15.63** | Improvement over the baseline, but below Model 2 |
| Colon | 56.45 / 0.99 | 57.00 / 29.00 | **59.35 / 1.09** | Highest ACC among the reported methods, but substantially lower NMI than Model 2 |
| Lymphoma | 45.00 / 51.00 | 57.00 / 71.00 | **59.01 / 66.72** | Highest ACC among the reported methods, but lower NMI than Model 2 |

**ACC:** Clustering Accuracy  
**NMI:** Normalized Mutual Information

### Key Findings

- **ORL:** AC-SMFAE achieves the highest reported ACC and NMI among the three methods.
- **Yale:** AC-SMFAE improves NMI over Model 2, while its ACC remains comparable.
- **Leukemia:** AC-SMFAE improves both metrics over the baseline, although Model 2 performs better.
- **Colon:** AC-SMFAE achieves the highest ACC, but its low NMI indicates that the clustering structure requires further investigation.
- **Lymphoma:** AC-SMFAE achieves higher ACC than the two reported comparisons, while Model 2 achieves higher NMI.

These results suggest that the proposed framework's performance varies across datasets and evaluation metrics. The differences should be interpreted in the context of dataset characteristics, feature-selection settings, clustering procedures, and experimental variability.

## Methodology

The proposed framework combines the following components:

1. **Data preprocessing:** Load and prepare the benchmark datasets using the corresponding preprocessing pipeline.
2. **Graph construction:** Build sample and feature graphs to represent relationships in the data.
3. **Representation learning:** Train an autoencoder with a contractive penalty to learn robust latent representations.
4. **Feature selection:** Apply L2,1-norm sparsity to encourage selection of informative features.
5. **Clustering evaluation:** Evaluate selected feature subsets using ACC and NMI.
6. **Comparison and analysis:** Compare results against the reported baseline and proposed model from the reference paper.

## Repository Contents

- `data/preprocessed/` — Preprocessed dataset files.
- `notebooks/` — Jupyter notebooks for implementation, experiments, and analysis.
- `reports/` — Summary tables, comparison plots, and exported results.
- `reports/accuracy_nmi.csv` — Consolidated ACC and NMI results, if included in the repository.

## Reproducibility

For reproducible comparisons, document the following for each experiment:

- Dataset source and preprocessing steps.
- Number of selected features.
- Model architecture and hyperparameters.
- Graph-construction settings.
- Random seeds and clustering configuration.
- Evaluation protocol and number of experimental runs.
- Mean performance and standard deviation, where applicable.

Ensure that baseline and proposed-model results use comparable evaluation protocols before drawing conclusions.

## Reference

Moslemi & Jamshidi (2025), *Unsupervised Feature Selection Using Sparse Manifold Learning*.

*Full publication details are pending verification.*

## Disclaimer

The numerical results in this README represent the currently reported experimental results. They should be independently checked against the original publication and experiment logs before formal publication or citation. Performance improvements alone do not establish the contribution of individual architectural components; controlled ablation studies and repeated experiments are needed to support such claims.

## License

Add the chosen license and its terms here before distributing the repository publicly.
