# Hippo Dependency MPRA

Dissecting cell-type-specific regulatory grammar between HepG2 and K562 using LentiMPRA + AlphaGenome models.

## Overview

Virtual MPRA screens on ~57k enhancer sequences with eigen-interaction decomposition of cross-cell-type DeepLIFT/SHAP attributions to identify shared vs. cell-type-specific regulatory modes and higher-order epistasis via SHAPIQ.

## Structure

```
models/                    Fine-tuned AlphaGenome models (K562, HepG2, WTC11)
data/                      Joint library sequences
eigen-interactions/        Submodule: eigen-decomposition of attributions
genomic_targets/
  scripts/                 Model validation, attribution pipeline, eigen-filtering
  results/                 DeepLIFT attributions, eigen-interactions results
syntax_SHAPIQ/
  scripts/                 Necessity/sufficiency SHAPIQ interaction tests
virtual_perturbations/     Combinatorial perturbation screens
```

## Key Analyses

- **Model validation**: Pearson r on held-out joint library (K562: 0.89, HepG2: 0.88, WTC11: 0.85)
- **Eigen decomposition**: Cross-cell-type covariance of attributions → motif syntax enrichment
- **Perturbation tests**: Dinucleotide-shuffle KO/KI to measure motif functional importance
- **SHAPIQ interactions**: Higher-order epistasis via Shapley Interaction Indices

## Dependencies

Python 3.9+, PyTorch, tensorflow/keras. See `Hippo_agft_venv/` for pre-built venv or install via `uv` (see CLAUDE.md).
