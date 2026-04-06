# Hippo Dependency MPRA

Dissecting cell-type-specific regulatory grammar between HepG2 and K562 using LentiMPRA + AlphaGenome models.

## Overview

Virtual MPRA screens on ~57k enhancer sequences with eigen-interaction decomposition of cross-cell-type DeepLIFT/SHAP attributions to identify shared vs. cell-type-specific regulatory modes and higher-order epistasis via SHAPIQ.

## Structure

```
models/                         Fine-tuned AlphaGenome models (K562, HepG2, WTC11)
data/                           Joint library sequences
eigen-interactions/             Submodule: EigenMap class for attribution decomposition
genomic_targets/
  scripts/validation/           Model validation
  scripts/2d_targeting/         Eigen decomposition & target selection
  scripts/3d_example/           Motif annotation & eigen visualization example
syntax_SHAPIQ/scripts/          SHAPIQ interaction tests
virtual_perturbations/scripts/  Perturbation screens on target library
```

## Scripts

### genomic_targets/

| Script | Purpose |
|--------|---------|
| `validation/validate_models.ipynb` | Validate 9 PyTorch models (3 cell types x 3 dropout) vs JAX baseline; Pearson r, Spearman rho, MSE |
| `2d_targeting/liver_blood_targets.ipynb` | Eigen-decompose full 57k library; EI_1/EI_2 eigenvector angles, polar histograms; identify shared vs differential mechanisms |
| `2d_targeting/hippo_target_selection.ipynb` | Filter 57k seqs by EI ratio + importance correlation + focus TF motifs (HNF, STAT, AP1, TEA) for experimental targets |
| `3d_example/eigen_interactions_filtering.ipynb` | Example: motif annotation with JASPAR, high-ratio vs low-ratio sequence logos |

### virtual_perturbations/

| Script | Purpose |
|--------|---------|
| `show_hippo_targets.ipynb` | Visualize 1,059-seq target library; HepG2 vs K562 predictions colored by EI_1 var x r; top candidates with attribution logos |
| `perturb_targets.ipynb` | Necessity/sufficiency tests on all targets (dinucleotide shuffle KO/KI, n_rep=30, max_order=3); per-motif and per-TF score distributions |

### syntax_SHAPIQ/

| Script | Purpose |
|--------|---------|
| `Hippo_nec_SHAPIQ.ipynb` | Necessity-mode SHAPIQ: k-SII for orders 1-4; context-aware decomposition with background/promoter/barcode players; comparison to KO scores |
| `Hippo_suf_SHAPIQ.ipynb` | Sufficiency-mode SHAPIQ: k-SII for orders 1-4; per-annotation cell type and condition violin plots |
