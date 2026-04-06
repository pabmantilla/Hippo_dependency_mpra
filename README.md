## Overview

Reading the genetic code is a highly chaotic, yet surprisingly robust process. Perturbations to key elements of the cis- and trans-regulatory code in model organisms have allowed independent definitions of epistasis, robustness, expressivity, and evolvability in reference to complex outcomes, or phenotypes. Deep learning has been instrumental to tasks in regulatory genomics by learning sequence elements predictive of function in a fixed trans-regulatory state. Using these models as virtual experimental platforms allows us to perturb in the same way as before — to understand known feature local contribution (necessity test — CRISPR KO) and global contribution (sufficiency test — CRISPR KD).

In this repo we apply a principled approach, EigenMaps, to characterize, focus, perturb, and decompose the cis-regulatory code from the MPRA joint library. We conclude with the decomposition of the 'defining' signaling pathways — independently explaining the differential dependence in the Hippo pathway, the surprisingly additive nature of regulatory code when promoting transcriptional initiation, and the combinatorial nature of TFs when chromatin remodeling.


## Hippo Dependency MPRA

MPRA of ~57k enhancer sequences in HepG2 (Liver Hepatocyte-Hepatoblastoma), K562 (Bone Marrow Myeloid-CML), and WTC11 (Skin Fibroblast-iPSC) lines, with eigen-interaction decomposition of cross-cell-type DeepLIFT/SHAP attributions to identify shared vs. cell-type-specific regulatory modes and higher-order epistasis via SHAPIQ.

**Key metrics:**
- EI_1 var × ρ: motif function similarity across cell types (+1 = same motifs same function, 0 = unrelated, -1 = same motifs different function)
- Necessity/sufficiency tests: dinucleotide-shuffle KO/KI on target library
- SHAPIQ k-SII: context-aware interaction decomposition (orders 1–4)

## Structure

```
models/                         Fine-tuned AlphaGenome models (K562, HepG2, WTC11)
data/                           Joint library sequences
eigen-interactions/             Submodule: EigenMap class for attribution decomposition
genomic_targets/
  scripts/validation/           Model validation
  scripts/2d_targeting/         Eigen decomposition & target selection
  scripts/3d_example/           Motif annotation & eigen visualization example
syntax_SHAPIQ/
  scripts/                      SHAPIQ interaction tests
  libraries/                    Target sequence libraries
virtual_perturbations/
  scripts/                      Perturbation screens on target library
```

## Scripts

### Validation of PyTorch vs. JAX AlphaGenome models
| `validation/validate_models.ipynb` | Validate 9 PyTorch models (3 cell types × 3 dropout rates) vs. JAX baseline |



### Characterizing MPRA library with EigenMaps (genomic_targets/)

| Script | Purpose |
|--------|---------|
| `3d_example/eigen_interactions_filtering.ipynb` | WTC11 cells can be described as linear combinations of liver cells |

### Isolate mechanistic space of joint library with liver-blood basis (genomic_targets/)

| Script | Purpose |
|--------|---------|
| `2d_targeting/hippo_target_selection.ipynb` | Focus 57k seqs by EI ratio + importance correlation to explore descriptive mechanisms |
| `2d_targeting/liver_blood_targets.ipynb` | Eigen-decompose full 57k library; EI_1/EI_2 eigenvector angles, polar histograms; identify shared vs. differential mechanisms |


### In-silico Perturbations (virtual_perturbations/)

| Script | Purpose |
|--------|---------|
| `show_hippo_targets.ipynb` | Visualize 1,059-seq target library; HepG2 vs. K562 predictions colored by EI_1 var × ρ; top candidates with attribution logos |
| `perturb_targets.ipynb` | Necessity/sufficiency tests on all targets (dinucleotide shuffle KO/KI, n_rep=30, max_order=3); per-motif and per-TF score distributions |

### In-silico Decomposition (syntax_SHAPIQ/)

| Script | Purpose |
|--------|---------|
| `Hippo_nec_SHAPIQ.ipynb` | Necessity-mode SHAPIQ: k-SII for orders 1–4; context-aware decomposition with background/promoter/barcode players; comparison to KO scores |
| `Hippo_suf_SHAPIQ.ipynb` | Sufficiency-mode SHAPIQ: k-SII for orders 1–4; per-annotation cell type and condition violin plots |

## Key Methods

**EigenMap decomposition:** Cross-cell-type attribution matrices are decomposed via eigendecomposition. The first eigenvector (EI_1) captures the dominant axis of variation in motif function across cell types; the second (EI_2) captures orthogonal variation. Sequences are scored on var × ρ (variance × Pearson correlation) to prioritize sequences where the same motifs play consistent functional roles (+1) vs. divergent roles (-1) across cell types.

**SHAPIQ:** Computes exact higher-order Shapley interaction indices via sampling. Necessity tests use sequences with shuffled dinucleotide backgrounds; sufficiency tests use the original background.

**Target library:** 1,059 sequences selected by high EI_1 ratio, correlated importance, and TF motif presence. Includes both HepG2-biased and K562-biased candidates for experimental validation.
