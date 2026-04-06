## Overview

Reading the genetic code is a highly chaotic, yet surprisingly robust process. Perturbations to key elements of the cis- and trans-regulatory code in model organisms have allowed independent definitions of epistasis, robustness, expressivity, and evolvability in reference to complex outcomes, or phenotypes. Deep learning models learn sequence elements predictive of function in a fixed trans-regulatory state. Using these models as virtual experimental platforms allows us to target known feature's (eg. tfbs) local contribution (necessity test — CRISPR KO) and global contribution (sufficiency test — CRISPR KI).

In this repo we apply a principled approach, EigenMaps, to characterize, focus, perturb, and decompose the cis-regulatory code from the MPRA joint library. Sequences are mechanistically classified into same-same (+1), same-diff (-1), and diff-diff (0) mechanism classes via EI_1 var × ρ. Necessity/sufficiency perturbation tests functionally assess motifs beyond sequence conservation or TFMoDISco+TOMTOM alone. We conclude with the decomposition of the 'defining' signaling pathways — independently explaining the differential dependence in the Hippo pathway, the surprisingly additive nature of regulatory code when promoting transcriptional initiation, and the combinatorial nature of TFs when chromatin remodeling.


## Hippo Dependency MPRA

LentiMPRA of ~57k enhancer sequences in HepG2 (Liver Hepatocyte-Hepatoblastoma), K562 (Bone Marrow Myeloid-CML), and WTC11 (Skin Fibroblast-iPSC) lines, with AlphaGenome fine-tuned models and eigen-interaction decomposition of cross-cell-type DeepLIFT/SHAP attributions to identify shared vs. cell-type-specific regulatory modes.

**Key metrics:**
- EI_1 var × ρ: motif function similarity across cell types (+1 = same motifs same function, 0 = unrelated, -1 = same motifs different function)
- Necessity/sufficiency tests: marginalized dinucleotide-shuffle KO/KI on target library
- SHAPIQ k-SII: context-aware interaction decomposition (orders 1–4), 2-player sufficiency SHAP decomposing motif syntax vs context

## Structure

```
data/                           Joint MPRA library (57k seqs, HepG2+K562+WTC11)
models/                         Fine-tuned AlphaGenome models (3 cell types × 3 dropout rates)
pytorch_base_model/             Base model checkpoint
eigen-interactions/             Submodule: EigenMap class for attribution decomposition (~2800 lines)
genomic_targets/
  scripts/validation/           Model validation
  scripts/2d_targeting/         Eigen decomposition & target selection
  scripts/3d_example/           3-cell-type decomposition
syntax_SHAPIQ/
  scripts/                      Necessity & sufficiency SHAPIQ interaction tests
virtual_perturbations/
  scripts/                      Perturbation screens on target library
  libraries/                    Pickled target libraries
SEAM_target_spaces/
  scripts/                      Mutagenesis, attributions, SEAM clustering on targets
Motif_context_swap/             Context swap experiments (WIP)
```

## Scripts

### Validation of PyTorch vs. JAX AlphaGenome models
| Script | Purpose |
|--------|---------|
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
| `Hippo_suf_SHAPIQ.ipynb` | Sufficiency-mode SHAPIQ: k-SII for orders 1–4; 2-player sufficiency SHAP decomposing motif syntax vs context; per-annotation cell type and condition violin plots |

### Decompose target spaces with SEAM (SEAM_target_spaces/)

| Script | Purpose |
|--------|---------|
| `SEAM_mutagenisis.py` | Generate mutagenesis libraries for 1,059 target sequences |
| `SEAM_attr.py` | AlphaGenome attributions on mutagenesis libraries |
| `SEAM_explainer.py` | SEAM clustering on attribution landscapes |

### Motif Context Swap (Motif_context_swap/)

Swapping motif syntax and backgrounds across cell lines, activity bins, and mechanism classes. WIP.

## Key Methods

**EigenMap decomposition:** Cross-cell-type attribution matrices are decomposed via eigendecomposition. The first eigenvector (EI_1) captures the dominant axis of variation in motif function across cell types; the second (EI_2) captures orthogonal variation. Sequences are scored on var × ρ (variance × Pearson correlation) to prioritize sequences where the same motifs play consistent functional roles (+1) vs. divergent roles (-1) across cell types. This isolates mechanism classes: same-same, same-diff, and diff-diff.

**In-silico CRISPR:** Necessity tests (KO) marginalize motif positions with dinucleotide-shuffled backgrounds; sufficiency tests (KI) embed motifs into shuffled sequences. This functionally assesses motifs rather than relying on sequence conservation or TFMoDISco+TOMTOM alone.

**SHAPIQ:** Computes higher-order Shapley interaction indices (k-SII) via sampling. Necessity games use sequences with shuffled backgrounds; sufficiency games use the original background. 2-player sufficiency SHAP decomposes regulatory mechanisms into motif syntax vs context contributions.

**SEAM:** Sequence-level Explainable Attribution Maps. Mutagenesis + AlphaGenome attributions clustered to identify coherent regulatory subspaces within target sequences.

**Target library:** 1,059 sequences selected by high EI_1 ratio, correlated importance, TF motif syntax enrichment, and TPM validation. Includes HepG2-biased and K562-biased candidates across mechanism classes.
