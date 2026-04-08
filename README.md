## Overview

Reading the genetic code is a highly chaotic, yet surprisingly robust process. Perturbations to key elements of the cis- and trans-regulatory code in model organisms have allowed independent definitions of epistasis, robustness, expressivity, and evolvability in reference to complex outcomes, or phenotypes. Deep learning models learn sequence elements predictive of function in a fixed trans-regulatory state. Using these models as virtual experimental platforms allows us to to target known feature's (eg. tfbs) local contribution (necessity test — CRISPR KO) and global contribution (sufficiency test — CRISPR KD).

In this repo we apply a principled approach, EigenMaps, to characterize, focus, perturb, and decompose the cis-regulatory code from the MPRA joint library. We conclude with the decomposition of the 'defining' signaling pathways — independently explaining the differential dependence in the Hippo pathway, the surprisingly additive nature of regulatory code when promoting transcriptional initiation, and the combinatorial nature of TFs when chromatin remodeling.


## Hippo Dependency MPRA

MPRA of ~57k enhancer sequences in HepG2 (Liver Hepatocyte-Hepatoblastoma), K562 (Bone Marrow Myeloid-CML), and WTC11 (Skin Fibroblast-iPSC) lines, with eigen-interaction decomposition of cross-cell-type DeepLIFT/SHAP attributions to identify shared vs. cell-type-specific regulatory modes and higher-order epistasis via SHAPIQ.

**Key metrics:**
- EI_1 var x r: motif function similarity across cell types (+1 = same motifs same function, 0 = unrelated, -1 = same motifs different function)
- Necessity/sufficiency tests: dinucleotide-shuffle KO/KI on target library
- SHAPIQ k-SII: context-aware interaction decomposition (orders 1-4)

## Structure

```
models/                         Fine-tuned AlphaGenome models (K562, HepG2, WTC11)
data/                           Joint library sequences
eigen-interactions/             Submodule: EigenMap class for attribution decomposition
genomic_targets/                Eigen decomposition & target selection
virtual_perturbations/          Necessity/sufficiency perturbation screens
syntax_SHAPIQ/                  Higher-order interaction decomposition
SEAM_target_spaces/             SEAM foreground/background separation on targets
```

## Experiments

### Characterizing MPRA library with EigenMaps

- **3-cell-type eigen decomposition** — EI_1/EI_2 eigenvector angles over the full 57k library show WTC11 is a linear combination of liver and blood regulatory programs. Polar histograms of EI angles reveal shared vs. cell-type-specific modes.
- **Liver-blood target selection** — Focus the 57k library down to 1,059 sequences by EI ratio + importance correlation. Balanced across three conditions: same-diff (353), diff-diff (353), same-same (353). Validated against TPM expression and motif enrichment.
- **Model validation** — 9 PyTorch AlphaGenome models (3 cell types x 3 dropout rates) validated against JAX baseline predictions.

### In-silico perturbations on target library

- **1st-order necessity/sufficiency** — Dinucleotide-shuffle KO/KI (n_rep=30, max_order=3) on all 1,059 targets. Per-motif and per-TF score distributions colored by EI_1 var x r. Context players (background, promoter, barcode) scored separately.
- **Target visualization** — HepG2 vs. K562 predicted log2FC scatter colored by EI_1 var x r (inferno); top candidates with attribution logos.

### In-silico decomposition (SHAPIQ)

- **Necessity-mode SHAPIQ** — k-SII orders 1-4 with background/promoter/barcode as context players. Comparison to KO necessity scores.
- **Sufficiency-mode SHAPIQ** — k-SII orders 1-4; per-annotation cell type and condition violin plots.

### SEAM foreground/background separation

- **Random mutagenesis libraries** — 25K mutants at 10% mutation rate per target sequence (squid RandomMutagenesis).
- **DeepSHAP attributions on mutant libraries** — AlphaGenome DeepLIFT/SHAP on all 25K mutants per sequence, per cell type.
- **SEAM clustering + MetaExplainer** — K-means (30 clusters) on attribution maps, then MetaExplainer to separate scaled foreground (motif signal) from background (sequence context). Foreground and background saved per sequence per cell type.
- **Foreground interpretation** — Compare raw DeepSHAP attributions vs. SEAM foregrounds: re-annotate motifs on foreground-only signal to isolate cell-type-specific TF binding from background noise.

## Key Methods

**EigenMap decomposition:** Cross-cell-type attribution matrices are decomposed via eigendecomposition. The first eigenvector (EI_1) captures the dominant axis of variation in motif function across cell types; the second (EI_2) captures orthogonal variation. Sequences are scored on var x r (variance x Pearson correlation) to prioritize sequences where the same motifs play consistent functional roles (+1) vs. divergent roles (-1) across cell types.

**SHAPIQ:** Computes exact higher-order Shapley interaction indices via sampling. Necessity tests use sequences with shuffled dinucleotide backgrounds; sufficiency tests use the original background.

**SEAM:** Surrogate Epistasis Attribution Maps. Random mutagenesis + DeepSHAP attributions on mutant libraries, clustered and decomposed via MetaExplainer into foreground (motif-driven) and background (context-driven) components. Separates the signal that matters from the signal that doesn't.

**Target library:** 1,059 sequences selected by high EI_1 ratio, correlated importance, and TF motif presence. Includes both HepG2-biased and K562-biased candidates for experimental validation.
