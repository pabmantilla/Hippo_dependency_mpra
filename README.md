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
SEAM_target_spaces/             SEAM foreground/background separation on targets
virtual_perturbations/          Necessity/sufficiency perturbation screens
syntax_SHAPIQ/                  Higher-order interaction decomposition
```

## Experiments

### Characterizing MPRA library with EigenMaps

- **3-cell-type eigen decomposition** — EI_1/EI_2 eigenvector angles over the full 57k library show WTC11 is a linear combination of liver and blood regulatory programs. Polar histograms of EI angles reveal shared vs. cell-type-specific modes.
- **Liver-blood target selection** — Focus the 57k library down to 1,059 sequences by EI ratio + importance correlation. Balanced across three conditions: same-diff (353), diff-diff (353), same-same (353). Hippo-relevant TF families (HNF, STAT, AP-1, TEA) used as selection keywords. Validated against TPM expression and motif enrichment.
- **Model validation** — 9 PyTorch AlphaGenome models (3 cell types x 3 dropout rates) validated against JAX baseline predictions.

### SEAM foreground/background separation

- **Random mutagenesis libraries** — 25K mutants per target at 10% mutation rate (squid RandomMutagenesis). WT at index 0, saved as gzipped HDF5 per sequence. Enhancer-only (230bp), promoter+barcode appended at prediction time.
- **DeepSHAP attributions on mutant libraries** — AlphaGenome DeepLIFT/SHAP on all 25K mutants per sequence, per cell type (K562 and HepG2 separately). Uses 20 dinucleotide shuffles of WT as references, hypothetical contributions mean-centered across nucleotide channels. Attributions trimmed to enhancer region (230bp).
- **SEAM clustering + MetaExplainer** — K-means (30 clusters) on flattened attribution maps, then MetaExplainer sorts clusters by median activity, computes MSM, and separates scaled foreground (motif signal) from background (sequence context) with adaptive background scaling (entropy_multiplier=0.5). Foreground, background, cluster maps, and ref cluster index saved per sequence per cell type.
- **Foreground interpretation** — Re-annotate motifs on SEAM foreground signal instead of raw DeepSHAP. Side-by-side comparison of raw attributions vs. foreground logos with TF annotations isolates cell-type-specific binding from background noise. Same EigenMap annotation pipeline (window_size, flank, pval_thresh, gaussian binding scores).

### In-silico perturbations on target library

- **Target visualization** — HepG2 vs. K562 predicted log2FC scatter colored by EI_1 var x r; EI score distributions by condition (same-diff, diff-diff, same-same). Focus TF family counts (HNF, STAT, AP-1, TEA) across same-diff and full library. Example attribution logos for extreme same-diff, same-same, and diff-diff sequences.
- **1st-order necessity/sufficiency** — Dinucleotide-shuffle KO/KI (n_rep=30, max_order=3) on all 1,059 targets. Necessity scores sign-flipped (positive = more necessary). Per-motif HepG2 vs. K562 scatter colored by EI_1 var x r (inferno). Per-TF mean necessity/sufficiency distributions and scatter (size=count, n>=3 filter) with top off-diagonal TFs labeled. Context players (background, promoter, barcode) scored separately with per-cell-type histograms.

### In-silico decomposition (SHAPIQ)

- **Necessity-mode SHAPIQ** — k-SII orders 1-4 with motifs + background as context-aware players. Promoter and barcode excluded from interaction terms. 1-SII (Shapley values) compared to KO necessity scores; 2-SII and 3-SII interactions visualized as scatter vs. necessity and sii_frac histograms. Max sii_frac per sequence identifies dominant interaction type per cell line.
- **Sufficiency-mode SHAPIQ** — k-SII orders 1-4 (n_rep=20), same player structure. Per-annotation cell type and condition violin plots. 7,238 interaction terms across 1,059 sequences (4,704 order-1, 1,928 order-2, 502 order-3, 104 order-4).

## Key Methods

**EigenMap decomposition:** Cross-cell-type attribution matrices are decomposed via eigendecomposition. The first eigenvector (EI_1) captures the dominant axis of variation in motif function across cell types; the second (EI_2) captures orthogonal variation. Sequences are scored on var x r (variance x Pearson correlation) to prioritize sequences where the same motifs play consistent functional roles (+1) vs. divergent roles (-1) across cell types.

**SHAPIQ:** Computes exact higher-order Shapley interaction indices via sampling. Necessity tests use sequences with shuffled dinucleotide backgrounds; sufficiency tests use the original background.

**SEAM:** Surrogate Epistasis Attribution Maps. Random mutagenesis + DeepSHAP attributions on mutant libraries, clustered and decomposed via MetaExplainer into foreground (motif-driven) and background (context-driven) components. Separates the signal that matters from the signal that doesn't.

**Target library:** 1,059 sequences selected by high EI_1 ratio, correlated importance, and TF motif presence. Includes both HepG2-biased and K562-biased candidates for experimental validation.
