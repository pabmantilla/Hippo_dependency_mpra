## Overview

Reading the genetic code is chaotic yet robust. Deep learning models trained on MPRA data serve as virtual platforms where we can perturb cis-regulatory features to measure local (necessity / CRISPR KO) and global (sufficiency / CRISPR KD) contributions.

This repo characterizes, focuses, perturbs, and decomposes the cis-regulatory code from a joint HepG2/K562/WTC11 MPRA library, ending in decomposition of signaling pathways — differential Hippo dependence, the additivity of transcriptional initiation, and the combinatorial nature of chromatin-remodeling TFs.

## Hippo Dependency MPRA

MPRA of ~57k enhancers in HepG2 (liver), K562 (blood), and WTC11 (iPSC). Cross-cell-type DeepLIFT/SHAP attributions are reduced to a similarity score per sequence, then perturbed and decomposed.

**Key metrics:**
- `EI_1 var x r`: cross-cell-type motif-function similarity (+1 shared, 0 unrelated, -1 flipped)
- necessity/sufficiency: dinucleotide-shuffle KO/KI on focus library
- SHAPIQ k-SII: interaction decomposition (orders 1–4)

## Structure

Shared inputs:
```
models/               Fine-tuned AlphaGenome models (K562, HepG2, WTC11)
data/                 Joint library sequences
eigen-interactions/   Submodule: EigenMap class for attribution decomposition
```

Pipeline (isolate → explain → perturb → decompose):
```
genomic_targets/         1. Eigen decomposition & focus target selection
SEAM_target_spaces/      2. SEAM foreground/background separation on focus lib
virtual_perturbations/   3. Necessity/sufficiency perturbation screens
syntax_SHAPIQ/           4. Higher-order interaction decomposition
```

## Experiments

### Characterize with EigenMaps
- **3-cell-type decomposition** — EI_1/EI_2 angles over the 57k library; WTC11 is a linear combination of liver and blood programs.
- **Focus selection** — 1,059 targets by EI ratio + importance correlation, balanced across same-diff / diff-diff / same-same (353 each). Hippo-relevant families (HNF, STAT, AP-1, TEA) used as keywords.
- **Model validation** — 9 PyTorch AlphaGenome models (3 cell types × 3 dropouts) against JAX baseline.

### SEAM foreground/background
- **Mutagenesis libraries** — 25K mutants/target at 10% rate (squid), 230bp enhancer, promoter+barcode appended at predict time.
- **DeepSHAP on mutants** — per cell type, 20 dinuc-shuffle references, mean-centered hypothetical contributions.
- **Clustering + MetaExplainer** — K-means (30) → MSM → scaled foreground (motif signal) vs background (context), entropy_multiplier=0.5.
- **Foreground interpretation** — re-annotate motifs on foreground to isolate cell-type-specific binding from context noise.

### Perturbations on focus library
- **Target visualization** — HepG2 vs K562 pred log2FC scatter colored by EI_1 var x r; EI distributions by condition; focus-family counts.
- **1st-order necessity/sufficiency** — dinuc-shuffle KO/KI (n_rep=30, max_order=3) on all 1,059. Per-motif HepG2 vs K562 scatter (inferno). Context players (background, promoter, barcode) scored separately.

### Decomposition (SHAPIQ)
- **Necessity-mode** — k-SII orders 1–4 with motifs + background as players (promoter/barcode excluded). 1-SII vs KO necessity; 2/3-SII vs necessity; per-sequence dominant-order via max sii_frac.
- **Sufficiency-mode** — k-SII orders 1–4 (n_rep=20). 7,238 interaction terms across 1,059 sequences (4,704 / 1,928 / 502 / 104 by order).

## Key Methods

**Cross-cell-type similarity (EigenMaps):** We reduce the (motifs × cell types) attribution matrix per sequence to a single similarity score. EigenMap does this via eigendecomposition — EI_1 captures the dominant axis of motif-function variation across cell types, EI_2 the orthogonal one; `var x r` (variance × Pearson r of the projection) scores shared (+1) vs divergent (-1) motif function. This is **one way** of measuring cross-cell-type similarity; per-position cosine similarity or Pearson r would work too. EigenMap was picked for convenience (plays well with motif ranking, annotation, and downstream SHAPIQ players), not because the metric is uniquely correct.

**SEAM:** Build a mutagenesis library, compute attribution maps, cluster them, and use MetaExplainer to separate foreground (mutation-sensitive motif signal) from background (mutation-robust context).

**SHAPIQ:** Exact higher-order Shapley interaction indices via sampling. Necessity tests use dinuc-shuffled backgrounds; sufficiency tests use the WT background.

**Focus library:** 1,059 sequences selected by high EI_1 ratio, correlated importance, and TF motif presence — both HepG2-biased and K562-biased candidates.
