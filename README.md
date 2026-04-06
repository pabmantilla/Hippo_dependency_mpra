# Hippo Dependency MPRA

Dissecting cell-type-specific regulatory grammar between HepG2 and K562 using LentiMPRA + AlphaGenome models.

Given a LentiMPRA library and cell lines, EigenMap supports:
- Training and validating cell-type-specific deep learning models
- Mechanistic interpretation via EigenMaps: eigendecomposition of cross-cell-type attributions to classify sequences by regulatory mechanism (same-same, same-diff, diff-diff)
- In silico CRISPR KO/KI: necessity and sufficiency tests via marginalized dinucleotide-shuffle perturbations to functionally assess motifs — not just sequence conservation or TFMoDISco+TOMTOM, but principled functional testing downstream of mechanistic description
- Isolating sequences that describe mechanistic differences and similarities between cell lines
- SEAM decomposition of target regulatory spaces
- n-SHAPIQ: higher-order Shapley interaction indices (k-SII, orders 1-4) for motifs and context players via necessity and sufficiency games
- Context SHAP: 2-player Shapley decomposition of regulatory mechanisms into motif syntax vs background context
- Motif Context Swap: swapping motif syntax and backgrounds across cell lines, activity bins, and mechanism classes

## Data

~57k enhancer sequences (281bp: 230bp enhancer + 51bp promoter/barcode) assayed in HepG2, K562, and WTC11 via LentiMPRA. AlphaGenome models fine-tuned per cell type (3 dropout rates each).

1,059 target sequences selected via EI decomposition, motif syntax enrichment, and TPM validation for downstream perturbation analysis.

## Structure

```
data/                           Joint MPRA library
models/                         Fine-tuned AlphaGenome models (3 cell types x 3 dropout rates)
pytorch_base_model/             Base model checkpoint
eigen-interactions/             Submodule: EigenMap class
genomic_targets/
  scripts/validation/           Model validation (PyTorch vs JAX)
  scripts/2d_targeting/         EI decomposition, target selection (1059 seqs)
  scripts/3d_example/           3-cell-type decomposition (WTC11 as linear combo)
virtual_perturbations/
  scripts/                      Necessity/sufficiency perturbation screens
  libraries/                    Target libraries
syntax_SHAPIQ/
  scripts/                      n-SHAPIQ and context SHAP notebooks
SEAM_target_spaces/
  scripts/                      Mutagenesis, attributions, SEAM clustering
Motif_context_swap/             Context swap experiments
```

## Pipeline

### 1. Model validation
`genomic_targets/scripts/validation/validate_models.ipynb` — validate 9 PyTorch models against JAX baseline.

### 2. EigenMap decomposition
`genomic_targets/scripts/2d_targeting/liver_blood_targets.ipynb` — eigendecompose 57k library, EI_1/EI_2 eigenvector analysis, polar histograms of mechanism space.

`genomic_targets/scripts/2d_targeting/hippo_target_selection.ipynb` — select 1,059 targets by EI ratio, importance correlation, motif syntax enrichment, TPM validation.

`genomic_targets/scripts/3d_example/eigen_interactions_filtering.ipynb` — WTC11 as linear combination of HepG2/K562.

### 3. In silico perturbations
`virtual_perturbations/scripts/perturb_targets.ipynb` — necessity (KO) and sufficiency (KI) tests on 1,059 targets. Dinucleotide-shuffle marginalization, n_rep=30, combinatorial orders up to 3.

### 4. n-SHAPIQ interaction decomposition
`syntax_SHAPIQ/scripts/Hippo_nec_SHAPIQ.ipynb` — necessity-mode k-SII (orders 1-4). Each motif is a player; context players (background, promoter, barcode) capture non-motif contributions. Exact computation over all 2^n coalitions.

`syntax_SHAPIQ/scripts/Hippo_suf_SHAPIQ.ipynb` — sufficiency-mode k-SII (orders 1-4). Same n-player game in sufficiency (KI) mode.

### 5. Context SHAP
`shapley_syntax_vs_background()` — 2-player game decomposing WT prediction into motif syntax (identity + spacing + orientation) vs background context. Closed-form Shapley values. Ratio shap(background)/shap(motif_syntax) quantifies how much context matters relative to motif grammar per sequence per cell line.

### 6. SEAM target spaces
`SEAM_target_spaces/scripts/` — mutagenesis perturbations, AlphaGenome attributions, SEAM clustering to decompose regulatory subspaces within the target library.

### 7. Motif Context Swap
`motif_context_swap()` — swap motif syntax from one group into backgrounds from another. Modes: cell_lines (within-sequence cross-annotation), activity (cross-sequence by predicted expression), mechanism (cross-sequence by EI_1 var x r class).

## Key metric

**EI_1 var x r**: captures cell-type divergence of motif function. +1 = same motifs, same function (same-same). -1 = same motifs, opposite function (diff-diff). 0 = uncorrelated (same-diff).
