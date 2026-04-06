## Overview

Reading the genetic code is a highly chaotic, yet suprisingly robust proccess. Peterburbations to key elements of the cis- and trans- regulatory code in model organisms has allowed idnpendepnt definitions of epistatisis, robustness, espressivity, and evolbability in reference to coomplex outcomes, or phenotypes. Deep learnign has been isntrimental to tasks in regulatory genomics by learning sequence elements predictiive of fucntion in a fized trans-regulatory state.Useing e models as virtual experimental platorms allows us to pereterb in the same way as before to understand known featuer local controbution (nceccesity test -CRIPS KO) and glabnal contibution (suff test -CRISPR KD ). 

In this repo applie a principled approach , EIgenMAPs, to chacterize,  focus, peterb and decompose the cisregulatory code from the mpra joiitn library. We conclude with the decomposiiton of the 'defining' signaling pathways- indpendently explaingin teh Differential dependence in the Hippo pathway, suprosingly additive nature of regulatory code when promoting trasnctiptional initaiton, and the [TBD] nature of TFs when chromatin remodling.  


## Hippo Dependency MPRA

  on MPRA of ~57k enhancer sequences in HepG2 (Liver Hepatocyte-Hepatoblastoma), K562(Bone marrow Myleoid-CML), and WTC11(Skin (leg) Fibroblast-iPSC) lines.   with eigen-interaction decomposition of cross-cell-type DeepLIFT/SHAP attributions to identify shared vs. cell-type-specific regulatory modes and higher-order epistasis via SHAPIQ.

**Key metrics:**
- EI_1 var × ρ: encodes cell-type divergence (negative = HepG2-specific, positive = shared/K562-biased)
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

### Validation of pytorch vs. jax alphagenome models
| `validation/validate_models.ipynb` | Validate 9 PyTorch models (3 cell types × 3 dropout rates) vs JAX baseline|



### Characterizing MPRA library with EigenMaps (genomic_targets/)

| Script | Purpose |
|--------|---------|
| `3d_example/eigen_interactions_filtering.ipynb` | WTC11 cells can be descibed as linear combinations of Li cells |

### Isolate mechanistic space of joint library with liver-blood basis. (genomic_targets/)

| Script | Purpose |
|--------|---------|
| `2d_targeting/hippo_target_selection.ipynb` | Focus 57k seqs by EI ratio + importance correlation to explore describitve mecahanisms |

| `2d_targeting/liver_blood_targets.ipynb` | Eigen-decompose full 57k library; EI_1/EI_2 eigenvector angles, polar histograms; identify shared vs differential mechanisms |


### In-silico Perturbations (virtual_perterbations/)

| Script | Purpose |
|--------|---------|
| `show_hippo_targets.ipynb` | Visualize 1,059-seq target library; HepG2 vs K562 predictions colored by EI_1 var × ρ; top candidates with attribution logos |
| `perturb_targets.ipynb` | Necessity/sufficiency tests on all targets (dinucleotide shuffle KO/KI, n_rep=30, max_order=3); per-motif and per-TF score distributions |

### In-silico Decomposition (syntax_SHAPIQ/)

| Script | Purpose |
|--------|---------|
| `Hippo_nec_SHAPIQ.ipynb` | Necessity-mode SHAPIQ: k-SII for orders 1–4; context-aware decomposition with background/promoter/barcode players; comparison to KO scores |
| `Hippo_suf_SHAPIQ.ipynb` | Sufficiency-mode SHAPIQ: k-SII for orders 1–4; per-annotation cell type and condition violin plots |

## Key Methods

**EigenMap decomposition:** Cross-cell-type attribution matrices are decomposed via eigendecomposition. The first eigenvector (EI_1) captures the dominant axis of cell-type divergence; the second (EI_2) captures orthogonal variation. Sequences are scored on "var × ρ" (variance × Pearson correlation) to prioritize stable, cell-type-divergent patterns.

**SHAPIQ:** Computes exact higher-order Shapley interaction indices via sampling. Necessity tests use sequences with shuffled dinucleotide backgrounds; sufficiency tests use the original background.

**Target library:** 1,059 sequences selected by high EI_1 ratio, correlated importance, and TF motif presence. Includes both HepG2-biased and K562-biased candidates for experimental validation.
