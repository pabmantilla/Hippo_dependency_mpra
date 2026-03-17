# Hippo Dependency MPRA

Using *in silico* MPRA experiments to dissect **Hippo pathway dependency** in cancer.

## Overview

Deep learning models trained on lentiMPRA data predict enhancer activity across cell lines (K562, HepG2, WTC11). We leverage these models to run virtual MPRA screens on the joint library (~57k sequences) and decompose regulatory logic across cell types using eigen-interactions of DeepLIFT/SHAP attribution maps.

## Structure

```
models/                     Fine-tuned AlphaGenome heads per cell line & dropout
data/                       Joint library sequences and activity measurements
eigen-interactions/         Submodule: eigen-decomposition of cross-cell-type attributions
genomic_targets/
  scripts/
    validate_models.ipynb           Model validation (Pearson r across cell lines)
    eigen_interactions_filtering.ipynb  Load attributions, eigendecompose, filter
    submit_attributions.sh          SLURM array job pipeline for DeepLIFT/SHAP
  data/
    attr_shards/                    Per-cell-type attribution shards (.npz)
    deeplift_attributions.npz       Merged hypothetical attribution maps
```

## Models

Best models per cell line (v6 two-step fine-tuned AlphaGenome):

| Cell line | Model | Pearson r |
|-----------|-------|-----------|
| K562 | K562_v6_do075 | 0.8915 |
| HepG2 | HepG2_v6_do03 | 0.8750 |
| WTC11 | WTC11_v6_do075 | 0.8457 |

## Attribution Pipeline

DeepLIFT/SHAP attributions (50 dinucleotide shuffles) are computed via SLURM array jobs:

```bash
bash genomic_targets/scripts/submit_attributions.sh test    # test 1 seq
bash genomic_targets/scripts/submit_attributions.sh         # submit 30 jobs (3 cell types x 10 shards)
bash genomic_targets/scripts/submit_attributions.sh merge   # merge shards
```

Attributions are saved as hypothetical corrected maps (mean-centered, pre one-hot multiply) so all 4 nucleotide channels are preserved. Logo-ready maps are derived on load via `attr_hyp * ohe`.

## Progress

1. Trained v6 two-step models for K562, HepG2, and WTC11
2. Validated predictions on the joint library
3. Computing DeepLIFT/SHAP attributions across all 57k sequences and 3 cell lines
4. Eigen-interaction decomposition of cross-cell-type covariance matrices

## Next Steps

- Eigendecompose attribution covariance to find shared vs cell-type-specific regulatory modes
- Filter eigen-interaction results for Hippo axis regulators: TEAD1-4, AP-1 (JUN/FOS), YAP/TAZ, VGLL
- 3D visualization of EI_1 projected onto (HepG2, K562, WTC11) coordinates
