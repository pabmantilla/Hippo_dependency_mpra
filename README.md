# 🦛 Hippo Dependency MPRA

Using *in silico* MPRA experiments to dissect **Hippo pathway dependency** in cancer.

## 🧬 Overview

Deep learning models trained on lentiMPRA data predict enhancer activity across cell lines (K562, HepG2, WTC11). We leverage these models to run virtual MPRA screens on joint libraries (~57k sequences) and decompose regulatory logic across cell types.

## 📂 Structure

- `models/` — Fine-tuned AlphaGenome heads per cell line & dropout regime
- `data/` — Joint library sequences and activity measurements
- `genomic_targets/` — Filtering and analysis notebooks
- `eigen-interactions/` — Submodule for eigen-decomposition of cross-cell-type attributions

## 🔬 Current Progress

1. ✅ Trained v6 two-step models for K562, HepG2, and WTC11
2. ✅ Validated predictions on the joint library
3. 🔄 Applying **eigen-interactions** to define regulatory syntax — eigendecomposing DeepLIFT attributions across cell types to identify shared and cell-type-specific regulatory grammar

## 🎯 Next Steps

Filter eigen-interaction results to interrogate key Hippo axis regulators:

- **TEAD** family (TEAD1–4)
- **AP-1** (JUN/FOS)
- **STAT** factors
- **YAP/TAZ** co-activators
- **VGLL** competitors
