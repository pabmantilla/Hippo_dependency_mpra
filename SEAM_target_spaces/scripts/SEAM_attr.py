"""Compute AlphaGenome predictions + DeepLIFT/SHAP attributions for SEAM.

For each sequence in the Hippo target library, loads its mutagenesis library
(25K mutants), prepends WT as index 0, and computes predictions + attributions
using AlphaGenome MPRA models for both K562 and HepG2.

Usage:
    python SEAM_attr.py [--start START] [--end END] [--n-shuffles 20] [--cell-type K562]
"""

import argparse
import gc
import os
import sys
import time
import pickle
import numpy as np
import h5py
import torch
from pathlib import Path

# Add eigen-interactions to path for model loading + patches
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EI_DIR = REPO_ROOT / "eigen-interactions"
sys.path.insert(0, str(EI_DIR))

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.transfer import remove_all_heads
from ag_deeplift_patches import patch_alphagenome, AGCustomGELU
from tangermeme.deep_lift_shap import deep_lift_shap, _nonlinear
from tangermeme.ersatz import dinucleotide_shuffle

patch_alphagenome()

# Config
ENCODER_DIM = 1536
PROMOTER_SEQ = 'TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG'
RAND_BARCODE = 'AGAGACTGAGGCCAC'
ENHANCER_LEN = 230
TOTAL_LEN = ENHANCER_LEN + len(PROMOTER_SEQ) + len(RAND_BARCODE)  # 281

WEIGHTS_PATH = str(REPO_ROOT / 'pytorch_base_model' / 'model_fold_0.safetensors')
RESULTS_DIR = str(REPO_ROOT / 'models')

MODEL_NAMES = {
    'K562':  'K562_v6_do075',
    'HepG2': 'HepG2_v6_do03',
}

SEAM_ROOT = Path(__file__).resolve().parent.parent
TARGET_LIB = SEAM_ROOT / "libraries/hippo_target_library.pkl"
MUT_LIB_DIR = SEAM_ROOT / "results/mutagenesis_lib"
OUT_DIR = SEAM_ROOT / "results/attributions"


# ---- Model classes (from eigen_steering.py) ----

class MPRAHead(torch.nn.Module):
    def __init__(self, n_positions=3, nl_size=1024, dropout=0.0,
                 activation='relu', pooling_type='flatten', center_bp=256):
        super().__init__()
        self.pooling_type = pooling_type
        self.n_positions = n_positions
        self.norm = torch.nn.LayerNorm(ENCODER_DIM)
        in_dim = n_positions * ENCODER_DIM if pooling_type == 'flatten' else ENCODER_DIM
        hidden_sizes = [nl_size] if isinstance(nl_size, int) else list(nl_size)
        layers = []
        for hs in hidden_sizes:
            layers.append(torch.nn.Linear(in_dim, hs))
            in_dim = hs
        self.hidden_layers = torch.nn.ModuleList(layers)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.output = torch.nn.Linear(in_dim, 1)
        self.act = torch.nn.GELU() if activation == 'gelu' else torch.nn.ReLU()

    def forward(self, encoder_output):
        x = self.norm(encoder_output)
        if self.pooling_type == 'flatten':
            x = x.flatten(1)
        for layer in self.hidden_layers:
            x = self.act(self.dropout(layer(x)))
        return self.output(x).squeeze(-1)


class AlphaGenomeMPRA(torch.nn.Module):
    """One-hot (B, 4, L) -> (B, 1) for tangermeme, or (B,) if squeeze=True."""
    def __init__(self, encoder, head, squeeze=False):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.squeeze = squeeze

    def forward(self, x):
        x = x.transpose(1, 2)
        org_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        enc_out = self.encoder(
            x, org_idx, encoder_only=True
        )['encoder_output'].transpose(1, 2)
        pred = self.head(enc_out)
        return pred if self.squeeze else pred.unsqueeze(-1)


def load_model(cell_type, device='cuda'):
    model_name = MODEL_NAMES[cell_type]
    ckpt_path = os.path.join(RESULTS_DIR, model_name, 'checkpoints', 'best_stage2.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(RESULTS_DIR, model_name, 'best_stage2.pt')
    enc = AlphaGenome.from_pretrained(WEIGHTS_PATH, device='cpu')
    remove_all_heads(enc)
    hd = MPRAHead()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    enc.load_state_dict(ckpt['model_state_dict'], strict=False)
    hd.load_state_dict(ckpt['head_state_dict'])
    return AlphaGenomeMPRA(enc, hd, squeeze=False).to(device).eval()


# ---- Helpers ----

ALPHA_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def str_to_onehot_cf(seq_str):
    """String -> channels-first (4, L) one-hot."""
    ohe = np.zeros((4, len(seq_str)), dtype=np.float32)
    for j, base in enumerate(seq_str):
        if base in ALPHA_MAP:
            ohe[ALPHA_MAP[base], j] = 1.0
    return ohe


def pad_to_281(x_enhancer_cf):
    """Pad (N, 4, 230) enhancer with promoter+barcode -> (N, 4, 281).

    AlphaGenome expects full 281bp construct: 230bp enhancer + 36bp promoter + 15bp barcode.
    """
    construct_suffix = PROMOTER_SEQ + RAND_BARCODE  # 51bp
    suffix_ohe = str_to_onehot_cf(construct_suffix)  # (4, 51)
    suffix_tiled = np.tile(suffix_ohe, (len(x_enhancer_cf), 1, 1))  # (N, 4, 51)
    return np.concatenate([x_enhancer_cf, suffix_tiled], axis=2)  # (N, 4, 281)


def compute_predictions(model, x_cf, batch_size=512):
    """Batched predictions. x_cf: (N, 4, 281) numpy, channels-first."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_cf), batch_size):
            batch = torch.from_numpy(x_cf[i:i+batch_size]).float().cuda()
            preds.append(model(batch).squeeze(-1).cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


# ---- Main ----

def process_sequence(model, seq_idx, condition, mut_path, out_path,
                     n_shuffles, cell_type):
    # Check if already done
    if out_path.exists():
        with h5py.File(out_path, 'r') as f:
            if 'predictions' in f and 'attributions' in f:
                return False

    # Load mutagenesis lib (NLC ACGT format, WT at index 0)
    with h5py.File(mut_path, 'r') as f:
        all_nlc = f['sequences'][:]  # (25000, 230, 4) — WT is index 0

    # Convert NLC -> NCL (channels-first)
    all_cf = all_nlc.transpose(0, 2, 1)  # (25000, 4, 230)
    del all_nlc

    # Pad to 281bp for AlphaGenome
    all_281 = pad_to_281(all_cf)  # (25000, 4, 281)
    del all_cf

    # Predictions
    print(f"    predictions...")
    predictions = compute_predictions(model, all_281)
    print(f"    -> wt_pred={predictions[0]:.4f}")

    # Attributions — use WT shuffles as references for all sequences
    X = torch.from_numpy(all_281).float()
    t0 = time.time()
    wt_tensor = X[0:1]  # (1, 4, 281)
    print(f"    computing {n_shuffles} dinucleotide shuffles of WT...")
    wt_refs = dinucleotide_shuffle(wt_tensor, n=n_shuffles, random_state=42)  # (1, n_shuffles, 4, 281)
    # Broadcast WT references to all 25K sequences
    refs = wt_refs.expand(len(X), -1, -1, -1)  # (25000, n_shuffles, 4, 281)
    print(f"    shuffles done in {(time.time()-t0)/60:.1f}min")

    print(f"    deep_lift_shap...")
    attr = deep_lift_shap(
        model, X, target=0, references=refs,
        hypothetical=True, batch_size=512,
        device='cuda',
        additional_nonlinear_ops={AGCustomGELU: _nonlinear},
        warning_threshold=0.01, verbose=False,
    ).cpu().numpy()  # (25000, 4, 281)

    # Mean-center across nucleotide channels (hypothetical correction)
    attr = attr - attr.mean(axis=1, keepdims=True)

    # Trim to enhancer region only (first 230bp)
    attr_enh = attr[:, :, :ENHANCER_LEN]  # (25000, 4, 230)

    # Save as NLC ACGT (consistent with mutagenesis lib format)
    attr_nlc = attr_enh.transpose(0, 2, 1)  # (25000, 230, 4)

    with h5py.File(out_path, 'w') as f:
        f.create_dataset('predictions', data=predictions)
        f.create_dataset('attributions', data=attr_nlc,
                         compression='gzip', compression_opts=4)
        f.attrs['seq_idx'] = int(seq_idx)
        f.attrs['condition'] = condition
        f.attrs['cell_type'] = cell_type
        f.attrs['n_shuffles'] = n_shuffles
        f.attrs['alphabet'] = 'ACGT'
        f.attrs['format'] = 'NLC'

    elapsed = (time.time() - t0) / 60
    print(f"    -> {attr_nlc.shape} in {elapsed:.1f}min")

    del X, refs, attr, attr_enh, attr_nlc, all_281
    gc.collect()
    torch.cuda.empty_cache()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--n-shuffles', type=int, default=20)
    parser.add_argument('--cell-type', type=str, default='K562',
                        choices=['K562', 'HepG2'])
    args = parser.parse_args()

    ct = args.cell_type
    out_dir = OUT_DIR / ct
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(TARGET_LIB, 'rb') as f:
        lib = pickle.load(f)
    df = lib['df'].reset_index(drop=True)

    end = args.end if args.end is not None else len(df)
    df = df.iloc[args.start:end]
    print(f"Processing {len(df)} sequences [{args.start}:{end}] | {ct}")

    print(f"Loading {ct} model...")
    model = load_model(ct)

    for i, (_, row) in enumerate(df.iterrows()):
        seq_idx = row['seq_idx']
        condition = row['condition']
        mut_path = MUT_LIB_DIR / f"{condition}_{seq_idx}.h5"
        out_path = out_dir / f"{condition}_{seq_idx}.h5"

        if not mut_path.exists():
            print(f"  [{i+1}/{len(df)}] {condition}/{seq_idx} — mutagenesis lib not found, skipping")
            continue

        print(f"  [{i+1}/{len(df)}] {condition}/{seq_idx}")
        try:
            if not process_sequence(model, seq_idx, condition, mut_path, out_path,
                                    args.n_shuffles, ct):
                print(f"    complete, skipping")
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
