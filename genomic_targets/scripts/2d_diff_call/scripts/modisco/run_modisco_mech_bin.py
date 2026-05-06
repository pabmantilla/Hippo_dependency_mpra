"""TF-MoDISco on one mech-axis bin (equal split into N_BINS contiguous slices
ordered by per-ct cossim of importance, z-scored).

argv[1] = ct (K562 | HepG2)
argv[2] = bin_idx (0 .. n_bins-1)
argv[3] = n_bins (default 10)
"""
import json, os, sys
import numpy as np
import pandas as pd

REPO = '/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra'
sys.path.insert(0, os.path.join(REPO, 'eigen-interactions'))
from eigen_steering import EigenMap, ENHANCER_LEN
from modiscolite.tfmodisco import TFMoDISco
from modiscolite.io import save_hdf5

CT_ALL = {'K562': 'K562_v6_do075', 'HepG2': 'HepG2_v6_do03'}
ct       = sys.argv[1]
bin_idx  = int(sys.argv[2])
n_bins   = int(sys.argv[3]) if len(sys.argv) > 3 else 10
assert ct in CT_ALL
assert 0 <= bin_idx < n_bins

ENH      = ENHANCER_LEN
OUT_ROOT = os.path.join(REPO, 'genomic_targets/data/mech_bins')
BIN_DIR  = os.path.join(OUT_ROOT, ct, f'n{n_bins}', f'bin_{bin_idx:03d}')
H5       = os.path.join(BIN_DIR, 'modisco.h5')
META     = os.path.join(BIN_DIR, 'meta.json')
IDX_NPY  = os.path.join(BIN_DIR, 'indices.npy')
os.makedirs(BIN_DIR, exist_ok=True)

if os.path.exists(H5) and os.path.exists(META):
    print(f'{ct} bin {bin_idx}: cached -> {H5}', flush=True)
    sys.exit(0)

CT = CT_ALL
df = pd.read_csv(os.path.join(REPO, 'data', 'joint_library_combined.csv'))
df = df.dropna(subset=['sequence'] + [f'{c}_log2FC' for c in CT]).reset_index(drop=True)

em = EigenMap(model_names=CT, device='cpu')
em.load_from_dataframe(df, seq_col='sequence')

raw = np.load(os.path.join(REPO, 'genomic_targets/data/deeplift_attributions.npz'))
df_full = pd.read_csv(os.path.join(REPO, 'data', 'joint_library_combined.csv'))
seq_valid = df_full['sequence'].notna()
n_full = seq_valid.sum()
keep = df_full.loc[seq_valid, ['sequence'] + [f'{c}_log2FC' for c in CT]].notna().all(axis=1).values
del df_full

ohe = em.X.numpy()
for c in CT:
    hyp = raw[f'attr_{c}'][:n_full][keep]
    em.attr_hyp[c] = hyp
    em.attr[c]     = hyp * ohe
    em.importance[c] = em.attr[c].sum(axis=1)

cossim = em.cosine_similarity(mode='importance', zscore=True)
fin = np.isfinite(cossim)
order = np.argsort(np.where(fin, cossim, np.nan), kind='stable')
order = order[~np.isnan(np.take(np.where(fin, cossim, np.nan), order))]
n = len(order)
edges = [int(round(i * n / n_bins)) for i in range(n_bins + 1)]
sel = np.sort(order[edges[bin_idx]:edges[bin_idx + 1]])

cossim_lo = float(cossim[sel].min())
cossim_hi = float(cossim[sel].max())
print(f'{ct} bin {bin_idx}/{n_bins}: n={len(sel)}/{n}  '
      f'cossim [{cossim_lo:+.3f}, {cossim_hi:+.3f}]   '
      f'OMP={os.environ.get("OMP_NUM_THREADS")}', flush=True)

hyp_ct = raw[f'attr_{ct}'][:n_full][keep]
hyp = hyp_ct[sel, :, :ENH].transpose(0, 2, 1).astype(np.float32)
oh  = ohe[sel,    :, :ENH].transpose(0, 2, 1).astype(np.float32)
pos, neg = TFMoDISco(hypothetical_contribs=hyp, one_hot=oh)
save_hdf5(H5, pos, neg, window_size=21)

np.save(IDX_NPY, sel)
with open(META, 'w') as f:
    json.dump({
        'ct': ct,
        'bin_idx': bin_idx,
        'n_bins': n_bins,
        'n_seqs': int(len(sel)),
        'n_total': int(n),
        'cossim_lo': cossim_lo,
        'cossim_hi': cossim_hi,
        'enh_len': ENH,
        'n_pos': len(pos or []),
        'n_neg': len(neg or []),
    }, f, indent=2)
print(f'{ct} bin {bin_idx}: pos={len(pos or [])} neg={len(neg or [])} -> {H5}', flush=True)
