"""Run TF-MoDISco on the full ~56K joint library for ONE cell type (passed as argv[1]).
Mirrors cells 1+3 of 2d_library.ipynb. Cached: skips if the h5 already exists.
"""
import os, sys, numpy as np, pandas as pd

REPO = '/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra'
sys.path.insert(0, os.path.join(REPO, 'eigen-interactions'))
from eigen_steering import EigenMap, ENHANCER_LEN
from modiscolite.tfmodisco import TFMoDISco
from modiscolite.io import save_hdf5

CT_ALL = {'K562': 'K562_v6_do075', 'HepG2': 'HepG2_v6_do03'}
ct = sys.argv[1]
assert ct in CT_ALL, f'unknown ct {ct!r}; pick from {list(CT_ALL)}'
CT = CT_ALL  # need both for load_from_dataframe / dropna logic

ENH = ENHANCER_LEN
OUT_DIR = os.path.join(REPO, 'genomic_targets/data/modisco')
HP_TAG = f'full56k_enh{ENH}_default'
os.makedirs(OUT_DIR, exist_ok=True)
h5 = os.path.join(OUT_DIR, f'{ct}_{HP_TAG}.h5')
if os.path.exists(h5):
    print(f'{ct}: cached -> {h5}', flush=True); sys.exit(0)

# --- load ---
df = pd.read_csv(os.path.join(REPO, 'data', 'joint_library_combined.csv'))
df = df.dropna(subset=['sequence'] + [f'{c}_log2FC' for c in CT]).reset_index(drop=True)

em = EigenMap(model_names=CT, device='cpu')
em.load_from_dataframe(df, seq_col='sequence')

raw = np.load(os.path.join(REPO, 'genomic_targets/data/deeplift_attributions.npz'))
df_full = pd.read_csv(os.path.join(REPO, 'data', 'joint_library_combined.csv'))
seq_valid = df_full['sequence'].notna(); n_full = seq_valid.sum()
keep = df_full.loc[seq_valid, ['sequence'] + [f'{c}_log2FC' for c in CT]].notna().all(axis=1).values
del df_full

ohe = em.X.numpy()
hyp_full = raw[f'attr_{ct}'][:n_full][keep]

print(f'{ct}: {len(df)} seqs   threads OMP={os.environ.get("OMP_NUM_THREADS")} '
      f'NUMBA={os.environ.get("NUMBA_NUM_THREADS")}', flush=True)

# --- run ---
hyp = hyp_full[:, :, :ENH].transpose(0, 2, 1).astype(np.float32)
oh  = ohe[:, :, :ENH].transpose(0, 2, 1).astype(np.float32)
pos, neg = TFMoDISco(hypothetical_contribs=hyp, one_hot=oh)
save_hdf5(h5, pos, neg, window_size=21)
print(f'{ct}: pos={len(pos or [])} neg={len(neg or [])} -> {h5}', flush=True)
