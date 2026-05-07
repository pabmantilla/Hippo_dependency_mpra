"""TF-MoDISco on the top 1000 sequences nearest a target cossim value.
  same-same : 1000 with cossim closest to +inf  (top 1000 desc by cossim)
  diff-diff : 1000 with |cossim| closest to  0
  same-diff : 1000 with cossim closest to -inf  (bottom 1000 asc by cossim)
argv[1] = ct (K562 | HepG2)
argv[2] = stratum (same-same | diff-diff | same-diff)
"""
import os, sys, numpy as np, pandas as pd

REPO = '/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra'
sys.path.insert(0, os.path.join(REPO, 'eigen-interactions'))
from eigen_steering import EigenMap, ENHANCER_LEN
from modiscolite.tfmodisco import TFMoDISco
from modiscolite.io import save_hdf5

CT_ALL = {'K562': 'K562_v6_do075', 'HepG2': 'HepG2_v6_do03'}
STRATA = ('same-same', 'diff-diff', 'same-diff')
ct, stratum = sys.argv[1], sys.argv[2]
assert ct in CT_ALL and stratum in STRATA

N_TOP = 1000
ENH = ENHANCER_LEN
OUT_DIR = os.path.join(REPO, 'genomic_targets/data/modisco')
HP_TAG  = f'strat1k_{stratum}_enh{ENH}_default'
os.makedirs(OUT_DIR, exist_ok=True)
h5 = os.path.join(OUT_DIR, f'{ct}_{HP_TAG}.h5')
if os.path.exists(h5):
    print(f'{ct} {stratum}: cached -> {h5}', flush=True); sys.exit(0)

CT = CT_ALL
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
for c in CT:
    hyp = raw[f'attr_{c}'][:n_full][keep]
    em.attr_hyp[c] = hyp
    em.attr[c] = hyp * ohe
    em.importance[c] = em.attr[c].sum(axis=1)

cossim = em.cosine_similarity(mode='importance', zscore=True)
fin    = np.isfinite(cossim)
fin_idx = np.where(fin)[0]
v = cossim[fin_idx]

if   stratum == 'same-same': order = fin_idx[np.argsort(-v, kind='stable')]
elif stratum == 'same-diff': order = fin_idx[np.argsort( v, kind='stable')]
else:                        order = fin_idx[np.argsort(np.abs(v), kind='stable')]
sel = np.sort(order[:N_TOP])

print(f'{ct} {stratum}: n={len(sel)}/{int(fin.sum())}   '
      f'cossim=[{cossim[sel].min():+.3f},{cossim[sel].max():+.3f}]   '
      f'OMP={os.environ.get("OMP_NUM_THREADS")}', flush=True)

hyp_ct = raw[f'attr_{ct}'][:n_full][keep]
hyp = hyp_ct[sel, :, :ENH].transpose(0, 2, 1).astype(np.float32)
oh  = ohe[sel,    :, :ENH].transpose(0, 2, 1).astype(np.float32)
pos, neg = TFMoDISco(hypothetical_contribs=hyp, one_hot=oh)
save_hdf5(h5, pos, neg, window_size=21)
print(f'{ct} {stratum}: pos={len(pos or [])} neg={len(neg or [])} -> {h5}', flush=True)
