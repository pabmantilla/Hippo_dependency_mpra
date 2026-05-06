"""TF-MoDISco on a mech-stratified third of the 56K joint library.
Stratum is by H<->K cossim (importance, z-scored) -- the mech axis.
  same-diff = bottom third (cossim ~ -1)
  diff-diff = middle third (cossim ~  0)
  same-same = top third    (cossim ~ +1)
argv[1] = ct (K562 | HepG2)         (which ct's attributions feed modisco)
argv[2] = stratum (same-diff | diff-diff | same-same)
"""
import os, sys, numpy as np, pandas as pd

REPO = '/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra'
sys.path.insert(0, os.path.join(REPO, 'eigen-interactions'))
from eigen_steering import EigenMap, ENHANCER_LEN
from modiscolite.tfmodisco import TFMoDISco
from modiscolite.io import save_hdf5

CT_ALL = {'K562': 'K562_v6_do075', 'HepG2': 'HepG2_v6_do03'}
STRATA = ['same-diff', 'diff-diff', 'same-same']   # bottom -> top thirds of cossim
ct, stratum = sys.argv[1], sys.argv[2]
assert ct in CT_ALL and stratum in STRATA

ENH = ENHANCER_LEN
OUT_DIR = os.path.join(REPO, 'genomic_targets/data/modisco')
HP_TAG  = f'mech_{stratum}_enh{ENH}_default'
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
fin = np.isfinite(cossim)
order = np.argsort(np.where(fin, cossim, np.nan), kind='stable')
order = order[~np.isnan(np.take(np.where(fin, cossim, np.nan), order))]
n = len(order)
edges = [0, n // 3, 2 * n // 3, n]
buckets = {name: order[edges[i]:edges[i+1]] for i, name in enumerate(STRATA)}
sel = np.sort(buckets[stratum])
print(f'{ct} {stratum}: n={len(sel)}/{n}  cossim range '
      f'[{cossim[sel].min():+.3f}, {cossim[sel].max():+.3f}]   '
      f'threads OMP={os.environ.get("OMP_NUM_THREADS")}', flush=True)

hyp_ct = raw[f'attr_{ct}'][:n_full][keep]
hyp = hyp_ct[sel, :, :ENH].transpose(0, 2, 1).astype(np.float32)
oh  = ohe[sel,    :, :ENH].transpose(0, 2, 1).astype(np.float32)
pos, neg = TFMoDISco(hypothetical_contribs=hyp, one_hot=oh)
save_hdf5(h5, pos, neg, window_size=21)
print(f'{ct} {stratum}: pos={len(pos or [])} neg={len(neg or [])} -> {h5}', flush=True)
