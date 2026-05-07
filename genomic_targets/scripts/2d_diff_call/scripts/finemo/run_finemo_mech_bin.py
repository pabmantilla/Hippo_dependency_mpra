"""finemo call-hits for one mech-axis bin.

argv[1] = ct
argv[2] = bin_idx
argv[3] = n_bins (default 10)
argv[4] = mode (eqc | eqw, default eqc)
"""
import os, sys, subprocess

REPO = '/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra'
ct      = sys.argv[1]
bin_idx = int(sys.argv[2])
n_bins  = int(sys.argv[3]) if len(sys.argv) > 3 else 10
mode    = sys.argv[4] if len(sys.argv) > 4 else 'eqc'
assert mode in ('eqc', 'eqw')

TAG      = f'n{n_bins}' if mode == 'eqc' else f'eqw_n{n_bins}'
BIN_DIR  = os.path.join(REPO, 'genomic_targets/data/mech_bins', ct, TAG, f'bin_{bin_idx:03d}')
H5       = os.path.join(BIN_DIR, 'modisco.h5')
REGIONS  = os.path.join(REPO, 'genomic_targets/data/motif', ct, 'regions.npz')
OUT_DIR  = os.path.join(BIN_DIR, 'finemo')
FINEMO   = os.path.join(REPO, '.venv/bin/finemo')

if not os.path.exists(H5):
    print(f'{ct} bin {bin_idx}: no modisco.h5 ({H5}); skipping finemo', flush=True)
    sys.exit(0)
assert os.path.exists(REGIONS), f'missing {REGIONS}'
os.makedirs(OUT_DIR, exist_ok=True)

if os.path.exists(os.path.join(OUT_DIR, 'hits.tsv')):
    print(f'{ct} bin {bin_idx}: finemo cached -> {OUT_DIR}', flush=True)
    sys.exit(0)

cmd = [FINEMO, 'call-hits', '-r', REGIONS, '-m', H5, '-o', OUT_DIR]
print(' '.join(cmd), flush=True)
subprocess.run(cmd, check=True)
print(f'{ct} bin {bin_idx}: finemo done -> {OUT_DIR}', flush=True)
