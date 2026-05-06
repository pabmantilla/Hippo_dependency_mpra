#!/bin/bash
#SBATCH --job-name=finemo_mech_bins
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/genomic_targets/scripts/2d_diff_call/scripts/logs/%x/%x.%A_%a.out
#SBATCH --error=/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/genomic_targets/scripts/2d_diff_call/scripts/logs/%x/%x.%A_%a.err

# Submit with: sbatch --array=0-$((2*N_BINS-1)) --export=ALL,N_BINS=$N_BINS run_finemo_mech_bins.sbatch
set -euo pipefail

REPO=/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra
VENV=$REPO/Hippo_agft_venv
N_BINS=${N_BINS:-10}

CTS=(K562 HepG2)
i=$SLURM_ARRAY_TASK_ID
CT=${CTS[$((i / N_BINS))]}
BIN=$((i % N_BINS))

cd "$REPO/genomic_targets/scripts/2d_diff_call/scripts/finemo"
"$VENV/bin/python" run_finemo_mech_bin.py "$CT" "$BIN" "$N_BINS"
