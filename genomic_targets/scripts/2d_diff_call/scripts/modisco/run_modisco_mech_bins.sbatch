#!/bin/bash
#SBATCH --job-name=modisco_mech_bins
#SBATCH --partition=highmemq
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/genomic_targets/scripts/2d_diff_call/scripts/logs/%x/%x.%A_%a.out
#SBATCH --error=/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/genomic_targets/scripts/2d_diff_call/scripts/logs/%x/%x.%A_%a.err

# Submit with: sbatch --array=0-$((2*N_BINS-1)) --export=ALL,N_BINS=$N_BINS run_modisco_mech_bins.sbatch
set -euo pipefail

REPO=/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra
VENV=$REPO/Hippo_agft_venv
N_BINS=${N_BINS:-10}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

CTS=(K562 HepG2)
i=$SLURM_ARRAY_TASK_ID
CT=${CTS[$((i / N_BINS))]}
BIN=$((i % N_BINS))

cd "$REPO/genomic_targets/scripts/2d_diff_call/scripts/modisco"
"$VENV/bin/python" run_modisco_mech_bin.py "$CT" "$BIN" "$N_BINS"
