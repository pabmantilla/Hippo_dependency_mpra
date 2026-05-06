#!/bin/bash
#SBATCH --job-name=modisco_mech
#SBATCH --partition=highmemq
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/genomic_targets/scripts/2d_diff_call/scripts/logs/%x/%x.%A_%a.out
#SBATCH --error=/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/genomic_targets/scripts/2d_diff_call/scripts/logs/%x/%x.%A_%a.err

set -euo pipefail

REPO=/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra
VENV=$REPO/Hippo_agft_venv

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

CTS=(K562 HepG2)
STRATA=(same-diff diff-diff same-same)
CT=${CTS[$((SLURM_ARRAY_TASK_ID / 3))]}
STRATUM=${STRATA[$((SLURM_ARRAY_TASK_ID % 3))]}

cd "$REPO/genomic_targets/scripts/2d_diff_call/scripts/modisco"
"$VENV/bin/python" run_modisco_mech.py "$CT" "$STRATUM"
