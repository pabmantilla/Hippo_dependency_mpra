#!/bin/bash
# Submit DeepLIFT attribution shards as SLURM array jobs.
#
# Each array task computes attributions for one (cell_type, chunk) pair.
# Layout: 3 cell types x N_SHARDS chunks = 3*N_SHARDS array tasks.
#   task_id = ct_idx * N_SHARDS + shard_idx
#
# Usage:
#   bash submit_attributions.sh              # submit full array (30 jobs)
#   bash submit_attributions.sh test         # 1 seq, 1 cell line, interactive
#   bash submit_attributions.sh merge        # merge after all jobs finish

set -euo pipefail

# --- Config ---
REPO="/grid/wsbs/home_norepl/pmantill/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra"
EIGEN="${REPO}/eigen-interactions/eigen_steering.py"
PYTHON="${REPO}/Hippo_agft_venv/bin/python"
CSV="${REPO}/data/joint_library_combined.csv"
SHARD_DIR="${REPO}/genomic_targets/data/attr_shards"
MERGED="${REPO}/genomic_targets/data/deeplift_attributions.npz"
WEIGHTS="/grid/wsbs/home_norepl/pmantill/LentiMPRA_mcs/alphagenome_torch_MPRAMoCon/weights/model_fold_0.safetensors"
RESULTS_DIR="${REPO}/models"

N_SHARDS=10
CELL_TYPES=("K562" "HepG2" "WTC11")
MODEL_NAMES=("K562_v6_do075" "HepG2_v6_do03" "WTC11_v6_do075")
N_TASKS=$(( ${#CELL_TYPES[@]} * N_SHARDS ))  # 30

# --- Test mode: 1 seq, 1 cell line, run locally ---
if [[ "${1:-}" == "test" ]]; then
    TEST_DIR="${SHARD_DIR}/test"
    rm -rf "${TEST_DIR}"
    mkdir -p "${TEST_DIR}"

    # Make a 1-sequence CSV
    head -1 "${CSV}" > "${TEST_DIR}/test.csv"
    # grab first row with a valid sequence
    awk -F',' 'NR>1 && $0 !~ /^,/ {print; exit}' "${CSV}" >> "${TEST_DIR}/test.csv"

    echo "=== Test: 1 sequence, K562 only ==="
    "${PYTHON}" "${EIGEN}" shard \
        --csv "${TEST_DIR}/test.csv" \
        --seq-col sequence \
        --cell-type K562 \
        --model-name K562_v6_do075 \
        --output-dir "${TEST_DIR}" \
        --shard-idx 0 \
        --n-shards 1 \
        --weights-path "${WEIGHTS}" \
        --results-dir "${RESULTS_DIR}" \
        --n-shuffles 50 \
        --batch-size 1

    echo ""
    echo "=== Checking output ==="
    "${PYTHON}" -c "
import numpy as np, os
f = '${TEST_DIR}/K562_shard_0000.npz'
d = np.load(f)
print(f'File: {f} ({os.path.getsize(f)/1e3:.1f} KB)')
for k in d:
    print(f'  {k}: shape={d[k].shape} dtype={d[k].dtype} range=[{d[k].min():.4f}, {d[k].max():.4f}]')
# Verify hypothetical: should have nonzero values at all 4 channels
attr = d['attr']
nonzero_per_channel = (attr[0] != 0).sum(axis=1)
print(f'  Nonzero positions per channel (should be >0 for all 4): {nonzero_per_channel}')
"
    echo ""
    echo "Test passed! Clean up with: rm -rf ${TEST_DIR}"
    exit 0
fi

# --- Merge mode ---
if [[ "${1:-}" == "merge" ]]; then
    echo "Merging shards from ${SHARD_DIR} -> ${MERGED}"
    "${PYTHON}" "${EIGEN}" merge \
        --output-dir "${SHARD_DIR}" \
        --cell-types "${CELL_TYPES[@]}" \
        --output-path "${MERGED}" \
        --cleanup
    exit 0
fi

# --- Submit array ---
LOG_DIR="${REPO}/genomic_targets/scripts/slurm_logs"
mkdir -p "${SHARD_DIR}" "${LOG_DIR}"

JOBSCRIPT=$(mktemp /tmp/attr_shard_XXXX.sbatch)
cat > "${JOBSCRIPT}" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=deeplift_shard
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=${LOG_DIR}/slurm_%A_%a.out

set -euo pipefail

N_SHARDS=${N_SHARDS}
CT_IDX=\$(( SLURM_ARRAY_TASK_ID / N_SHARDS ))
SHARD_IDX=\$(( SLURM_ARRAY_TASK_ID % N_SHARDS ))

CELL_TYPES_ARR=(${CELL_TYPES[*]})
MODEL_NAMES_ARR=(${MODEL_NAMES[*]})

CT="\${CELL_TYPES_ARR[\$CT_IDX]}"
MODEL="\${MODEL_NAMES_ARR[\$CT_IDX]}"

echo "Task \${SLURM_ARRAY_TASK_ID}: \${CT} shard \${SHARD_IDX}/\${N_SHARDS}"

${PYTHON} ${EIGEN} shard \\
    --csv ${CSV} \\
    --cell-type "\${CT}" \\
    --model-name "\${MODEL}" \\
    --output-dir ${SHARD_DIR} \\
    --shard-idx "\${SHARD_IDX}" \\
    --n-shards \${N_SHARDS} \\
    --weights-path ${WEIGHTS} \\
    --results-dir ${RESULTS_DIR} \\
    --n-shuffles 50 \\
    --batch-size 50
SBATCH_EOF

echo "Submitting ${N_TASKS} array tasks (${#CELL_TYPES[@]} cell types x ${N_SHARDS} shards)"
echo "Jobscript: ${JOBSCRIPT}"
cat "${JOBSCRIPT}"
echo ""

sbatch --array=0-$(( N_TASKS - 1 )) "${JOBSCRIPT}"
