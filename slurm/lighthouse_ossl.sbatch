#!/bin/bash
#SBATCH --job-name=ossl-events
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/ossl-%A_%a.out
#SBATCH --error=logs/ossl-%A_%a.err

set -euo pipefail
: "${PROJECT_DIR:?Set PROJECT_DIR to the repository on Lighthouse}"
: "${OSSL_ROOT:?Set OSSL_ROOT to the downloaded dataset}"
: "${RESULT_ROOT:?Set RESULT_ROOT to the output directory}"
: "${HF_HOME:?Set HF_HOME to a writable model cache on project storage}"
: "${SLURM_ARRAY_TASK_ID:?Submit with --array=0-N}"
: "${SLURM_ARRAY_TASK_COUNT:?Missing array size}"
# Use contiguous zero-based arrays with step 1; the mapping must not overlap.
if [[ "${SLURM_ARRAY_TASK_MIN}" != 0 || "${SLURM_ARRAY_TASK_STEP}" != 1 ]]; then
    echo 'Use a contiguous zero-based array, e.g. --array=0-15%2' >&2
    exit 2
fi
cd "$PROJECT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}"
"$PROJECT_DIR/.venv/bin/python" -u extract_ossl_batch.py \
    --dataset-root "$OSSL_ROOT" \
    --output-root "$RESULT_ROOT" \
    --cache-dir "$HF_HOME/hub" \
    --local-files-only --device cuda \
    --batch-size "${BATCH_SIZE:-4}" \
    --num-shards "$SLURM_ARRAY_TASK_COUNT" \
    --shard-index "$SLURM_ARRAY_TASK_ID" "$@"
