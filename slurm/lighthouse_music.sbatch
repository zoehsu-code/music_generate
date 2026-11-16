#!/bin/bash
#SBATCH --job-name=ossl-music
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/music-%A_%a.out
#SBATCH --error=logs/music-%A_%a.err

set -euo pipefail
module load python/3.12.1
PROJECT_DIR="${PROJECT_DIR:-/home/xvming/OSSL_Event_Alignment/music_generate}"
OSSL_ROOT="${OSSL_ROOT:-/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/data/OSSL-v2}"
HF_HOME="${HF_HOME:-/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/cache/huggingface}"
INPUT_KIND="${INPUT_KIND:-wav}"
MUSIC_RESULT_ROOT="${MUSIC_RESULT_ROOT:-/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment/music_event_curves/msd_l7_${INPUT_KIND}_run01}"
: "${SLURM_ARRAY_TASK_ID:?Submit with --array=0-N}"
: "${SLURM_ARRAY_TASK_COUNT:?Missing array size}"
if [[ "$SLURM_ARRAY_TASK_MIN" != 0 || "$SLURM_ARRAY_TASK_STEP" != 1 || "$SLURM_ARRAY_TASK_MAX" != "$((SLURM_ARRAY_TASK_COUNT - 1))" ]]; then
    echo 'Use a contiguous zero-based array, e.g. --array=0-63%2' >&2
    exit 2
fi
cd "$PROJECT_DIR"
export HF_HOME
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}"
"$PROJECT_DIR/.venv/bin/python" -u extract_ossl_music_batch.py \
    --dataset-root "$OSSL_ROOT" --output-root "$MUSIC_RESULT_ROOT" \
    --cache-dir "$HF_HOME/hub" --local-files-only --device cuda \
    --input-kind "$INPUT_KIND" \
    --num-shards "$SLURM_ARRAY_TASK_COUNT" \
    --shard-index "$SLURM_ARRAY_TASK_ID" "$@"
