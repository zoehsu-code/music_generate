#!/bin/bash
#SBATCH --job-name=ossl-shift-diff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/shift-diff-%j.out
#SBATCH --error=logs/shift-diff-%j.err

set -euo pipefail
module load python/3.12.1
PROJECT_DIR="${PROJECT_DIR:-/home/xvming/OSSL_Event_Alignment/music_generate}"
ROOT=/scratch/aimusic_project_root/xvming/OSSL_Event_Alignment
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-shift-diff-${SLURM_JOB_ID}"
"$PROJECT_DIR/.venv/bin/python" -u analyze_train_shift_difference.py \
  --scores "${ALIGNMENT_SCORES:-$ROOT/analysis/train_saved_curves_28185053/alignment_scores.csv}" \
  --shift-scores "${PRIOR_SHIFT_SCORES:-$ROOT/analysis/train_temporal_shift_28224289/per_clip.csv}" \
  --video-root "${VIDEO_ROOT:-$ROOT/event_curves}" \
  --music-root "${MUSIC_RESULT_ROOT:-$ROOT/music_event_curves/msd_l7_wav_run01}" \
  --output-root "${SHIFT_DIFFERENCE_RESULT_ROOT:-$ROOT/analysis/train_shift_difference_${SLURM_JOB_ID}}" "$@"
