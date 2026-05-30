#!/bin/bash
#SBATCH --job-name=lavida_vrg
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/data/jindong_gu/LaViDa/logs/vrg_%j.log

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate lavida

cd /data/jindong_gu/LaViDa

mkdir -p logs/vrg

# Optional overrides when submitting:
# sbatch --export=ALL,TASKS=textvqa_val,LIMIT=100,VRG_ALPHA_START=0.0,VRG_ALPHA_END=2.0,VRG_ALPHA_SCHEDULE=linear eval/submit_vrg_ablation.sh

MODEL_PATH=${MODEL_PATH:-/data/jindong_gu/LaViDa/weight/lavida}
OUTPUT_PATH=${OUTPUT_PATH:-./logs/vrg/}
LIMIT=${LIMIT:-}

EXTRA_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi

export TASKS=${TASKS:-textvqa_val}
export VRG_ALPHA_START=${VRG_ALPHA_START:-0.0}
export VRG_ALPHA_END=${VRG_ALPHA_END:-2.0}
export VRG_ALPHA_SCHEDULE=${VRG_ALPHA_SCHEDULE:-linear}
export VRG_ALPHA_POWER=${VRG_ALPHA_POWER:-2.0}
export VRG_NULL_VISUAL_MODE=${VRG_NULL_VISUAL_MODE:-zeros}

export RUN_TAG=${RUN_TAG:-vrg_${TASKS}_as${VRG_ALPHA_START}_ae${VRG_ALPHA_END}_sched${VRG_ALPHA_SCHEDULE}_pow${VRG_ALPHA_POWER}_null${VRG_NULL_VISUAL_MODE}}

echo "Running one VRG experiment:"
echo "  TASKS=${TASKS}"
echo "  RUN_TAG=${RUN_TAG}"
echo "  alpha_start=${VRG_ALPHA_START}"
echo "  alpha_end=${VRG_ALPHA_END}"
echo "  alpha_schedule=${VRG_ALPHA_SCHEDULE}"
echo "  alpha_power=${VRG_ALPHA_POWER}"
echo "  null_visual_mode=${VRG_NULL_VISUAL_MODE}"

bash eval/run_vrg.sh \
  --output_path "${OUTPUT_PATH}" \
  "${EXTRA_ARGS[@]}"

echo "Single VRG experiment finished."
