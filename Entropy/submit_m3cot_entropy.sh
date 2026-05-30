#!/bin/bash
#SBATCH --job-name=m3cot_entropy
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/data/jindong_gu/LaViDa/logs/m3cot_entropy_%j.log

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lavida
cd /data/jindong_gu/LaViDa

DATASET_PATH=${DATASET_PATH:-LightChen2333/M3CoT}
SPLIT=${SPLIT:-test}
PRETRAINED=${PRETRAINED:-weight/lavida-reason}
PROMPT=${PROMPT:-cot}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
BLOCK_LENGTH=${BLOCK_LENGTH:-64}
STEP_PER_BLOCK=${STEP_PER_BLOCK:-}
STEP_RATIO=${STEP_RATIO:-0.5}
LIMIT=${LIMIT:-400}
REMASKING=${REMASKING:-low_confidence}
OUTPUT_DIR=${OUTPUT_DIR:-Entropy/outputs/m3cot_entropy_reason_cot}

ARGS=(
  --dataset-path "${DATASET_PATH}"
  --split "${SPLIT}"
  --pretrained "${PRETRAINED}"
  --prompt "${PROMPT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --block-length "${BLOCK_LENGTH}"
  --limit "${LIMIT}"
  --remasking "${REMASKING}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${STEP_PER_BLOCK}" ]]; then
  ARGS+=(--step-per-block "${STEP_PER_BLOCK}")
else
  ARGS+=(--step-ratio "${STEP_RATIO}")
fi

python Entropy/run_m3cot_entropy.py "${ARGS[@]}"
