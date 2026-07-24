#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/nlp/zhz/PostVRG"
cd "${PROJECT_ROOT}"

source /home/nlp/anaconda3/etc/profile.d/conda.sh
conda activate lavida

export HF_HOME="${PROJECT_ROOT}/data/hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/data/hf_cache/datasets"
export HF_HUB_CACHE="${PROJECT_ROOT}/data/hf_cache/hub"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LIMIT="${LIMIT:-9999}"
SEED="${SEED:-42}"
PROMPT="${PROMPT:-cot}"
MODEL_PATH="${MODEL_PATH:-weight/lavida-reason}"
VISION_TOWER="${VISION_TOWER:-weight/siglip}"
REFINE_VISUAL_MODE="${REFINE_VISUAL_MODE:-spotlight}"
OUTPUT_ROOT="${OUTPUT_ROOT:-M3CoT/PostVRG/outputs/edge_noise_sweep_refine-${REFINE_VISUAL_MODE}_seed${SEED}_n${LIMIT}}"
LOG_DIR="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --dataset-path LightChen2333/M3CoT
  --split test
  --limit "${LIMIT}"
  --sample-mode random
  --sample-seed "${SEED}"
  --prompt "${PROMPT}"
  --max-new-tokens 64
  --block-length 64
  --step-ratio 0.5
  --pretrained "${MODEL_PATH}"
  --vision-tower "${VISION_TOWER}"
  --model-name llava_llada
  --conv-template llada
  --device cuda
  --device-map cuda:0
  --torch-dtype bfloat16
  --print-every 10
)

run_experiment() {
  local name="$1"
  shift

  echo "========== ${name} =========="
  python M3CoT/PostVRG/postvrg_final.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "${OUTPUT_ROOT}/${name}" \
    "$@" \
    2>&1 | tee "${LOG_DIR}/${name}.log"
}

run_experiment "baseline_draft64_s32_t2" \
  --draft-visual-mode full \
  --refine-visual-mode full \
  --draft-steps 32 \
  --postmask-steps 0

for noise_step in 0 50 100 150 200; do
  run_experiment "edge_noise${noise_step}_draft16_refine16_${REFINE_VISUAL_MODE}_d4_r2" \
    --draft-visual-mode edge_noise \
    --refine-visual-mode "${REFINE_VISUAL_MODE}" \
    --vcd-noise-step "${noise_step}" \
    --vcd-noise-seed 42 \
    --draft-steps 16 \
    --postmask-steps 16 \
    --fixed-set-size 32 \
    --fixed-refill-per-step 2
done
