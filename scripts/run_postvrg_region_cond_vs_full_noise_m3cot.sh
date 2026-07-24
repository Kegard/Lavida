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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_HOME="/usr/local/cuda"
export PATH="${CUDA_HOME}/bin:${PATH}"

LIMIT="${LIMIT:-9999}"
SEED="${SEED:-42}"
PROMPT="${PROMPT:-cot}"
NOISE_STEP="${NOISE_STEP:-50}"
VCD_REFILL_ALPHA="${VCD_REFILL_ALPHA:-0.5}"
REFILL_GUIDANCE_STEPS="${REFILL_GUIDANCE_STEPS:-16}"
MODEL_PATH="${MODEL_PATH:-weight/lavida-reason}"
VISION_TOWER="${VISION_TOWER:-weight/siglip}"
OUTPUT_ROOT="${OUTPUT_ROOT:-M3CoT/PostVRG/outputs/region_cond_vs_full_noise_m3cot_seed${SEED}_noise${NOISE_STEP}_alpha${VCD_REFILL_ALPHA}_k${REFILL_GUIDANCE_STEPS}_n${LIMIT}}"
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
  --draft-steps 16
  --postmask-steps 16
  --fixed-set-size 32
  --fixed-refill-per-step 2
  --refill-guidance vcd
  --refill-weak-visual-mode full_noise
  --vcd-refill-alpha "${VCD_REFILL_ALPHA}"
  --refill-guidance-steps "${REFILL_GUIDANCE_STEPS}"
  --vcd-noise-step "${NOISE_STEP}"
  --vcd-noise-seed 42
  --print-every 10
)

run_experiment() {
  local cond_mode="$1"
  local name="${cond_mode}_cond_vs_full_noise"

  echo "========== ${name} =========="
  python M3CoT/PostVRG/postvrg.py \
    "${COMMON_ARGS[@]}" \
    --refill-cond-visual-mode "${cond_mode}" \
    --output-dir "${OUTPUT_ROOT}/${name}" \
    2>&1 | tee "${LOG_DIR}/${name}.log"
}

run_experiment "crop"
run_experiment "spotlight"
