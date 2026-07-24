#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/nlp/zhz/PostVRG}"
cd "${PROJECT_ROOT}"

source /home/nlp/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-lavida}"

export HF_HOME="${PROJECT_ROOT}/data/hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/data/hf_cache/datasets"
export HF_HUB_CACHE="${PROJECT_ROOT}/data/hf_cache/hub"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
GPU_MAX_USED_MEM_MB="${GPU_MAX_USED_MEM_MB:-1000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
POLL_SECONDS="${POLL_SECONDS:-60}"

LIMIT="${LIMIT:-400}"
SEED="${SEED:-42}"
PROMPT="${PROMPT:-cot}"
SAMPLE_MODE="${SAMPLE_MODE:-random}"
NOISE_STEP="${NOISE_STEP:-50}"
VCD_NOISE_SEED="${VCD_NOISE_SEED:-42}"
REFILL_GUIDANCE="${REFILL_GUIDANCE:-none}"
REFILL_COND_VISUAL_MODE="${REFILL_COND_VISUAL_MODE:-crop}"
PRIOR_TOP_K="${PRIOR_TOP_K:-8}"
PRIOR_ALPHAS="${PRIOR_ALPHAS:-0.1 0.3 0.5}"
RECORD_PRIOR_EVENTS="${RECORD_PRIOR_EVENTS:-0}"
PRIOR_EVENT_TOP_K="${PRIOR_EVENT_TOP_K:-8}"
MODEL_PATH="${MODEL_PATH:-weight/lavida-reason}"
VISION_TOWER="${VISION_TOWER:-weight/siglip}"

if [[ "${REFILL_GUIDANCE}" != "none" ]]; then
  echo "This sweep is for single-branch refine only; set REFILL_GUIDANCE=none." >&2
  exit 1
fi
if [[ "${REFILL_COND_VISUAL_MODE}" != "crop" ]]; then
  echo "This sweep is for crop refine only; set REFILL_COND_VISUAL_MODE=crop." >&2
  exit 1
fi

EVENT_TAG=""
if [[ "${RECORD_PRIOR_EVENTS}" == "1" ]]; then
  EVENT_TAG="_events_topk${PRIOR_EVENT_TOP_K}"
fi
ALPHA_TAG="${PRIOR_ALPHAS// /_}"
OUTPUT_ROOT="${OUTPUT_ROOT:-M3CoT/PostVRG/outputs/prior_alpha_sweep_m3cot_single_refine_crop_seed${SEED}_topk${PRIOR_TOP_K}_alphas${ALPHA_TAG}${EVENT_TAG}_n${LIMIT}}"
LOG_DIR="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

wait_for_gpu() {
  if [[ "${WAIT_FOR_GPU}" != "1" ]]; then
    return
  fi

  while true; do
    local stats used util
    stats="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
    IFS=',' read -r used util <<< "${stats}"
    used="${used//[[:space:]]/}"
    util="${util//[[:space:]]/}"

    echo "[wait] gpu=${GPU_ID} used=${used}MiB util=${util}% thresholds=${GPU_MAX_USED_MEM_MB}MiB/${GPU_MAX_UTIL}%"
    if [[ "${used}" -le "${GPU_MAX_USED_MEM_MB}" && "${util}" -le "${GPU_MAX_UTIL}" ]]; then
      break
    fi
    sleep "${POLL_SECONDS}"
  done
}

COMMON_ARGS=(
  --dataset-path LightChen2333/M3CoT
  --split test
  --limit "${LIMIT}"
  --sample-mode "${SAMPLE_MODE}"
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
  --device-map "${DEVICE_MAP}"
  --torch-dtype bfloat16
  --draft-steps 16
  --postmask-steps 16
  --fixed-set-size 32
  --fixed-refill-per-step 2
  --draft-guidance none
  --refill-guidance "${REFILL_GUIDANCE}"
  --refill-cond-visual-mode "${REFILL_COND_VISUAL_MODE}"
  --vcd-noise-step "${NOISE_STEP}"
  --vcd-noise-seed "${VCD_NOISE_SEED}"
  --print-every 10
)

run_experiment() {
  local prior_alpha="$1"
  local name="draft_topk_margin_alpha${prior_alpha}"
  local out_dir="${OUTPUT_ROOT}/${name}"
  local log_file="${LOG_DIR}/${name}.log"
  local prior_event_args=()

  if [[ "${RECORD_PRIOR_EVENTS}" == "1" ]]; then
    prior_event_args=(--record-prior-events --prior-event-top-k "${PRIOR_EVENT_TOP_K}")
  fi

  wait_for_gpu
  echo "========== ${name} =========="
  python M3CoT/PostVRG/postvrg.py \
    "${COMMON_ARGS[@]}" \
    --refine-prior draft_topk_margin \
    --prior-top-k "${PRIOR_TOP_K}" \
    --prior-alpha "${prior_alpha}" \
    "${prior_event_args[@]}" \
    --output-dir "${out_dir}" \
    2>&1 | tee "${log_file}"
}

for prior_alpha in ${PRIOR_ALPHAS}; do
  run_experiment "${prior_alpha}"
done
