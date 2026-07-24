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

LIMIT="${LIMIT:-999999}"
SEED="${SEED:-42}"
PROMPT="${PROMPT:-cot}"
SAMPLE_MODE="${SAMPLE_MODE:-random}"
NOISE_STEP="${NOISE_STEP:-50}"
VCD_NOISE_SEED="${VCD_NOISE_SEED:-42}"
PRIOR_TOP_K="${PRIOR_TOP_K:-8}"
PRIOR_ALPHA="${PRIOR_ALPHA:-0.1}"
RECORD_PRIOR_EVENTS="${RECORD_PRIOR_EVENTS:-0}"
PRIOR_EVENT_TOP_K="${PRIOR_EVENT_TOP_K:-8}"
MODEL_PATH="${MODEL_PATH:-weight/lavida-reason}"
VISION_TOWER="${VISION_TOWER:-weight/siglip}"

REFILL_GUIDANCE="none"
REFILL_COND_VISUAL_MODE="spotlight"

EVENT_TAG=""
if [[ "${RECORD_PRIOR_EVENTS}" == "1" ]]; then
  EVENT_TAG="_events_topk${PRIOR_EVENT_TOP_K}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-M3CoT/PostVRG/outputs/prior_alpha_sweep_m3cot_single_refine_spotlight_seed${SEED}_topk${PRIOR_TOP_K}_alphas${PRIOR_ALPHA}${EVENT_TAG}_n${LIMIT}}"
RUN_NAME="draft_topk_margin_alpha${PRIOR_ALPHA}"
OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

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

prior_event_args=()
if [[ "${RECORD_PRIOR_EVENTS}" == "1" ]]; then
  prior_event_args=(--record-prior-events --prior-event-top-k "${PRIOR_EVENT_TOP_K}")
fi

wait_for_gpu

echo "========== ${RUN_NAME} =========="
echo "output_dir=${OUT_DIR}"
echo "setting: draft=full image, refine=${REFILL_COND_VISUAL_MODE} single-branch, prior=draft_topk_margin alpha=${PRIOR_ALPHA}"

python M3CoT/PostVRG/postvrg.py \
  --dataset-path LightChen2333/M3CoT \
  --split test \
  --limit "${LIMIT}" \
  --sample-mode "${SAMPLE_MODE}" \
  --sample-seed "${SEED}" \
  --prompt "${PROMPT}" \
  --max-new-tokens 64 \
  --block-length 64 \
  --step-ratio 0.5 \
  --pretrained "${MODEL_PATH}" \
  --vision-tower "${VISION_TOWER}" \
  --model-name llava_llada \
  --conv-template llada \
  --device cuda \
  --device-map "${DEVICE_MAP}" \
  --torch-dtype bfloat16 \
  --draft-steps 16 \
  --postmask-steps 16 \
  --fixed-set-size 32 \
  --fixed-refill-per-step 2 \
  --draft-guidance none \
  --refill-guidance "${REFILL_GUIDANCE}" \
  --refill-cond-visual-mode "${REFILL_COND_VISUAL_MODE}" \
  --vcd-noise-step "${NOISE_STEP}" \
  --vcd-noise-seed "${VCD_NOISE_SEED}" \
  --refine-prior draft_topk_margin \
  --prior-top-k "${PRIOR_TOP_K}" \
  --prior-alpha "${PRIOR_ALPHA}" \
  "${prior_event_args[@]}" \
  --print-every 10 \
  --output-dir "${OUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"
