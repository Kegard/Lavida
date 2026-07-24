#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/nlp/zhz/PostVRG"
cd "${PROJECT_ROOT}"

source /home/nlp/anaconda3/etc/profile.d/conda.sh
conda activate lavida

export HF_HOME="${PROJECT_ROOT}/data/hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/data/hf_cache/datasets"
export HF_HUB_CACHE="${PROJECT_ROOT}/data/hf_cache/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEED="${SEED:-42}"
PROMPT="${PROMPT:-cot}"
MODEL_PATH="${MODEL_PATH:-weight/lavida-reason}"
VISION_TOWER="${VISION_TOWER:-weight/siglip}"
SAMPLER="${SAMPLER:-remdm-rescale}"
ETA="${ETA:-1.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-M3CoT/ReMDM/outputs/multi_bench_${SAMPLER}_eta${ETA}_seed${SEED}}"
LOG_DIR="${OUTPUT_ROOT}/logs"
BENCHMARKS="${BENCHMARKS:-m3cot vstar scienceqa_img mmbench_en_dev}"
LIMIT="${LIMIT:-999999}"

mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --limit "${LIMIT}"
  --sample-mode sequential
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
  --sampler "${SAMPLER}"
  --eta "${ETA}"
  --noise-removal
  --print-every 10
)

benchmark_args() {
  local bench="$1"
  case "${bench}" in
    m3cot)
      echo "--benchmark m3cot --dataset-path LightChen2333/M3CoT --split test"
      ;;
    vstar)
      echo "--benchmark vstar --dataset-path craigwu/vstar_bench --split test --image-root data/vstar_bench"
      ;;
    scienceqa_img)
      echo "--benchmark scienceqa_img --dataset-path lmms-lab/ScienceQA --dataset-name ScienceQA-IMG --split test"
      ;;
    mmbench_en_dev)
      echo "--benchmark mmbench_en_dev --dataset-path lmms-lab/MMBench --dataset-name en --split dev"
      ;;
    *)
      echo "Unknown benchmark: ${bench}" >&2
      return 1
      ;;
  esac
}

run_experiment() {
  local bench="$1"
  local name="${SAMPLER}_eta${ETA}"
  local out_dir="${OUTPUT_ROOT}/${bench}/${name}"
  local log_file="${LOG_DIR}/${bench}_${name}.log"

  echo "========== ${bench}/${name} =========="
  # shellcheck disable=SC2207
  local bench_args=($(benchmark_args "${bench}"))
  python M3CoT/ReMDM/remdm.py \
    "${COMMON_ARGS[@]}" \
    "${bench_args[@]}" \
    --output-dir "${out_dir}" \
    2>&1 | tee "${log_file}"
}

for bench in ${BENCHMARKS}; do
  run_experiment "${bench}"
done
