#!/bin/bash
#SBATCH --job-name=lavida_budget
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/data/jindong_gu/LaViDa/logs/textvqa_native_budget_%j.log

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lavida

cd /data/jindong_gu/LaViDa

mkdir -p logs/textvqa_native_budget_sweep

# 这个脚本只跑“原始策略 / 同链路 baseline”，不启用 proposal-refine。
# 目标是系统检查：
# 1. max_new_tokens=32 时，不同 step_ratio 对 TextVQA 指标的影响；
# 2. step_ratio=1 时，不同 max_new_tokens 对 TextVQA 指标的影响。
#
# 默认配置：
# - 样本数：1000
# - 模型：weight/lavida
# - 任务：textvqa_val
#
# 使用方式：
# 1. 默认直接提交两组 sweep
#    sbatch eval/submit_textvqa_native_budget_sweep.sh
#
# 2. 只跑 step-ratio sweep
#    sbatch --export=ALL,EXPERIMENT_SET=step_ratio eval/submit_textvqa_native_budget_sweep.sh
#
# 3. 只跑 max-new-tokens sweep
#    sbatch --export=ALL,EXPERIMENT_SET=max_tokens eval/submit_textvqa_native_budget_sweep.sh
#
# 4. 改样本数
#    sbatch --export=ALL,LIMIT=500 eval/submit_textvqa_native_budget_sweep.sh

MODEL_PATH=${MODEL_PATH:-/data/jindong_gu/LaViDa/weight/lavida}
OUTPUT_ROOT=${OUTPUT_ROOT:-./logs/textvqa_native_budget_sweep}
LIMIT=${LIMIT:-1000}
TASKS=${TASKS:-textvqa_val}
EXPERIMENT_SET=${EXPERIMENT_SET:-both}

run_one() {
  local run_tag="$1"
  local max_new_tokens="$2"
  local block_length="$3"
  local step_ratio="$4"

  echo "========================================"
  echo "Running native budget sweep experiment"
  echo "  TASKS=${TASKS}"
  echo "  RUN_TAG=${run_tag}"
  echo "  PROPOSAL_REFINE_ENABLE=False"
  echo "  MAX_NEW_TOKENS=${max_new_tokens}"
  echo "  BLOCK_LENGTH=${block_length}"
  echo "  STEP_RATIO=${step_ratio}"
  echo "  LIMIT=${LIMIT}"
  echo "  OUTPUT_PATH=${OUTPUT_ROOT}"
  echo "========================================"

  TASKS="${TASKS}" \
  RUN_TAG="${run_tag}" \
  MODEL_PATH="${MODEL_PATH}" \
  PROPOSAL_REFINE_ENABLE="False" \
  PROPOSAL_STEP="8" \
  PROPOSAL_REMASK_RATIO="0.25" \
  LATE_REFINE_STEPS="8" \
  MAX_NEW_TOKENS="${max_new_tokens}" \
  BLOCK_LENGTH="${block_length}" \
  STEP_RATIO="${step_ratio}" \
  bash eval/run_proposal_refine.sh \
    --output_path "${OUTPUT_ROOT}" \
    --limit "${LIMIT}"
}

run_step_ratio_sweep() {
  local max_new_tokens="32"
  local block_length="32"
  local ratios=(
    "0.03125"
    "0.0625"
    "0.125"
    "0.25"
    "0.5"
    "1.0"
  )

  for ratio in "${ratios[@]}"; do
    local tag="native_m32_sr${ratio//./p}"
    run_one "${tag}" "${max_new_tokens}" "${block_length}" "${ratio}"
  done
}

run_max_token_sweep() {
  local values=("32" "16" "8" "4")

  for max_new_tokens in "${values[@]}"; do
    local tag="native_m${max_new_tokens}_sr1p0"
    run_one "${tag}" "${max_new_tokens}" "${max_new_tokens}" "1.0"
  done
}

case "${EXPERIMENT_SET}" in
  both)
    run_step_ratio_sweep
    run_max_token_sweep
    ;;
  step_ratio)
    run_step_ratio_sweep
    ;;
  max_tokens)
    run_max_token_sweep
    ;;
  *)
    echo "Unsupported EXPERIMENT_SET=${EXPERIMENT_SET}"
    echo "Expected one of: both, step_ratio, max_tokens"
    exit 1
    ;;
esac

echo "All requested native budget sweep experiments finished."
