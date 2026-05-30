#!/bin/bash
#SBATCH --job-name=lavida_reason_base
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/data/jindong_gu/LaViDa/logs/proposal_refine_reason_baseline_%j.log

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lavida

cd /data/jindong_gu/LaViDa

mkdir -p logs/proposal_refine_reason_baseline

# 使用方式示例：
# 1. 默认提交一组 lavida-reason / TextVQA reasoning prompt / baseline 实验
#    sbatch eval/submit_proposal_refine_reason_baseline.sh
#
# 2. 只跑前 100 条做快速验证
#    sbatch --export=ALL,LIMIT=100 eval/submit_proposal_refine_reason_baseline.sh

MODEL_PATH=${MODEL_PATH:-/data/jindong_gu/LaViDa/weight/lavida-reason}
OUTPUT_PATH=${OUTPUT_PATH:-./logs/proposal_refine_reason_baseline}
LIMIT=${LIMIT:-}

export TASKS=${TASKS:-textvqa_val}
export TEXTVQA_PROMPT_MODE=${TEXTVQA_PROMPT_MODE:-reasoning}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
export BLOCK_LENGTH=${BLOCK_LENGTH:-}
export STEP_PER_BLOCK=${STEP_PER_BLOCK:-}
export STEP_RATIO=${STEP_RATIO:-0.5}
export PROPOSAL_REFINE_ENABLE=${PROPOSAL_REFINE_ENABLE:-False}
export PROPOSAL_STEP=${PROPOSAL_STEP:-8}
export PROPOSAL_REMASK_RATIO=${PROPOSAL_REMASK_RATIO:-0.25}
export LATE_REFINE_STEPS=${LATE_REFINE_STEPS:-8}
export REMASK_POLICY=${REMASK_POLICY:-confidence}
export NULL_VISUAL_MODE=${NULL_VISUAL_MODE:-zeros}
export RUN_TAG=${RUN_TAG:-baseline_reason}

EXTRA_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi

echo "Running one proposal-refine reasoning baseline experiment:"
echo "  TASKS=${TASKS}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  TEXTVQA_PROMPT_MODE=${TEXTVQA_PROMPT_MODE}"
echo "  MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "  BLOCK_LENGTH=${BLOCK_LENGTH:-AUTO}"
echo "  STEP_PER_BLOCK=${STEP_PER_BLOCK:-AUTO}"
echo "  STEP_RATIO=${STEP_RATIO:-AUTO}"
echo "  RUN_TAG=${RUN_TAG}"
echo "  PROPOSAL_REFINE_ENABLE=${PROPOSAL_REFINE_ENABLE}"
if [[ -n "${LIMIT}" ]]; then
  echo "  LIMIT=${LIMIT}"
else
  echo "  LIMIT=FULL"
fi

MODEL_PATH="${MODEL_PATH}" \
bash eval/run_proposal_refine.sh \
  --output_path "${OUTPUT_PATH}" \
  "${EXTRA_ARGS[@]}"

echo "Single proposal-refine reasoning baseline experiment finished."
