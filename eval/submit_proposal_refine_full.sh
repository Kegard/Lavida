#!/bin/bash
#SBATCH --job-name=lavida_propref
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/data/jindong_gu/LaViDa/logs/proposal_refine_%j.log

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lavida

cd /data/jindong_gu/LaViDa

mkdir -p logs/proposal_refine

# 使用方式示例：
# 1. 默认顺序跑两组全量实验
#    sbatch eval/submit_proposal_refine_full.sh
#
# 2. 只跑主结果配置（p8_rr0p25_r8）
#    sbatch --export=ALL,EXPERIMENT_SET=main eval/submit_proposal_refine_full.sh
#
# 3. 只跑高效配置（p8_rr0p1_r4）
#    sbatch --export=ALL,EXPERIMENT_SET=efficient eval/submit_proposal_refine_full.sh
#
# 4. 先在 500 样本上验证
#    sbatch --export=ALL,LIMIT=500 eval/submit_proposal_refine_full.sh
#
# 5. 改输出目录
#    sbatch --export=ALL,OUTPUT_ROOT=./logs/proposal_refine_full eval/submit_proposal_refine_full.sh
#
# 6. 跑同链路 baseline（关闭 proposal-refine，只保留完全相同的 eval 链路）
#    sbatch --export=ALL,EXPERIMENT_SET=baseline eval/submit_proposal_refine_full.sh

MODEL_PATH=${MODEL_PATH:-/data/jindong_gu/LaViDa/weight/lavida}
OUTPUT_ROOT=${OUTPUT_ROOT:-./logs/proposal_refine_full}
LIMIT=${LIMIT:-}
TASKS=${TASKS:-textvqa_val}
EXPERIMENT_SET=${EXPERIMENT_SET:-both}

COMMON_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  COMMON_ARGS+=(--limit "${LIMIT}")
fi

run_one() {
  local run_tag="$1"
  local proposal_step="$2"
  local remask_ratio="$3"
  local refine_steps="$4"
  local proposal_refine_enable="$5"

  echo "========================================"
  echo "Running proposal-refine experiment"
  echo "  TASKS=${TASKS}"
  echo "  RUN_TAG=${run_tag}"
  echo "  proposal_refine_enable=${proposal_refine_enable}"
  echo "  proposal_step=${proposal_step}"
  echo "  proposal_remask_ratio=${remask_ratio}"
  echo "  late_refine_steps=${refine_steps}"
  echo "  OUTPUT_PATH=${OUTPUT_ROOT}"
  if [[ -n "${LIMIT}" ]]; then
    echo "  LIMIT=${LIMIT}"
  else
    echo "  LIMIT=FULL"
  fi
  echo "========================================"

  TASKS="${TASKS}" \
  RUN_TAG="${run_tag}" \
  PROPOSAL_REFINE_ENABLE="${proposal_refine_enable}" \
  PROPOSAL_STEP="${proposal_step}" \
  PROPOSAL_REMASK_RATIO="${remask_ratio}" \
  LATE_REFINE_STEPS="${refine_steps}" \
  MODEL_PATH="${MODEL_PATH}" \
  bash eval/run_proposal_refine.sh \
    --output_path "${OUTPUT_ROOT}" \
    "${COMMON_ARGS[@]}"
}

case "${EXPERIMENT_SET}" in
  both)
    run_one "p8_rr0p25_r8" "8" "0.25" "8" "True"
    run_one "p8_rr0p1_r4" "8" "0.1" "4" "True"
    ;;
  main)
    run_one "p8_rr0p25_r8" "8" "0.25" "8" "True"
    ;;
  efficient)
    run_one "p8_rr0p1_r4" "8" "0.1" "4" "True"
    ;;
  baseline)
    run_one "baseline_same_pipeline" "8" "0.25" "8" "False"
    ;;
  *)
    echo "Unsupported EXPERIMENT_SET=${EXPERIMENT_SET}"
    echo "Expected one of: both, main, efficient, baseline"
    exit 1
    ;;
esac

echo "All requested proposal-refine experiments finished."
