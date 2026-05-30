#!/bin/bash
#SBATCH --job-name=lavida_reweight     # 任务名称
#SBATCH --partition=tamper_resistance  # 确认分区名，通常训练用 GPU 分区
#SBATCH --nodes=1                      # 申请 1 个节点
#SBATCH --cpus-per-task=2              # 建议申请 8 个以上的 CPU 核心
#SBATCH --mem=64G                      # 内存限制
#SBATCH --gpus=1                   # 申请 1 块 GPU (按需修改)
#SBATCH --time=24:00:00               # 任务运行时间上限，格式为 HH:MM:SS
#SBATCH --output=/data/jindong_gu/LaViDa/logs/reweights_%j.log     # 日志存放路径（请确保 logs 文件夹已存在）


set -euo pipefail
# 激活你的 conda 环境

source $(conda info --base)/etc/profile.d/conda.sh
conda activate lavida # 请替换为你配置好的环境名


cd /data/jindong_gu/LaViDa

mkdir -p logs/reweight

# Optional overrides when submitting:
# sbatch --export=ALL,TASKS=textvqa_val,LIMIT=100,REWEIGHT_ALPHA_VISUAL=1.05,REWEIGHT_ALPHA_MASK=0.95 eval/submit_taca_ablation.sh

MODEL_PATH=${MODEL_PATH:-/data/jindong_gu/LaViDa/weight/lavida}
OUTPUT_PATH=${OUTPUT_PATH:-./logs/reweight/}
LIMIT=${LIMIT:-}

# -----------------------------
# Run one reweight experiment
# -----------------------------

EXTRA_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi

export TASKS=${TASKS:-textvqa_val}
export REWEIGHT_ALPHA_PROMPT=${REWEIGHT_ALPHA_PROMPT:-1.0}
export REWEIGHT_ALPHA_VISUAL=${REWEIGHT_ALPHA_VISUAL:-1.0}
export REWEIGHT_ALPHA_GENERATED=${REWEIGHT_ALPHA_GENERATED:-1.0}
export REWEIGHT_ALPHA_MASK=${REWEIGHT_ALPHA_MASK:-${REWEIGHT_ALPHA_GENERATED}}
export REWEIGHT_ALPHA_NORMAL=${REWEIGHT_ALPHA_NORMAL:-${REWEIGHT_ALPHA_GENERATED}}
export REWEIGHT_ALPHA_SPECIAL=${REWEIGHT_ALPHA_SPECIAL:-${REWEIGHT_ALPHA_GENERATED}}

export RUN_TAG=${RUN_TAG:-reweight_${TASKS}_ap${REWEIGHT_ALPHA_PROMPT}_av${REWEIGHT_ALPHA_VISUAL}_ag${REWEIGHT_ALPHA_GENERATED}_am${REWEIGHT_ALPHA_MASK}_an${REWEIGHT_ALPHA_NORMAL}_as${REWEIGHT_ALPHA_SPECIAL}}

echo "Running one reweight experiment:"
echo "  TASKS=${TASKS}"
echo "  RUN_TAG=${RUN_TAG}"
echo "  alpha_prompt=${REWEIGHT_ALPHA_PROMPT}"
echo "  alpha_visual=${REWEIGHT_ALPHA_VISUAL}"
echo "  alpha_generated=${REWEIGHT_ALPHA_GENERATED}"
echo "  alpha_mask=${REWEIGHT_ALPHA_MASK}"
echo "  alpha_normal=${REWEIGHT_ALPHA_NORMAL}"
echo "  alpha_special=${REWEIGHT_ALPHA_SPECIAL}"

bash eval/run_reweight.sh \
  --output_path "${OUTPUT_PATH}" \
  "${EXTRA_ARGS[@]}"

echo "Single reweight experiment finished."
