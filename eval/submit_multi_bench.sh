#!/bin/bash
#SBATCH --job-name=multibench
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/jindong_gu/LaViDa/logs/multibench_%A_%a.log

# 4 datasets x 1 experiment = 4 tasks (array 0-3)
#
#   sbatch eval/submit_multi_bench.sh
#
# Datasets: mme, pope, textvqa_val, mmbench_en_dev
# Experiment: proposal

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lavida
cd /data/jindong_gu/LaViDa

# ---- dataset table ----
DATASETS=(
  "mme"
  "pope"
  "textvqa_val"
  "mmbench_en_dev"
)
NUM_DATASETS=${#DATASETS[@]}

# ---- experiment table ----
# name | proposal_refine_enable | refine_guidance | gate_tau (empty=none)
EXPS=(
  "proposal          True  none  "
)
NUM_EXPS=${#EXPS[@]}

# ---- pick this task's dataset + experiment ----
if [[ "${SLURM_ARRAY_TASK_ID:-}" == "" ]]; then
  TASK_ID=${TASK_ID:-0}
else
  TASK_ID=${SLURM_ARRAY_TASK_ID}
fi

TOTAL=$((NUM_DATASETS * NUM_EXPS))
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "TASK_ID ${TASK_ID} out of range [0, $((TOTAL - 1))]."
  exit 1
fi

DATASET_IDX=$((TASK_ID / NUM_EXPS))
EXP_IDX=$((TASK_ID % NUM_EXPS))

DATASET="${DATASETS[DATASET_IDX]}"
read -r EXP_NAME PR_ENABLE REFINE_GUID GATE_TAU <<< "${EXPS[EXP_IDX]}"

OUTPUT_ROOT=${OUTPUT_ROOT:-logs/multi_bench}
OUTPUT_PATH="${OUTPUT_ROOT}/${DATASET}/${EXP_NAME}"
mkdir -p "${OUTPUT_PATH}"

echo "TASK_ID=${TASK_ID} DATASET=${DATASET} EXP=${EXP_NAME}"
echo "proposal_refine_enable=${PR_ENABLE} refine_guidance=${REFINE_GUID} gate_tau=${GATE_TAU:-none}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"

export TASKS="${DATASET}"
export CUDA_VISIBLE_DEVICES=0

PROPOSAL_REFINE_ENABLE="${PR_ENABLE}" \
PROPOSAL_STEP=16 \
PROPOSAL_REMASK_RATIO=0.5 \
LATE_REFINE_STEPS=16 \
REMASK_POLICY=confidence \
NULL_VISUAL_MODE=zeros \
REFINE_GUIDANCE="${REFINE_GUID}" \
REFINE_WEAK_VISUAL_MODE=diffusion_noise \
VCD_REFINE_ALPHA=0.5 \
VCD_NOISE_STEP=500 \
VCD_NOISE_SEED=42 \
REFINE_GATE_TAU="${GATE_TAU}" \
MAX_NEW_TOKENS=64 \
STEP_RATIO=0.5 \
MODEL_PATH=/data/jindong_gu/LaViDa/weight/lavida-reason \
RUN_TAG="${EXP_NAME}" \
bash eval/run_proposal_refine.sh \
  --output_path "${OUTPUT_PATH}"
