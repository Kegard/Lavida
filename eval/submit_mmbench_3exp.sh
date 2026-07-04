#!/bin/bash
#SBATCH --job-name=mmbench4
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/jindong_gu/LaViDa/logs/mmbench4_%A_%a.log

# Submit ONCE. The job array (--array=0-3) runs 4 experiments in parallel,
# each on its own GPU:
#
#   sbatch eval/submit_mmbench_3exp.sh
#
# Experiments: baseline / proposal / proposal_vrg / proposal_vrg_gate
# Config aligned with M3CoT experiments.

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lavida
cd /data/jindong_gu/LaViDa

# ---- experiment table ----
# name | proposal_refine_enable | refine_guidance | gate_tau (empty=none)
CASES=(
  # "baseline          False none  "
  "proposal          True  none  "
  # "proposal_vrg      True  vcd   "
  # "proposal_vrg_gate True  vcd  0.0"
)

# ---- pick this task's experiment ----
if [[ "${SLURM_ARRAY_TASK_ID:-}" == "" ]]; then
  TASK_ID=${TASK_ID:-0}
else
  TASK_ID=${SLURM_ARRAY_TASK_ID}
fi
if (( TASK_ID < 0 || TASK_ID >= ${#CASES[@]} )); then
  echo "TASK_ID ${TASK_ID} out of range [0, $((${#CASES[@]} - 1))]."
  exit 1
fi

read -r EXP_NAME PR_ENABLE REFINE_GUID GATE_TAU <<< "${CASES[TASK_ID]}"

OUTPUT_ROOT=${OUTPUT_ROOT:-logs/mmbench_4exp}
OUTPUT_PATH="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_PATH}"

echo "TASK_ID=${TASK_ID} EXP=${EXP_NAME}"
echo "proposal_refine_enable=${PR_ENABLE} refine_guidance=${REFINE_GUID} gate_tau=${GATE_TAU:-none}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"

# ---- shared config (aligned with M3CoT) ----
export TASKS="mmbench_en_dev"
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
