#!/bin/bash
#SBATCH --job-name=textvqa_bpk4
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/data/jindong_gu/LaViDa/logs/textvqa_bpk4_%A_%a.log

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lavida
cd /data/jindong_gu/LaViDa

mkdir -p logs/textvqa_base_proposal_k4

MODEL_PATH=${MODEL_PATH:-/data/jindong_gu/LaViDa/weight/lavida-reason}
OUTPUT_ROOT=${OUTPUT_ROOT:-VRG/outputs/textvqa_base_proposal_k4}
LIMIT=${LIMIT:-}

export TEXTVQA_PROMPT_MODE=${TEXTVQA_PROMPT_MODE:-reasoning}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
export BLOCK_LENGTH=${BLOCK_LENGTH:-64}
export STEP_PER_BLOCK=${STEP_PER_BLOCK:-}
export STEP_RATIO=${STEP_RATIO:-0.5}
export BASE_DRAFT_STEPS=${BASE_DRAFT_STEPS:-32}
export DRAFT_STEPS=${DRAFT_STEPS:-16}
export POSTMASK_STEPS=${POSTMASK_STEPS:-16}
export REMASK_PER_STEP=${REMASK_PER_STEP:-4}
export FIXED_SET_SIZE=${FIXED_SET_SIZE:-32}
export FIXED_REFILL_PER_STEP=${FIXED_REFILL_PER_STEP:-2}
export REMASK_POLICY=${REMASK_POLICY:-confidence}
export NULL_VISUAL_MODE=${NULL_VISUAL_MODE:-zeros}
export VCD_REFILL_ALPHA=${VCD_REFILL_ALPHA:-0.5}
export VCD_NOISE_STEP=${VCD_NOISE_STEP:-500}
export VCD_NOISE_SEED=${VCD_NOISE_SEED:-42}

CASES=(
  "base"
  "proposal"
  "k4"
)

if [[ "${SLURM_ARRAY_TASK_ID:-}" == "" ]]; then
  CASE_ID=${CASE_ID:-0}
else
  CASE_ID=${SLURM_ARRAY_TASK_ID}
fi

if (( CASE_ID < 0 || CASE_ID >= ${#CASES[@]} )); then
  echo "CASE_ID ${CASE_ID} is out of range [0, $((${#CASES[@]} - 1))]."
  exit 1
fi

CASE_NAME=${CASES[CASE_ID]}
export OUTPUT_PATH="${OUTPUT_ROOT}/${CASE_NAME}"
export RUN_TAG="${CASE_NAME}_reason_mnt${MAX_NEW_TOKENS}_sr${STEP_RATIO}"

if [[ "${CASE_NAME}" == "base" ]]; then
  EFFECTIVE_DRAFT_STEPS=${BASE_DRAFT_STEPS}
  EFFECTIVE_POSTMASK_STEPS=0
  REFILL_GUIDANCE=none
  REFILL_GUIDANCE_STEPS=
elif [[ "${CASE_NAME}" == "proposal" ]]; then
  EFFECTIVE_DRAFT_STEPS=${DRAFT_STEPS}
  EFFECTIVE_POSTMASK_STEPS=${POSTMASK_STEPS}
  REFILL_GUIDANCE=none
  REFILL_GUIDANCE_STEPS=
elif [[ "${CASE_NAME}" == "k4" ]]; then
  EFFECTIVE_DRAFT_STEPS=${DRAFT_STEPS}
  EFFECTIVE_POSTMASK_STEPS=${POSTMASK_STEPS}
  REFILL_GUIDANCE=vcd
  REFILL_WEAK_VISUAL_MODE=diffusion_noise
  REFILL_GUIDANCE_STEPS=4
else
  echo "Unsupported case: ${CASE_NAME}"
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if [[ -n "${STEP_PER_BLOCK}" ]]; then
  EXTRA_ARGS+=(--step-per-block "${STEP_PER_BLOCK}")
else
  EXTRA_ARGS+=(--step-ratio "${STEP_RATIO}")
fi
if [[ -n "${REFILL_GUIDANCE_STEPS}" ]]; then
  EXTRA_ARGS+=(--refill-guidance-steps "${REFILL_GUIDANCE_STEPS}")
fi

echo "Running TextVQA ${CASE_NAME}:"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  OUTPUT_PATH=${OUTPUT_PATH}"
echo "  LIMIT=${LIMIT:-FULL}"
echo "  DRAFT_STEPS=${EFFECTIVE_DRAFT_STEPS}"
echo "  POSTMASK_STEPS=${EFFECTIVE_POSTMASK_STEPS}"
echo "  REMASK_SELECTION=proposal_confidence"
echo "  POSTMASK_MODE=fixed_set"
echo "  FIXED_SET_SIZE=${FIXED_SET_SIZE}"
echo "  FIXED_REFILL_PER_STEP=${FIXED_REFILL_PER_STEP}"
echo "  REFILL_GUIDANCE=${REFILL_GUIDANCE}"
echo "  REFILL_WEAK_VISUAL_MODE=${REFILL_WEAK_VISUAL_MODE:-none}"
echo "  REFILL_GUIDANCE_STEPS=${REFILL_GUIDANCE_STEPS:-none}"
echo "  VCD_REFILL_ALPHA=${VCD_REFILL_ALPHA}"
echo "  VCD_NOISE_STEP=${VCD_NOISE_STEP}"
echo "  VCD_NOISE_SEED=${VCD_NOISE_SEED}"

python VRG/run_textvqa_postmask.py \
  --pretrained "${MODEL_PATH}" \
  --prompt-mode "${TEXTVQA_PROMPT_MODE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --block-length "${BLOCK_LENGTH}" \
  --draft-steps "${EFFECTIVE_DRAFT_STEPS}" \
  --postmask-steps "${EFFECTIVE_POSTMASK_STEPS}" \
  --remask-per-step "${REMASK_PER_STEP}" \
  --remask-selection proposal_confidence \
  --postmask-mode fixed_set \
  --fixed-set-size "${FIXED_SET_SIZE}" \
  --fixed-refill-per-step "${FIXED_REFILL_PER_STEP}" \
  --null-visual-mode "${NULL_VISUAL_MODE}" \
  --refill-guidance "${REFILL_GUIDANCE}" \
  --refill-weak-visual-mode "${REFILL_WEAK_VISUAL_MODE:-diffusion_noise}" \
  --vcd-refill-alpha "${VCD_REFILL_ALPHA}" \
  --vcd-noise-step "${VCD_NOISE_STEP}" \
  --vcd-noise-seed "${VCD_NOISE_SEED}" \
  --output-dir "${OUTPUT_PATH}" \
  "${EXTRA_ARGS[@]}"

echo "TextVQA ${CASE_NAME} finished."
