#!/bin/bash
#SBATCH --job-name=lavida_sink
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/data/jindong_gu/LaViDa/logs/sink/sink_%A_%a.log

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lavida

cd /data/jindong_gu/LaViDa
mkdir -p logs/sink

MODEL_PATH=${MODEL_PATH:-/data/jindong_gu/LaViDa/weight/lavida}
LLADA_VISION_ENCODER=${LLADA_VISION_ENCODER:-/data/jindong_gu/LaViDa/weight/siglip}
TASKS=${TASKS:-textvqa_val}
LIMIT=${LIMIT:-1000}
OUTPUT_PATH=${OUTPUT_PATH:-./logs/sink/}
SINK_SEED=${SINK_SEED:-0}
SINK_DEBUG=${SINK_DEBUG:-False}


CASES=(
  "baseline False none top_attn 1 last both text none all"
  "mask_dirsink_top2_layer15_decode True mask_attn direction_sink 2 15 decode text none all"
  "zero_k_dirsink_top2_layer15_decode True zero_k direction_sink 2 15 decode text none all"
  "zero_v_dirsink_top2_layer15_decode True zero_v direction_sink 2 15 decode text none all"
  "direction_debias_dirsink_top2_layer15_decode True direction_debias_1p0 direction_sink 2 15 decode text none all"
  "zero_v_random_top2_layer15_control True zero_v direction_sink 2 15 decode text random_visual all"

  # Previous 1000-sample validations:
  # baseline=0.4873, mask_topattn_both=0.4884, mask_topattn_decode=0.4884,
  # mask_topcos_both=0.4884, random_visual_control=0.4923.
  # zero_v_topattn_decode=0.4909 was the best hard K/V intervention.
  # direction_debias_0p5_topattn=0.4885, topcos=0.4875, random_control=0.4919.
  # direction_debias_1p0_topattn_decode=0.4895, 2p0_topattn_decode=0.4885,
  # 2p0_random_control_decode=0.4826.
  #
  # Mechanism attribution for decoder attention visual sinks.
  # Keep layer15, direction_sink top2, decode only fixed, then perturb:
  # attention mass (mask_attn), routing key (zero_k / direction_debias), and value content (zero_v).
  # zero_v_random_top2_layer15_control tests whether perturbing random visual V is similarly harmful/helpful.
)

if [[ "${SLURM_ARRAY_TASK_ID:-}" == "" ]]; then
  CASE_ID=${CASE_ID:-0}
else
  CASE_ID=${SLURM_ARRAY_TASK_ID}
fi

if (( CASE_ID < 0 || CASE_ID >= ${#CASES[@]} )); then
  echo "CASE_ID ${CASE_ID} is out of range [0, $((${#CASES[@]} - 1))]." >&2
  exit 1
fi

read -r CASE_NAME SINK_ENABLE SINK_INTERVENTION SINK_SELECTOR SINK_TOPK SINK_LAYERS SINK_STEPS SINK_QUERY_SCOPE SINK_CONTROL SINK_HEAD_SCOPE <<< "${CASES[CASE_ID]}"

RUN_TAG=${RUN_TAG:-sink_${CASE_NAME}_${TASKS}_n${LIMIT:-full}}
CASE_OUTPUT_PATH="${OUTPUT_PATH}/${CASE_NAME}"
mkdir -p "${CASE_OUTPUT_PATH}"

EXTRA_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi

export MODEL_PATH
export LLADA_VISION_ENCODER
export TASKS
export RUN_TAG
export SINK_ENABLE
export SINK_INTERVENTION
export SINK_SELECTOR
export SINK_TOPK
export SINK_LAYERS
export SINK_STEPS
export SINK_QUERY_SCOPE
export SINK_CONTROL
export SINK_HEAD_SCOPE
export SINK_SEED
export SINK_DEBUG

echo "Running sink case ${CASE_ID}:"
echo "  case_name=${CASE_NAME}"
echo "  tasks=${TASKS}"
echo "  limit=${LIMIT:-none}"
echo "  run_tag=${RUN_TAG}"
echo "  output_path=${CASE_OUTPUT_PATH}"
echo "  sink_enable=${SINK_ENABLE}"
echo "  intervention=${SINK_INTERVENTION}"
echo "  selector=${SINK_SELECTOR}"
echo "  topk=${SINK_TOPK}"
echo "  layers=${SINK_LAYERS}"
echo "  steps=${SINK_STEPS}"
echo "  query_scope=${SINK_QUERY_SCOPE}"
echo "  control=${SINK_CONTROL}"
echo "  head_scope=${SINK_HEAD_SCOPE}"

bash eval/run_sink.sh \
  --output_path "${CASE_OUTPUT_PATH}" \
  "${EXTRA_ARGS[@]}"

echo "Sink case ${CASE_ID} finished."
