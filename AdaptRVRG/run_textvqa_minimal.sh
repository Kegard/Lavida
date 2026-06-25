#!/bin/bash
set -euo pipefail

cd /data/jindong_gu/LaViDa

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATASET_PATH=${DATASET_PATH:-lmms-lab/textvqa}
SPLIT=${SPLIT:-validation}
PRETRAINED=${PRETRAINED:-weight/lavida}
VISION_TOWER=${VISION_TOWER:-weight/siglip}
LIMIT=${LIMIT:-1000}
START_INDEX=${START_INDEX:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
STEP_RATIO=${STEP_RATIO:-1.0}
CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.7}
GUIDE_LAMBDA=${GUIDE_LAMBDA:-0.2}
REGIONAL_TAU=${REGIONAL_TAU:-1.0}
TOP_K=${TOP_K:-100}
BLUR_RADIUS=${BLUR_RADIUS:-10.0}
METHODS=${METHODS:-native,global,regional_weighted}
OUT_ROOT=${OUT_ROOT:-AdaptRVRG/outputs}
RUN_TAG=${RUN_TAG:-textvqa_minimal_n${LIMIT}_lambda${GUIDE_LAMBDA}_thr${CONFIDENCE_THRESHOLD}}
OUTPUT_DIR=${OUTPUT_DIR:-${OUT_ROOT}/${RUN_TAG}}

python AdaptRVRG/run_textvqa_adapt_rvrg.py \
  --dataset-path "${DATASET_PATH}" \
  --split "${SPLIT}" \
  --start-index "${START_INDEX}" \
  --limit "${LIMIT}" \
  --pretrained "${PRETRAINED}" \
  --vision-tower "${VISION_TOWER}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --block-length "${BLOCK_LENGTH}" \
  --step-ratio "${STEP_RATIO}" \
  --methods "${METHODS}" \
  --confidence-threshold "${CONFIDENCE_THRESHOLD}" \
  --guide-lambda "${GUIDE_LAMBDA}" \
  --regional-tau "${REGIONAL_TAU}" \
  --top-k "${TOP_K}" \
  --blur-radius "${BLUR_RADIUS}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
