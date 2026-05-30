export LLADA_VISION_ENCODER=${LLADA_VISION_ENCODER:-"/data/jindong_gu/LaViDa/weight/siglip"}
MODEL_PATH=${MODEL_PATH:-"/data/jindong_gu/LaViDa/weight/lavida-reason"}
OUTPUT_PATH=${OUTPUT_PATH:-"./logs/"}
PROPOSAL_REFINE_ENABLE=${PROPOSAL_REFINE_ENABLE:-True}
REMASK_POLICY=${REMASK_POLICY:-confidence}
NULL_VISUAL_MODE=${NULL_VISUAL_MODE:-zeros}
TEXTVQA_PROMPT_MODE=${TEXTVQA_PROMPT_MODE:-reasoning}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
BLOCK_LENGTH=${BLOCK_LENGTH:-}
STEP_PER_BLOCK=${STEP_PER_BLOCK:-}
STEP_RATIO=${STEP_RATIO:-}

set -x
export TASKS=${TASKS:-"textvqa_val"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo $TASKS
echo "Proposal-refine config: model=${MODEL_PATH} proposal_refine_enable=${PROPOSAL_REFINE_ENABLE} proposal_step=${PROPOSAL_STEP:-8} remask_ratio=${PROPOSAL_REMASK_RATIO:-0.25} late_refine_steps=${LATE_REFINE_STEPS:-8} remask_policy=${REMASK_POLICY} null_visual_mode=${NULL_VISUAL_MODE}"
echo "TextVQA prompt mode: ${TEXTVQA_PROMPT_MODE}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Block length: ${BLOCK_LENGTH:-AUTO}"
echo "Step per block: ${STEP_PER_BLOCK:-AUTO}"
echo "Step ratio: ${STEP_RATIO:-AUTO}"
echo "Output path: ${OUTPUT_PATH}"

GEN_KWARGS="prefix_lm=True,max_new_tokens=${MAX_NEW_TOKENS}"
if [[ -n "${BLOCK_LENGTH}" ]]; then
  GEN_KWARGS+=",block_length=${BLOCK_LENGTH}"
fi
if [[ -n "${STEP_PER_BLOCK}" ]]; then
  GEN_KWARGS+=",step_per_block=${STEP_PER_BLOCK}"
fi
if [[ -n "${STEP_RATIO}" ]]; then
  GEN_KWARGS+=",step_ratio=${STEP_RATIO}"
fi

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_llada_proposal_refine \
    --model_args pretrained=${MODEL_PATH},conv_template=llada,model_name=llava_llada,proposal_refine_enable=${PROPOSAL_REFINE_ENABLE},proposal_step=${PROPOSAL_STEP:-8},proposal_remask_ratio=${PROPOSAL_REMASK_RATIO:-0.25},late_refine_steps=${LATE_REFINE_STEPS:-8},remask_policy=${REMASK_POLICY},null_visual_mode=${NULL_VISUAL_MODE} \
    --tasks $TASKS \
    --batch_size 1 \
    --gen_kwargs "${GEN_KWARGS}" \
    --log_samples \
    --log_samples_suffix ${RUN_TAG:-llava_llada_proposal_refine} \
    --output_path "${OUTPUT_PATH}" --verbosity=DEBUG \
    "$@"
