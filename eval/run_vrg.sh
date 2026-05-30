export LLADA_VISION_ENCODER=${LLADA_VISION_ENCODER:-"/data/jindong_gu/LaViDa/weight/siglip"}
MODEL_PATH=${MODEL_PATH:-"/data/jindong_gu/LaViDa/weight/lavida"}

set -x
export TASKS=${TASKS:-"textvqa_val"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DEBUG_PRINT_IMAGE_RES=${DEBUG_PRINT_IMAGE_RES:-1}
echo $TASKS
echo "VRG config: model=${MODEL_PATH} vision=${LLADA_VISION_ENCODER} alpha_start=${VRG_ALPHA_START:-0.0} alpha_end=${VRG_ALPHA_END:-2.0} alpha_schedule=${VRG_ALPHA_SCHEDULE:-linear} alpha_power=${VRG_ALPHA_POWER:-2.0} null_visual_mode=${VRG_NULL_VISUAL_MODE:-zeros}"

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_llada_vrg \
    --model_args pretrained=${MODEL_PATH},conv_template=llada,model_name=llava_llada,vrg_enable=True,vrg_alpha_start=${VRG_ALPHA_START:-0.0},vrg_alpha_end=${VRG_ALPHA_END:-2.0},vrg_alpha_schedule=${VRG_ALPHA_SCHEDULE:-linear},vrg_alpha_power=${VRG_ALPHA_POWER:-2.0},vrg_null_visual_mode=${VRG_NULL_VISUAL_MODE:-zeros} \
    --tasks $TASKS \
    --batch_size 1 \
    --gen_kwargs prefix_lm=True \
    --log_samples \
    --log_samples_suffix ${RUN_TAG:-llava_llada_vrg} \
    --output_path ./logs/ --verbosity=DEBUG \
    "$@"
