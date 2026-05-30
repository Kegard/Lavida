
export LLADA_VISION_ENCODER=${LLADA_VISION_ENCODER:-"/data/jindong_gu/LaViDa/weight/siglip"}
MODEL_PATH=${MODEL_PATH:-"/data/jindong_gu/LaViDa/weight/lavida"}

set -x
export TASKS=${TASKS:-"textvqa_val,mmmu_val"}
export CUDA_VISIBLE_DEVICES=0
export DEBUG_PRINT_IMAGE_RES=1
echo $TASKS
echo "Scale config: model=${MODEL_PATH} vision=${LLADA_VISION_ENCODER} gamma_prompt=${TACA_GAMMA_PROMPT:-1.5} gamma_visual=${TACA_GAMMA_VISUAL:-1.0} scale_generated=${TACA_SCALE_GENERATED:-1.0}"

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_llada_scale \
    --model_args pretrained=${MODEL_PATH},conv_template=llada,model_name=llava_llada,taca_enable=True,taca_gamma_prompt=${TACA_GAMMA_PROMPT:-1.5},taca_gamma_visual=${TACA_GAMMA_VISUAL:-1.0},taca_scale_generated=${TACA_SCALE_GENERATED:-1.0} \
    --tasks $TASKS \
    --batch_size 1 \
    --gen_kwargs prefix_lm=True \
    --log_samples \
    --log_samples_suffix llava_llada_scale \
    --output_path ./logs/ --verbosity=DEBUG \
    $@ \
