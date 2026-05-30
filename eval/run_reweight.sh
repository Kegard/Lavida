
export LLADA_VISION_ENCODER=${LLADA_VISION_ENCODER:-"/data/jindong_gu/LaViDa/weight/siglip"}
MODEL_PATH=${MODEL_PATH:-"/data/jindong_gu/LaViDa/weight/lavida"}

set -x
export TASKS=${TASKS:-"textvqa_val"}
export CUDA_VISIBLE_DEVICES=0
export DEBUG_PRINT_IMAGE_RES=1
echo $TASKS


accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_llada_reweight \
    --model_args pretrained=${MODEL_PATH},conv_template=llada,model_name=llava_llada,reweight_enable=True,reweight_alpha_prompt=${REWEIGHT_ALPHA_PROMPT:-1.0},reweight_alpha_visual=${REWEIGHT_ALPHA_VISUAL:-1.01},reweight_alpha_generated=${REWEIGHT_ALPHA_GENERATED:-1.0},reweight_alpha_mask=${REWEIGHT_ALPHA_MASK:-${REWEIGHT_ALPHA_GENERATED:-1.0}},reweight_alpha_normal=${REWEIGHT_ALPHA_NORMAL:-${REWEIGHT_ALPHA_GENERATED:-1.0}},reweight_alpha_special=${REWEIGHT_ALPHA_SPECIAL:-${REWEIGHT_ALPHA_GENERATED:-1.0}} \
    --tasks $TASKS \
    --batch_size 1 \
    --gen_kwargs prefix_lm=True \
    --log_samples \
    --log_samples_suffix ${RUN_TAG:-llava_llada_reweight} \
    --output_path ./logs/slurm/ --verbosity=DEBUG \
    "$@"
