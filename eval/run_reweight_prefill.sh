export LLADA_VISION_ENCODER=${LLADA_VISION_ENCODER:-"/data/jindong_gu/LaViDa/weight/siglip"}
MODEL_PATH=${MODEL_PATH:-"/data/jindong_gu/LaViDa/weight/lavida"}

set -x
export TASKS=${TASKS:-"textvqa_val"}
export CUDA_VISIBLE_DEVICES=0
export DEBUG_PRINT_IMAGE_RES=1
echo $TASKS
echo "Reweight prefill config: model=${MODEL_PATH} vision=${LLADA_VISION_ENCODER} gamma=${REWEIGHT_PREFILL_GAMMA:-1.2} enabled=${REWEIGHT_PREFILL_ENABLE:-True}"

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_llada_reweight_prefill \
    --model_args pretrained=${MODEL_PATH},conv_template=llada,model_name=llava_llada,reweight_prefill_enable=${REWEIGHT_PREFILL_ENABLE:-True},reweight_prefill_gamma=${REWEIGHT_PREFILL_GAMMA:-1.0} \
    --tasks $TASKS \
    --batch_size 1 \
    --gen_kwargs prefix_lm=True \
    --log_samples \
    --log_samples_suffix ${RUN_TAG:-llava_llada_reweight_prefill} \
    --output_path ./logs/slurm/ --verbosity=DEBUG \
    "$@"
