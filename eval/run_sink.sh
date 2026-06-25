export LLADA_VISION_ENCODER=${LLADA_VISION_ENCODER:-"/data/jindong_gu/LaViDa/weight/siglip"}
MODEL_PATH=${MODEL_PATH:-"/data/jindong_gu/LaViDa/weight/lavida"}

set -x
export TASKS=${TASKS:-"textvqa_val"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DEBUG_PRINT_IMAGE_RES=${DEBUG_PRINT_IMAGE_RES:-1}
echo $TASKS
echo "Sink config: model=${MODEL_PATH} vision=${LLADA_VISION_ENCODER} intervention=${SINK_INTERVENTION:-mask_attn} selector=${SINK_SELECTOR:-top_attn} topk=${SINK_TOPK:-1} layers=${SINK_LAYERS:-last} steps=${SINK_STEPS:-both} query_scope=${SINK_QUERY_SCOPE:-text} control=${SINK_CONTROL:-none} head_scope=${SINK_HEAD_SCOPE:-all}"

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_llada_sink \
    --model_args pretrained=${MODEL_PATH},conv_template=llada,model_name=llava_llada,sink_enable=${SINK_ENABLE:-True},sink_intervention=${SINK_INTERVENTION:-mask_attn},sink_selector=${SINK_SELECTOR:-top_attn},sink_topk=${SINK_TOPK:-1},sink_layers=${SINK_LAYERS:-last},sink_steps=${SINK_STEPS:-both},sink_query_scope=${SINK_QUERY_SCOPE:-text},sink_control=${SINK_CONTROL:-none},sink_head_scope=${SINK_HEAD_SCOPE:-all},sink_seed=${SINK_SEED:-0},sink_debug=${SINK_DEBUG:-False} \
    --tasks $TASKS \
    --batch_size 1 \
    --gen_kwargs prefix_lm=True \
    --log_samples \
    --log_samples_suffix ${RUN_TAG:-llava_llada_sink} \
    --output_path ./logs/ --verbosity=DEBUG \
    "$@"
