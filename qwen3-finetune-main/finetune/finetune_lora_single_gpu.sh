#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

#MODEL="Qwen/Qwen-7B" # Set the path if you do not want to load from huggingface directly
MODEL="C:\\software\\modelscope\\hub\\models\\qwen\\Qwen3-0.6B"
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="C:\\Users\\ouyangboyu\\Desktop\\qwen3-finetune-main\\fold2.json"

#function usage() {
#    echo '
#Usage: bash finetune/finetune_lora_single_gpu.sh [-m MODEL_PATH] [-d DATA_PATH]
#'
#}
#
#while [[ "$1" != "" ]]; do
#    case $1 in
#        -m | --model )
#            shift
#            MODEL=$1
#            ;;
#        -d | --data )
#            shift
#            DATA=$1
#            ;;
#        -h | --help )
#            usage
#            exit 0
#            ;;
#        * )
#            echo "Unknown argument ${1}"
#            exit 1
#            ;;
#    esac
#    shift
#done
#
#export CUDA_VISIBLE_DEVICES=0
#
#python finetune.py \
#  --model_name_or_path $MODEL \
#  --data_path $DATA \
#  --bf16 True \
#  --output_dir output_qwen \
#  --num_train_epochs 5 \
#  --per_device_train_batch_size 2 \
#  --per_device_eval_batch_size 1 \
#  --gradient_accumulation_steps 8 \
#  --evaluation_strategy "no" \
#  --save_strategy "steps" \
#  --save_steps 1000 \
#  --save_total_limit 10 \
#  --learning_rate 3e-4 \
#  --weight_decay 0.1 \
#  --adam_beta2 0.95 \
#  --warmup_ratio 0.01 \
#  --lr_scheduler_type "cosine" \
#  --logging_steps 1 \
#  --report_to "none" \
#  --model_max_length 512 \
#  --lazy_preprocess True \
#  --gradient_checkpointing \
#  --use_lora
#export CUDA_VISIBLE_DEVICES=0

#CUDA_VISIBLE_DEVICES=0 \
#swift sft \
#    --model $MODEL \
#    --train_type lora \
#    --dataset $DATA \
#    --torch_dtype bfloat16 \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size 2 \
#    --per_device_eval_batch_size 1 \
#    --learning_rate 1e-4 \
#    --lora_rank 8 \
#    --lora_alpha 32 \
#    --target_modules all-linear \
#    --gradient_accumulation_steps 16 \
#    --eval_steps 50 \
#    --save_steps 50 \
#    --save_total_limit 2 \
#    --logging_steps 5 \
#    --max_length 2048 \
#    --output_dir output \
#    --system 'You are a helpful assistant.' \
#    --warmup_ratio 0.05 \
#    --dataloader_num_workers 4 \
#    --model_author swift \
#    --model_name swift-robot \


CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model $MODEL \
    --train_type lora \
    --dataset $DATA \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 131072 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \

    #    --lora_lr_ratio 16 \
