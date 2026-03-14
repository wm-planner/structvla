#!/usr/bin/env bash
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=
export CUDA_VISIBLE_DEVICES=

DATAPATH='/remote-home/jinminghao/structvla/datasets/processed_data/meta/simplerenv_bridge_trainval.pkl'
ACTION_TOKENIZER_PATH="/remote-home/jinminghao/structvla/pretrain/fast_bridge_t5_s50"
EXP_NAME="STRUCTVLA_SIMPLER_20K"

export PYTHONPATH=$(pwd)
export PYTHONPATH=$(pwd)
# Run wandb in offline mode and write locally only
export WANDB_MODE=offline

# Local log directory; keep it consistent with your output_dir
export WANDB_DIR="logs/${EXP_NAME}/wandb"

# Optional: project name and run name
export WANDB_PROJECT="structvla"
export WANDB_RUN_GROUP="${EXP_NAME}"

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=${RANK} \
    train/train_moe.py \
    --model_name_or_path /remote-home/jinminghao/structvla/logs/STRUCTURED_PLANNER_SIMPLER_30K/merged \
    --model_config_path /remote-home/jinminghao/structvla/configs/moe_fast_video.json \
    --deepspeed scripts/sft/zero3.json \
    --output_dir "logs/"${EXP_NAME} \
    --learning_rate 8e-5 \
    --null_prompt_prob 0.15 \
    --weight_decay 0.1 \
    --min_learning_rate 5e-6 \
    --max_grad_norm 2.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path ${DATAPATH} \
    --max_steps 20000 \
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 500 \
    --per_device_train_batch_size 4 \
    --frames 2 \
    --action_frames 5 \
    --max_position_embeddings 2400 \
    --seed 42 \
    --logging_steps 20 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 4 \
    --save_strategy steps \
    --save_steps 2000 \
    --eval_strategy no \
    --apply_loss_on_only_vision False \
    --apply_loss_on_only_action True \
    --actions True \
    --actions_format "fast" \
    --use_gripper False \
    --video_format "interleave" \
    --action_tokenizer_path ${ACTION_TOKENIZER_PATH} \
    --report_to "wandb" \
    --run_name ${EXP_NAME} \
    > /remote-home/jinminghao/structvla/simpler.log 2>&1
