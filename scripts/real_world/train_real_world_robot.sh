WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=
export CUDA_VISIBLE_DEVICES=
DATAPATH='/remote-home/share/jinminghao/real-data/meta/real_all_norm.pkl'
ACTION_TOKENIZER_PATH="/remote-home/jinminghao/structvla/pretrain/fast"
EXP_NAME="REAL_WORLD_FRANKA_PICK_PLACE_2K"
PRETRAIN="/remote-home/jinminghao/structvla/logs/STRUCTURED_PLANNER_REAL_DATA_2K/checkpoint-2000"
export PYTHONPATH=$(pwd)

export WANDB_MODE=offline
torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=${RANK} \
    train/train_moe.py \
    --model_name_or_path ${PRETRAIN} \
    --model_config_path /remote-home/jinminghao/structvla/configs/moe_fast_video.json \
    --deepspeed scripts/sft/zero3.json \
    --output_dir "logs/"${EXP_NAME} \
    --learning_rate 6e-5 \
    --null_prompt_prob 0.15 \
    --weight_decay 0.1 \
    --min_learning_rate 5e-6 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path ${DATAPATH} \
    --max_steps 2000 \
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 10 \
    --per_device_train_batch_size 4 \
    --frames 2 \
    --action_frames 5 \
    --max_position_embeddings 2800 \
    --seed 42 \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 2 \
    --save_strategy steps \
    --save_steps 1000 \
    --eval_strategy no \
    --apply_loss_on_only_vision False \
    --apply_loss_on_only_action True \
    --actions True \
    --actions_format "fast" \
    --real_robot False \
    --video_format "interleave" \
    --use_gripper True \
    --action_tokenizer_path ${ACTION_TOKENIZER_PATH} \
    --report_to "wandb" \
    --run_name ${EXP_NAME} \
    > /remote-home/jinminghao/structvla/output_real.log 2>&1


