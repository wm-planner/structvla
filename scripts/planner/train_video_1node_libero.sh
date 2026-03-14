WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23457}
NGPUS=
export CUDA_VISIBLE_DEVICES=

DATAPATH='/remote-home/jinminghao/structvla/datasets/processed_data/meta/libero_all_norm.pkl'
ACTION_TOKENIZER_PATH="/remote-home/jinminghao/structvla/pretrain/fast"
EXP_NAME="STRUCTURED_PLANNER_LIBERO_10K"

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
    --nnodes=${WORLD_SIZE} \
    --master_port=${MASTER_PORT} \
    --master_addr=${MASTER_ADDR} \
    --node_rank=${RANK} \
    train/train_moe_planner.py \
    --model_name_or_path /remote-home/jinminghao/structvla/experiments/ckpts/WORLD_MODEL_POSTTRAIN \
    --model_config_path /remote-home/jinminghao/structvla/configs/moe_fast_video_pretrain.json \
    --ddp_find_unused_parameters False \
    --output_dir "logs/"${EXP_NAME} \
    --learning_rate 2e-4 \
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
    --max_steps 10000 \
    --dataloader_num_workers 12 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 50 \
    --per_device_train_batch_size 2 \
    --frames 4 \
    --action_frames 10 \
    --max_position_embeddings 2800 \
    --seed 42 \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 4 \
    --save_strategy steps \
    --save_steps 2000 \
    --eval_strategy no \
    --apply_loss_on_only_vision True \
    --apply_loss_on_only_action False \
    --actions False \
    --use_gripper False \
    --video_format "interleave" \
    --post_training True \
    --report_to "wandb" \
    --run_name ${EXP_NAME} \
    --planner True \
    --keystep_path "/remote-home/jinminghao/structvla/documents/libero_keysteps/triplets_manifest.csv" \
    --keystep_key_from "index" \
    --allow_short_context False \
    --fallback_gap_after_context 5 \
    --use_lora True \
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --supervise_context False \
    --planner_expand_by_offset True \
    --max_groups_per_keystep 10 \
    > /remote-home/jinminghao/structvla/outputs.log 2>&1
