import os
import os.path as osp
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import pathlib
import transformers as tf
from datasets_class import Emu3SFTDataset
import sys
sys.path.append("/remote-home/jinminghao/structvla/reference/Emu3")
from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM, Emu3MoE, Emu3MoEConfig
from transformers import Trainer
from datasets_class import Emu3WorldModelDataset,Emu3RealRobotDataset,Emu3CoTDataset
from torch.utils.data import WeightedRandomSampler, DataLoader

# --- debug hook: enable when DEBUG=1 ---

if os.getenv("DEBUG", "0") == "1":
    try:
        import debugpy
        # DDP-compatible: let each rank listen on a different port
        rank = int(os.getenv("LOCAL_RANK", os.getenv("RANK", 0)))
        base = int(os.getenv("DEBUG_BASE_PORT", "5678"))
        port = base + rank
        debugpy.listen(("0.0.0.0", port))
        print(f"[debugpy] listening on {port} (rank={rank})", flush=True)

        if os.getenv("DEBUG_WAIT", "1") == "1":
            print("[debugpy] waiting for client...", flush=True)
            debugpy.wait_for_client()  # Continue only after VS Code attaches
    except Exception as e:
        print(f"[debugpy] setup failed: {e}", flush=True)
# --- end debug hook ---
class WeightedSamplerTrainer(Trainer):
    def get_train_dataloader(self):
        # Assuming train_dataset has a sample_weights attribute
        sample_weights = torch.tensor(
            self.train_dataset.sample_weights, dtype=torch.double
        )

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Gen")
    model_config_path: Optional[str] = field(default="pretrain/Emu3-Base")

@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    null_prompt_prob: float = field(default=0.05)
    apply_loss_on_only_vision: bool = field(default=True)
    apply_loss_on_only_text: bool = field(default=False)
    apply_loss_on_only_action: bool = field(default=False) 
    ignore_index: int = field(default=-100)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: Optional[int] = field(default=32768)
    frames: int = field(default=4)
    VL: bool = field(default=False)
    actions: bool = field(default=False)
    actions_format: str = field(default="openvla")
    action_frames: int = field(default=8)
    use_gripper: bool = field(default=False)
    action_tokenizer_path: Optional[str] = field(default=None)
    video_format: str = field(default=None)
    random_frame_sampling: bool = field(default=True)
    raw_image: bool = field(default=False)
    post_training: bool = field(default=False)
    datasets_weight: bool = field(default=False)
    without_text: bool = field(default=False)
    real_robot: bool = field(default=False)
    with_cot: bool = field(default=False)

@dataclass
class TrainingArguments(tf.TrainingArguments):
    report_to: List[str] = field(default_factory=list)
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    attn_type: Optional[str] = field(default="fa2")
    image_area: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)
    from_scratch: bool = field(default=False)
    dataloader_num_workers: Optional[int] = field(default=0)

def load_model(model_args, model_config, training_args):
    """
    Load model based on whether to train from scratch or fine-tune from a pre-trained model.
    """
    if training_args.from_scratch:
        model_config.torch_dtype = torch.bfloat16 if training_args.bf16 else None
        model_config.attn_implementation = "flash_attention_2" if training_args.attn_type == "fa2" else None
        return Emu3MoE(config=model_config)
    else:
        return Emu3MoE.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            attn_implementation="flash_attention_2" if training_args.attn_type == "fa2" else None,
            torch_dtype=torch.bfloat16 if training_args.bf16 else None,
        )

def get_dataset(data_args, tokenizer):
    """
    Initialize and return the training dataset.
    """
    if data_args.post_training:
        return Emu3WorldModelDataset(data_args, tokenizer=tokenizer)
        # return Emu3SFTDataset(data_args, tokenizer=tokenizer)
    elif data_args.real_robot:
        return Emu3RealRobotDataset(data_args, tokenizer=tokenizer)
    elif data_args.with_cot:
        return Emu3CoTDataset(data_args, tokenizer=tokenizer)
    return Emu3SFTDataset(data_args, tokenizer=tokenizer)

def get_dataset_split(data_args, tokenizer):
    """
    Initialize and return the training dataset.
    """
    if data_args.post_training:
        full_dataset = Emu3WorldModelDataset(data_args, tokenizer=tokenizer)
    else:
        full_dataset = Emu3SFTDataset(data_args, tokenizer=tokenizer)
    # Automatically split 90% train and 10% val
    split = full_dataset.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]

def update_configs(model_config, args, fields):
    cross_update = lambda a, b, field_name: (
        setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None else
        setattr(a, field_name, getattr(b, field_name))
    )

    for f in fields:
        cross_update(model_config, args, f)

def train():
    """
    Main function to train the model.
    """
    # Parse arguments
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set environment variable for WANDB logging
    os.environ["WANDB_DIR"] = osp.join(training_args.output_dir, "wandb")

    # Load model configuration and tokenizer
    model_config = Emu3MoEConfig.from_pretrained(model_args.model_config_path)
    update_configs(model_config, training_args, ["image_area", "max_position_embeddings"])
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate
    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )

    # Initialize model
    model = load_model(model_args, model_config, training_args)

    # Initialize dataset
    train_dataset = get_dataset(data_args, tokenizer)

    if data_args.datasets_weight:
        trainer = WeightedSamplerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset, 
            tokenizer=tokenizer,
        )
    else:
        # Setup Trainer
        trainer = tf.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,  # Pass tokenizer to trainer
        )

    # Check if resuming from checkpoint
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model and training state
    trainer.save_state()
    torch.cuda.synchronize()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()
