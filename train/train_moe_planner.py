import os
import os.path as osp
import torch
from dataclasses import dataclass, field
from typing import Optional, List
import pathlib
import transformers as tf
from datasets_class import Emu3SFTDataset
import sys
sys.path.append("/remote-home/jinminghao/structvla/reference/Emu3")
from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM, Emu3MoE, Emu3MoEConfig, Emu3Planner
from transformers import AutoModel,Trainer
from datasets_class import Emu3WorldModelDataset,Emu3RealRobotDataset,Emu3CoTDataset,Emu3PlannerDataset
from torch.utils.data import WeightedRandomSampler, DataLoader

# Added
from peft import LoraConfig, get_peft_model, PeftModel
import re


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
class LoraArguments:
    use_lora: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    # Try to cover Emu3 attention/MLP linear layers; fall back automatically if names do not match exactly
    lora_target: str = field(default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

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

    # Planner switches and alignment
    planner: bool = field(default=False)
    keystep_path: Optional[str] = field(default=None)   # CSV/JSON/PKL/TXT; CSV prefers triplets_manifest.csv
    keystep_key_from: str = field(default="index")      # 'index' | 'field' | 'stem'
    keystep_key_field: Optional[str] = field(default="episode_id")  # Field name used when key_from='field'
    planner_one_group_per_episode: bool = field(default=True)  # Sample only one keystep group per episode
    # Sampling window and fallback
    allow_short_context: bool = field(default=False)    # Allow short context when nearby keysteps make start<0
    fallback_gap_after_context: int = field(default=5)  # Fallback target = (c + (T-1) + gap), default 5

    supervise_context: bool = True          # Whether to label context segments as well
    ctx_loss_weight: float = 1.0            # Positional weight for context visual tokens
    keystep_loss_weight: float = 3.0        # Positional weight for target-frame visual tokens
    planner_overfit_first_n: Optional[int] = field(default=None) 
    planner_expand_by_offset: bool = True          
    max_groups_per_keystep: int = 5            
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
def _auto_pick_target_modules(model, candidates):
    """
    Automatically select the Linear layer names that actually exist from the candidate list.
    Fall back to all Linear layers if nothing matches.
    """
    found = []
    names = [n for n, _ in model.named_modules()]
    for c in candidates:
        # Exact or suffix match (for llama-like structures)
        patt = re.compile(rf"(^|\.){re.escape(c)}$")
        hit = any(patt.search(n) for n in names)
        if hit:
            found.append(c)
    if len(found) == 0:
        # Fallback strategy: all linear layers
        return "all-linear"
    return ",".join(sorted(set(found)))

def attach_lora(model, lora_args: "LoraArguments"):
    if not lora_args.use_lora:
        return model

    # Automatically detect target_modules, aiming to hit q/k/v/o_proj and up/down/gate_proj
    requested = [s.strip() for s in lora_args.lora_target.split(",") if s.strip()]
    target = _auto_pick_target_modules(model, requested)

    # if target == "all-linear":
    #     lconf = LoraConfig(
    #         r=lora_args.lora_r, lora_alpha=lora_args.lora_alpha,
    #         lora_dropout=lora_args.lora_dropout,
    #         target_modules=None,   # None = all Linear layers
    #         bias="none", task_type="CAUSAL_LM"
    #     )
    # else:
    #     lconf = LoraConfig(
    #         r=lora_args.lora_r, lora_alpha=lora_args.lora_alpha,
    #         lora_dropout=lora_args.lora_dropout,
    #         target_modules=target.split(","),
    #         bias="none", task_type="CAUSAL_LM"
    #     )
    # More generic: use the suffix list that was just detected
    lconf = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lconf)
    try:
        model.enable_input_require_grads()
    except AttributeError:
        # Fallback for older peft versions: hook the embedding output
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        def _make_inputs_require_grad(module, inputs, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)
        base.model.embed_tokens.register_forward_hook(_make_inputs_require_grad)
    # Training stability: disable cache to avoid checkpoint warnings
    try:
        model.base_model.model.config.use_cache = False
    except Exception:
        try:
            model.model.config.use_cache = False
        except Exception:
            pass

    # Print trainable parameters to confirm only LoRA is trained
    try:
        model.print_trainable_parameters()
    except Exception:
        # Compatible with older peft versions
        trainable, total = 0, 0
        for n, p in model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        pct = 100.0 * trainable / max(1, total)
        print(f"[LoRA] trainable params: {trainable}/{total} ({pct:.2f}%)", flush=True)

    return model

def load_model(model_args, model_config, training_args, lora_args):
    if training_args.from_scratch:
        model_config.torch_dtype = torch.bfloat16 if training_args.bf16 else None
        model_config.attn_implementation = "flash_attention_2" if training_args.attn_type == "fa2" else None
        base = Emu3Planner(config=model_config)
    else:
        base = Emu3Planner.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            attn_implementation="flash_attention_2" if training_args.attn_type == "fa2" else None,
            torch_dtype=torch.bfloat16 if training_args.bf16 else None,
        )
    # Inject LoRA if enabled
    model = attach_lora(base, lora_args)
    # Ensure cache stays disabled when gradient checkpointing is used
    try:
        model.config.use_cache = False
    except Exception:
        pass
    return model


def get_dataset(data_args, tokenizer):
    """
    Initialize and return the training dataset.
    """
    if data_args.post_training:
        if getattr(data_args, "planner", False):
            return Emu3PlannerDataset(data_args, tokenizer=tokenizer)   # <-- NEW
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
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

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
    model = load_model(model_args, model_config, training_args, lora_args)


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
    # 2) If LoRA is used, save two extra copies:
    #    - adapter version (smallest size)
    #    - merged version (LoRA merged into the base model for direct inference)
    try:
        from peft import PeftModel
        is_main = (not hasattr(trainer.args, "local_rank")) or (trainer.args.local_rank in (-1, 0))
        if is_main and isinstance(trainer.model, PeftModel):
            adapter_dir = osp.join(training_args.output_dir, "lora_adapter")
            merged_dir  = osp.join(training_args.output_dir, "merged")

            # Save adapter only (lightweight)
            trainer.model.save_pretrained(adapter_dir)

            # Merge LoRA into the base model and save the full model for direct inference
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)

            print(f"[Save] LoRA adapter -> {adapter_dir}", flush=True)
            print(f"[Save] Merged full model -> {merged_dir}", flush=True)
    except Exception as e:
        print(f"[Save][WARN] LoRA extra saving skipped: {e}", flush=True)

if __name__ == "__main__":
    train()
