# Pretrained Checkpoints

This directory is expected to contain the pretrained resources used by StructVLA.

## Required Downloads

### 1. Emu3 Base

- Hugging Face: <https://huggingface.co/BAAI/Emu3-Stage1>
- Local name used in this project: `Emu3-Base`

### 2. Emu3 Vision Tokenizer

- Hugging Face: <https://huggingface.co/BAAI/Emu3-VisionTokenizer>
- Local name used in this project: `Emu3-VisionVQ` or `Emu3-VisionTokenizer`

Note:
- The naming difference comes from modifications in the original baseline project.
- In this project, both names may appear in code or scripts. They refer to the same vision tokenizer resource from BAAI.

### 3. World Model Checkpoint

- Hugging Face: <https://huggingface.co/Yuqi1997/UniVLA/tree/main/WORLD_MODEL_POSTTRAIN>
- This is the world model checkpoint used in this project.

## Suggested Local Layout

After downloading, place the checkpoints under this directory with names consistent with the scripts, for example:

```text
pretrain/
├── Emu3-Base/
├── Emu3-VisionVQ/
└── WORLD_MODEL_POSTTRAIN/
```

If you choose different local folder names, update the corresponding script or config paths accordingly.
