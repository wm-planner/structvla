import transformers
import copy

FLAMINGO_MODEL_CONFIGS = {
    "mpt_1b": {
        "lang_encoder_path": "VLMs/mpt-1b-redpajama-200b",
        "tokenizer_path": "VLMs/mpt-1b-redpajama-200b",
        "cross_attn_every_n_layers": 1,
        "openflamingo_checkpoint": "VLMs/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt",
    },
    "mpt_1b_ift": {
        "lang_encoder_path": "VLMs/mpt-1b-redpajama-200b-dolly",
        "tokenizer_path": "VLMs/mpt-1b-redpajama-200b-dolly",
        "cross_attn_every_n_layers": 1,
        "openflamingo_checkpoint": "VLMs/OpenFlamingo-3B-vitl-mpt1b-langinstruct/checkpoint.pt",
    },
    "mpt_3b": {
        "lang_encoder_path": "VLMs/RedPajama-INCITE-Base-3B-v1",
        "tokenizer_path": "VLMs/RedPajama-INCITE-Base-3B-v1",
        "cross_attn_every_n_layers": 2,
        "openflamingo_checkpoint": "VLMs/OpenFlamingo-4B-vitl-rpj3b/checkpoint.pt",
    },
    "mpt_3b_ift": {
        "lang_encoder_path": "VLMs/RedPajama-INCITE-Instruct-3B-v1",
        "tokenizer_path": "VLMs/RedPajama-INCITE-Instruct-3B-v1",
        "cross_attn_every_n_layers": 2,
        "openflamingo_checkpoint": "VLMs/OpenFlamingo-4B-vitl-rpj3b-langinstruct/checkpoint.pt",
    },
    "mpt_7b": {
        "lang_encoder_path": "VLMs/mpt-7b",
        "tokenizer_path": "VLMs/mpt-7b",
        "cross_attn_every_n_layers": 4,
        "openflamingo_checkpoint": "VLMs/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt",
    },
    "llama_7b": {
        "lang_encoder_path": "VLMs/.cache/llama-7b-hf-jxu124",
        "tokenizer_path": "VLMs/.cache/llama-7b-hf-jxu124",
        "cross_attn_every_n_layers": 4,
        "openflamingo_checkpoint": "VLMs/.cache/OpenFlamingo-9B/checkpoint.pt",
    },
}

LLM_CFG = {
    "MPT1b": {
        "type": "AutoModelForCausalLM",
        "pretrained_model_name_or_path": "VLMs/mpt-1b-redpajama-200b",
    },
    "MPT1bIFT": {
        "type": "AutoModelForCausalLM",
        "pretrained_model_name_or_path": "VLMs/mpt-1b-redpajama-200b-dolly",
    },
    "MPT3b": {
        "type": "AutoModelForCausalLM",
        "pretrained_model_name_or_path": "VLMs/RedPajama-INCITE-Base-3B-v1",
    },
    "MPT3bIFT": {
        "type": "AutoModelForCausalLM",
        "pretrained_model_name_or_path": "VLMs/RedPajama-INCITE-Instruct-3B-v1",
    },
    "MPT7b": {
        "type": "AutoModelForCausalLM",
        "pretrained_model_name_or_path": "VLMs/mpt-7b",
    },
    "LLaMA7b": {
        "type": "AutoModelForCausalLM",
        "pretrained_model_name_or_path": "VLMs/.cache/llama-7b-hf-jxu124",
    },
}


def default_llm_flamingo_config(llm_name):
    llm_cfg = {}

    llm_cfg["mpt_1b"] = LLM_CFG["MPT1b"]
    llm_cfg["mpt_1b_ift"] = LLM_CFG["MPT1bIFT"]

    llm_cfg["mpt_3b"] = LLM_CFG["MPT3b"]
    llm_cfg["mpt_3b_ift"] = LLM_CFG["MPT3bIFT"]

    llm_cfg["mpt_7b"] = LLM_CFG["MPT7b"]

    llm_cfg["llama_7b"] = LLM_CFG["LLaMA7b"]

    assert llm_name in llm_cfg, f"Unknown llm_name: {llm_name}"
    return llm_cfg[llm_name]


def build_llm_flamingo(llm_config):
    if isinstance(llm_config, str):
        llm_config = default_llm_flamingo_config(llm_config)

    llm_config = copy.deepcopy(llm_config)
    llm_type = llm_config.pop("type")
    # llm_path = os.path.join('pretrained/llms', llm_type)
    llm_path = llm_config.pop("pretrained_model_name_or_path")

    llm = getattr(transformers, llm_type).from_pretrained(
        llm_path, local_files_only=False, trust_remote_code=True
    )

    return llm
