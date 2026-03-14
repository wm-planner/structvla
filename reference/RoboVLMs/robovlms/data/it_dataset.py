import datasets
import random
import torch
import torchvision.transforms as T
from PIL import Image
from functools import partial
import numpy as np

from robovlms.data.data_utils import list_dir_with_cache
from robovlms.data.data_utils import b64_2_img
from robovlms.utils.model_utils import build_tokenizer


def _init_preprocess(
    input_size,
    img_mean=(0.48145466, 0.4578275, 0.40821073),
    img_std=(0.26862954, 0.26130258, 0.27577711),
    is_training=True,
):
    if is_training:
        static_preprocess = T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (input_size, input_size),
                    interpolation=Image.BICUBIC,
                    antialias=False,
                ),
                T.Normalize(img_mean, img_std),
            ]
        )
    else:
        static_preprocess = T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (input_size, input_size),
                    interpolation=Image.BICUBIC,
                    antialias=False,
                ),
                T.Normalize(img_mean, img_std),
            ]
        )

    return static_preprocess


def _generate_mim_attention(patch_num, ratio):
    mim_attention = np.ones((patch_num), dtype=bool)
    n_masked_tokens = int(ratio * patch_num)
    masked_inds = np.random.choice(np.arange(patch_num), n_masked_tokens, replace=False)
    mim_attention[masked_inds] = False
    return torch.from_numpy(mim_attention).long()


def batch_mapping(
    sample,
    tokenizer,
    preprocess,
    pad_token,
    text_seq_len,
    image_size,
    patch_num,
    is_training,
    use_mim_mask,
    vision_masked_ratio,
):
    # preprocess image
    image = b64_2_img(sample.get("image")).convert("RGB")
    image = preprocess(image)
    sample["rgb"] = image

    # tokenize text
    text = sample.get("caption")
    tokens = tokenizer.tokenize(text)
    tokenized_text_data = tokenizer.encode(tokens)
    token_tensor = torch.zeros(text_seq_len).long().fill_(pad_token)
    token_len = min(len(tokenized_text_data), text_seq_len)
    token_tensor[:token_len] = torch.tensor(tokenized_text_data[:token_len])
    sample["language"] = token_tensor

    # mim_attention
    if use_mim_mask and is_training:
        sample["mim_mask"] = _generate_mim_attention(patch_num, vision_masked_ratio)
    else:
        sample["mim_mask"] = None
    sample["data_type"] = "itpair"
    return sample


def ImageTextDataset(
    data_dir,
    tokenizer,
    text_seq_len=77,
    image_size=224,
    patch_num=10,
    is_training=True,
    use_mim_mask=False,
    vision_masked_ratio=0.8,
    seed=123,
    buffer_size=2000,
    **kwargs,
):
    tokenizer = build_tokenizer(tokenizer_config=tokenizer)
    pad_token = tokenizer.pad_token
    if pad_token is None:
        pad_token = 0
    else:
        pad_token = tokenizer.convert_tokens_to_ids(pad_token)
    preprocessor = _init_preprocess(image_size)
    map_func = partial(
        batch_mapping,
        tokenizer=tokenizer,
        preprocess=preprocessor,
        pad_token=pad_token,
        text_seq_len=text_seq_len,
        image_size=image_size,
        patch_num=patch_num,
        is_training=is_training,
        use_mim_mask=use_mim_mask,
        vision_masked_ratio=vision_masked_ratio,
    )

    # FIXME: hardcode for caching
    data_list = list_dir_with_cache(data_dir, cache_dir="cache")
    data_list = [f for f in data_list if f.endswith(".parquet")]
    random.shuffle(data_list)
    split = "train" if is_training else "test"
    ds = (
        datasets.load_dataset(
            "parquet", data_files=data_list, split=split, streaming=True
        )
        .shuffle(seed=seed, buffer_size=buffer_size)
        .map(map_func)
        .select_columns(["rgb", "language", "mim_mask", "data_type"])
    )
    return ds
