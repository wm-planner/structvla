# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from turtle import pd
from typing import Dict, Optional, Sequence, List
import torch
import transformers
from robovlms.data.data_utils import (
    get_llava_image_processor,
    get_llava_video_processor,
)

from robovlms.data.vid_llava_constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN,
    MAX_IMAGE_LENGTH,
    MAX_VIDEO_LENGTH,
)
from torch.utils.data import Dataset

import robovlms.data.conversation as conversation_lib
from robovlms.data.vid_llava_mm_utils import tokenizer_image_token

from PIL import Image
from robovlms.data.data_utils import order_pick_k
from robovlms.utils.model_utils import build_tokenizer
from robovlms.data.data_utils import get_text_function


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = "square"
    # ===================================================================
    data_path: Optional[List[str]] = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    num_frames: int = 8
    # ===================================================================


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # ======================================================================================================
            if sentence["value"].startswith(DEFAULT_IMAGE_TOKEN) or sentence[
                "value"
            ].startswith(
                DEFAULT_VIDEO_TOKEN
            ):  # run with multi-im, multi-vid, multi-im & multi-vid
                # <video><video><image><image>\nxxxxxxxxxxxxx  # must <video> first
                # <image>\nxxxxxxxxxxxxx -> <image>\nxxxxxxxxxxxxx
                # <video>\nxxxxxxxxxxxxx -> <video>\nxxxxxxxxxxxxx

                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )

                IMAGE_TOKEN_NUM = sentence["value"].count(DEFAULT_IMAGE_TOKEN)
                if IMAGE_TOKEN_NUM > MAX_IMAGE_LENGTH:
                    sentence["value"] = (
                        sentence["value"]
                        .replace(
                            DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM,
                            DEFAULT_IMAGE_TOKEN * MAX_IMAGE_LENGTH,
                        )
                        .strip()
                    )
                VIDEO_TOKEN_NUM = sentence["value"].count(DEFAULT_VIDEO_TOKEN)
                if VIDEO_TOKEN_NUM > MAX_VIDEO_LENGTH:
                    raise ValueError(f"{sentence['value']}")
                    sentence["value"] = (
                        sentence["value"]
                        .replace(
                            DEFAULT_VIDEO_TOKEN * VIDEO_TOKEN_NUM,
                            DEFAULT_VIDEO_TOKEN * MAX_VIDEO_LENGTH,
                        )
                        .strip()
                    )

            # a <video> is treated as `num_frames * <image>`
            replace_token, vid_replace_token = (
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IMAGE_TOKEN * data_args.num_frames,
            )
            if data_args.mm_use_im_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
                vid_replace_token = (
                    DEFAULT_VID_START_TOKEN + vid_replace_token + DEFAULT_VID_END_TOKEN
                )

            # <video><video><image><image>\nxxxxxxxxxxxxx -> `num_frames*<image>``num_frames*<image>`<image><image>\nxxxxxxxxxxxxx
            # <video>\nxxxxxxxxxxxxx -> `num_frames*<image>`\nxxxxxxxxxxxxx
            # print('before replace_token:', [sentence['value']])
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_VIDEO_TOKEN, vid_replace_token
            )
            # print('after replace_token:', [sentence['value']])
            # ======================================================================================================

    return sources


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_llama_2(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    # import pdb; pdb.set_trace()
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack(
        [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversations
        ],
        dim=0,
    )
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(
                conv.sep.join(rounds[conv_idx : conv_idx + 2])
            )  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(
                tokenizer_image_token(conv.sep, tokenizer)
            )
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = (
            source[0]["value"]
            + source[1]["value"]
            + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int = 64,
    system_message: str = "You are a helpful assistant.",
) -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    IGNORE_TOKEN_ID = -100
    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        # import pdb; pdb.set_trace()
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        system = (
            [im_start]
            + _system
            + tokenizer(system_message).input_ids
            + [im_end]
            + nl_tokens
        )
        input_id += system
        target += (
            [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        )
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # _input_id = tokenizer(role).input_ids + nl_tokens + \
            #     tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            _input_id = (
                tokenizer(role).input_ids
                + nl_tokens
                + tokenizer_image_token(sentence["value"], tokenizer)
                + [im_end]
                + nl_tokens
            )
            input_id += _input_id
            if role == "<|im_start|>user":
                _target = (
                    [im_start]
                    + [IGNORE_TOKEN_ID] * (len(_input_id) - 3)
                    + [im_end]
                    + nl_tokens
                )
            elif role == "<|im_start|>assistant":
                _target = (
                    [im_start]
                    + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids)
                    + _input_id[len(tokenizer(role).input_ids) + 1 : -2]
                    + [im_end]
                    + nl_tokens
                )
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.PLAIN
    ):
        return preprocess_plain(sources, tokenizer)
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.QWEN
    ):
        return preprocess_qwen(sources, tokenizer)
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.LLAMA_2
    ):
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversations
        ]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn(
                [header] + [s["value"] for s in source], tokenizer
            )["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedDataset, self).__init__()
        # ================================================
        # data_path: list of json ann file paths
        # image_folder, video_folder: folder of image and video files
        # ================================================
        list_data_dict = []
        for data in data_path:
            data = json.load(open(data, "r"))
            for i in data:
                i["id"] = len(list_data_dict)
                list_data_dict.append(i)
        # ================================================

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    # @property
    # def lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         img_tokens = 128 if 'image' in sample else 0
    #         length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
    #     return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            # ===========================================================================
            cur_len = cur_len if ("image" in sample or "video" in sample) else -cur_len
            # ===========================================================================
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            # ======================================================================================================
            if "image" in sources[0] and "video" not in sources[0]:
                # rank0_print('image')
                image_file = self.list_data_dict[i]["image"]
                image_folder = self.data_args.image_folder
                image_processor = self.data_args.image_processor
                image_file = (
                    image_file if isinstance(image_file, list) else [image_file]
                )
                image_file = order_pick_k(image_file, MAX_IMAGE_LENGTH)
                # print(f"total {len(self.list_data_dict[i]['image'])} now {len(image_file)}")
                image = [
                    Image.open(os.path.join(image_folder, file)).convert("RGB")
                    for file in image_file
                ]
                # import pdb; pdb.set_trace()
                if self.data_args.image_aspect_ratio == "pad":
                    image = [
                        expand2square(
                            i, tuple(int(x * 255) for x in image_processor.image_mean)
                        )
                        for i in image
                    ]
                    image = [
                        image_processor.preprocess(i, return_tensors="pt")[
                            "pixel_values"
                        ][0]
                        for i in image
                    ]
                else:
                    image = [
                        image_processor.preprocess(i, return_tensors="pt")[
                            "pixel_values"
                        ][0]
                        for i in image
                    ]
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]), self.data_args
                )
                data_dict = preprocess(sources, self.tokenizer, has_image=True)

            elif "image" not in sources[0] and "video" in sources[0]:
                # rank0_print('video')
                video_file = self.list_data_dict[i]["video"]
                video_folder = self.data_args.video_folder
                video_processor = self.data_args.video_processor
                video_file = (
                    video_file if isinstance(video_file, list) else [video_file]
                )
                video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
                video = [os.path.join(video_folder, file) for file in video_file]
                image = [
                    video_processor(i, return_tensors="pt")["pixel_values"][0]
                    for i in video
                ]  # fake image
                # image = [torch.randn(3, 8, 224, 224) for i in video]  # fake image
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]), self.data_args
                )
                # print('after preprocess_multimodal', sources[0])
                data_dict = preprocess(sources, self.tokenizer, has_image=True)
                # print('after preprocess', data_dict['input_ids'])

            elif "image" in sources[0] and "video" in sources[0]:
                # rank0_print('image & video')
                # video must before image
                video_file = self.list_data_dict[i]["video"]
                video_folder = self.data_args.video_folder
                video_processor = self.data_args.video_processor

                image_file = self.list_data_dict[i]["image"]
                image_folder = self.data_args.image_folder
                image_processor = self.data_args.image_processor

                image_file = (
                    image_file if isinstance(image_file, list) else [image_file]
                )
                image_file = order_pick_k(image_file, MAX_IMAGE_LENGTH)
                image = [
                    Image.open(os.path.join(image_folder, file)).convert("RGB")
                    for file in image_file
                ]
                if self.data_args.image_aspect_ratio == "pad":
                    image = [
                        expand2square(
                            i, tuple(int(x * 255) for x in image_processor.image_mean)
                        )
                        for i in image
                    ]
                    image = [
                        image_processor.preprocess(i, return_tensors="pt")[
                            "pixel_values"
                        ][0]
                        for i in image
                    ]
                else:
                    image = [
                        image_processor.preprocess(i, return_tensors="pt")[
                            "pixel_values"
                        ][0]
                        for i in image
                    ]

                video_file = (
                    video_file if isinstance(video_file, list) else [video_file]
                )
                video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
                video = [os.path.join(video_folder, file) for file in video_file]
                video = [
                    video_processor(i, return_tensors="pt")["pixel_values"][0]
                    for i in video
                ]  # fake image

                image = video + image  # video must before image

                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]), self.data_args
                )
                data_dict = preprocess(sources, self.tokenizer, has_image=True)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
                data_dict = preprocess(sources, self.tokenizer, has_image=False)

            # ==========================================================================================================

            if isinstance(i, int):
                data_dict = dict(
                    input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
                )
            # image exist in the data
            if "image" in self.list_data_dict[i] or "video" in self.list_data_dict[i]:
                data_dict["image"] = image
            elif self.data_args.is_multimodal:
                # image does not exist in the data, but the model is multimodal
                # crop_size = self.data_args.image_processor.crop_size
                crop_size = {"height": 224, "width": 224}  # dummy image
                data_dict["image"] = [
                    torch.zeros(3, crop_size["height"], crop_size["width"])
                ]
            return data_dict
        except Exception as e:
            print(f"Error with {e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    model_name: str

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        if self.model_name == "flamingo":
            input_ids[input_ids == IMAGE_TOKEN_INDEX] = self.tokenizer.encode(
                "<image>"
            )[-1]
            input_ids = torch.cat([input_ids[:, :1], input_ids[:, 2:]], dim=1)
            labels = torch.cat([labels[:, :1], labels[:, 2:]], dim=1)

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # ======================================================================================================
        # origin image, if batch_size=6: [[image], [image], [video], [image, image], [video, video], [video, image]]
        """
            will be converted to a sequence of list, if batch size=6:
            [
                image(3, 224, 224),      # sample 1
                image(3, 224, 224),      # sample 2
                video(8, 3, 224, 224),   # sample 3
                image(3, 224, 224),      # sample 4
                image(3, 224, 224),      # sample 4
                video(8, 3, 224, 224),   # sample 5
                video(8, 3, 224, 224),   # sample 5
                video(8, 3, 224, 224),   # sample 6
                image(3, 224, 224),      # sample 6
            ]
        """
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            # adapt to multi-video or multi-image or multi-image & video
            new_images = []
            for image in images:
                if type(image) is list:
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images

            # ==========Too many videos or images may lead to OOM, so we encode them one by one======================
            batch["images"] = images
        #     if all(x is not None and x.shape == images[0].shape for x in images):  # if all images or all videos
        #         batch['images'] = torch.stack(images)
        #     else:
        #         batch['images'] = images
        else:
            raise ValueError(f"pretrain, {instances}")

        batch["data_source"] = "vl_pretrain"

        batch["rgb"] = batch.get("images", None)
        batch["instr_and_action_ids"] = batch["input_ids"]
        batch["instr_and_action_mask"] = batch["attention_mask"]
        batch["instr_and_action_labels"] = batch["labels"]

        batch["text"] = batch["input_ids"]
        batch["text_mask"] = batch["attention_mask"]

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    from torch.utils.data import Dataset, DataLoader

    dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator)
    return dataloader


class VideoLLaVADataset(LazySupervisedDataset):
    def __init__(
        self,
        data_paths,
        tokenizer,
        lazy_preprocess: bool = False,
        is_multimodal: bool = False,
        image_aspect_ratio: str = "square",
        image_folder: Optional[str] = field(default=None),
        video_folder: Optional[str] = field(default=None),
        model_name="vicuna",
        vit_name="openai/clip-vit-large-patch14-336",
        **kwargs,
    ):
        if isinstance(tokenizer, dict):
            tokenizer_type = tokenizer["tokenizer_type"]
            max_text_len = max(tokenizer["max_text_len"], 1024)
            tokenizer = build_tokenizer(tokenizer_config=tokenizer)
            self.tokenizer = tokenizer
            self.text_fn = get_text_function(tokenizer, tokenizer_type, max_text_len)
        else:
            self.text_fn = tokenizer

        data_args = DataArguments(
            lazy_preprocess=lazy_preprocess,
            is_multimodal=is_multimodal,
            data_path=data_paths,
            image_folder=image_folder,
            video_folder=video_folder,
            image_aspect_ratio=image_aspect_ratio,
        )
        if "vicuna" in model_name:
            k = "v1"
        elif "mpt" in model_name or "flamingo" in model_name:
            k = "mpt"
        elif "qwen" in model_name:
            k = "qwen"
        else:
            k = "llava_v1"

        conversation_lib.default_conversation = conversation_lib.conv_templates[k]
        image_processor = get_llava_image_processor(vit_name)
        video_processor = get_llava_video_processor(tokenizer)
        data_args.image_processor = image_processor
        data_args.video_processor = video_processor
        self.collater = DataCollatorForSupervisedDataset(
            tokenizer=tokenizer, model_name=model_name
        )
        super(VideoLLaVADataset, self).__init__(
            data_path=data_paths, tokenizer=tokenizer, data_args=data_args
        )
