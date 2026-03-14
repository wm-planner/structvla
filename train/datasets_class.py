# -*- coding: utf-8 -*-
import json
import os.path as osp
import random
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union
from torch.utils.data import Dataset
from PIL import Image
import sys
sys.path.append("/remote-home/jinminghao/structvla")
from models.tokenizer.action_tokenizer import ActionTokenizer
from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor
import time
import os, re, csv, json, pickle, math, random
from pathlib import Path
from typing import Dict, Optional
from collections import OrderedDict


class Emu3SFTDataset(Dataset):

    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__()

        self.args = args
        # data args
        self.random_frame_sampling = args.random_frame_sampling
        self.raw_image = args.raw_image
        
        with open(args.data_path,'rb') as f:
            self.data = pickle.load(f)
        
        if not self.random_frame_sampling:
            self.data = list(self.sliding_window_sampling(self.data, interval=args.action_frames*args.frames))
        
        self.tokenizer = tokenizer
        self.bov = tokenizer.encode(args.visual_token_pattern.format(token_id=0))[0]
        self.eov = tokenizer.encode(args.visual_token_pattern.format(token_id=args.codebook_size - 1))[0]
        self.chat_template="You are a helpful assistant. USER: {image_prompt}{text_prompt}. ASSISTANT:"
        self.gen_template="You are a powerful painter. USER: {text} ASSISTANT:{image}"
        self.act_template="Action: {action_prompt}"
        self.VL = args.VL
        self.cfg = False
        self.post_training = args.post_training

        # pretrain use
        if self.post_training:
            # v2
            # self.dataset_fps = {'rt1':3, 'bridgev2':5, 'droid':15, '1x':1, 'kuka':3, 'calvin':5, 'libero':5} 
            # v3
            self.dataset_fps = {'1x':1, 'SSv2':1,'rt1':3, 'kuka':3, \
                                'bridgev2':5, 'taco_play':5, \
                                'calvin':10, 'libero':10,'maniskill':10,'cmu_play_fusion':10,'utaustin_mutex':10, \
                                'droid':15, 'viola':15, \
                                'toto':20}
        else:
            self.dataset_fps = {}
        self.T = args.frames
        self.action_frames = args.action_frames
        
        self.actions = args.actions
        self.actions_format = args.actions_format

        self.use_gripper = args.use_gripper  

        self.video_format = args.video_format

        if self.raw_image:
            self.vision_hub = "/remote-home/jinminghao/structvla/pretrain/Emu3-VisionVQ"
            self.image_processor = AutoImageProcessor.from_pretrained(self.vision_hub, trust_remote_code=True)
            self.image_tokenizer = AutoModel.from_pretrained(self.vision_hub, trust_remote_code=True)
            self.image_processor.min_pixels = 80 * 80
        if self.actions_format == "openvla":
            self.action_tokenizer = ActionTokenizer(tokenizer, bins=256, min_action=-1.0, max_action=1.0)
        elif self.actions_format == "fast":
            self.fast_path = args.action_tokenizer_path
            self.action_tokenizer = AutoProcessor.from_pretrained(self.fast_path, trust_remote_code=True)

    def __len__(self):
        return len(self.data)
    
    def sliding_window_sampling(self, data, interval=5):
        """
        Implement sliding window sampling using a generator.
        Keep the original behavior: raise when the sequence is too short so the
        issue is surfaced during dataset construction.
        """
        for item in data:
            T = len(item['image'])
            if T <= interval:
                raise ValueError("Length of 'image', 'action', and 'gripper' must be greater than 'interval'.")
            for start_idx in range(0, T - interval + 1, 1):
                yield {
                    'text': item['text'],
                    'image': item['image'][start_idx:start_idx+interval],
                    'action': item['action'][start_idx:start_idx+interval],
                    'gripper_image': item['gripper_image'][start_idx:start_idx+interval],
                }

    def random_frames_to_tensor(self, img_list, T, action_prompt=None, gripper=None):
        """
        Randomly sample a length-T segment from img_list and return different
        tuples depending on whether action / gripper are provided.
        - Return None if a corrupted .npy causes np.load to fail, so __getitem__
          can retry with another sample.
        - Clamp T to avoid a negative randint upper bound.
        """
        # 1) Clamp T using the shortest valid length across image / action / gripper
        max_T = len(img_list)
        if action_prompt is not None:
            max_T = min(max_T, len(action_prompt))
        if gripper is not None:
            max_T = min(max_T, len(gripper))

        if max_T == 0:
            return None

        if T > max_T:
            T = max_T

        # 2) Randomly choose the start index
        try:
            start_idx = random.randint(0, max_T - T)
        except ValueError:
            # max_T - T can be negative in extreme cases; return None directly
            return None

        # 3) Raw-image branch (usually unused when pre-tokenized inputs are available)
        if hasattr(self, 'raw_image') and self.raw_image:
            self.image_tokenizer.eval()
            try:
                selected_frames = [Image.open(img_path) for img_path in img_list[start_idx:start_idx + T]]
                selected_frames = [
                    self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
                    for img in selected_frames
                ]
            except Exception:
                return None

            tensor_frames = torch.stack(selected_frames, dim=0)
            with torch.no_grad():
                image_code = self.image_tokenizer.encode(tensor_frames)
            
            if gripper is not None and action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                try:
                    selected_gripper = [Image.open(img_path) for img_path in gripper[start_idx:start_idx + T]]
                    selected_gripper = [
                        self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
                        for img in selected_gripper
                    ]
                except Exception:
                    return None
                tensor_gripper = torch.stack(selected_gripper, dim=0)
                with torch.no_grad():
                    gripper_code = self.image_tokenizer.encode(tensor_gripper)
                return image_code, selected_actions, gripper_code
            elif action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                return image_code, selected_actions
            elif gripper is not None:
                try:
                    selected_gripper = [Image.open(img_path) for img_path in gripper[start_idx:start_idx + T]]
                    selected_gripper = [
                        self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
                        for img in selected_gripper
                    ]
                except Exception:
                    return None
                tensor_gripper = torch.stack(selected_gripper, dim=0)
                with torch.no_grad():
                    gripper_code = self.image_tokenizer.encode(tensor_gripper)
                return image_code, gripper_code
            else:
                return image_code

        # 4) Pre-tokenized (.npy) branch
        sel_img_paths = img_list[start_idx:start_idx + T]
        selected_frames = []
        for p in sel_img_paths:
            try:
                arr = np.load(p, allow_pickle=False)
                selected_frames.append(torch.from_numpy(arr))
            except Exception:
                return None

        if len(selected_frames) == 0:
            return None

        tensor = torch.stack(selected_frames, dim=1)  # [C, T] or similar

        if gripper is not None and action_prompt is not None:
            selected_actions = action_prompt[start_idx:start_idx + T]
            sel_gripper_paths = gripper[start_idx:start_idx + T]
            gripper_frames = []
            for p in sel_gripper_paths:
                try:
                    g = np.load(p, allow_pickle=False)
                    gripper_frames.append(torch.from_numpy(g))
                except Exception:
                    return None
            if len(gripper_frames) == 0:
                return None
            tensor_gripper = torch.stack(gripper_frames, dim=1)
            return tensor.squeeze(0), selected_actions, tensor_gripper.squeeze(0)

        elif action_prompt is not None:
            selected_actions = action_prompt[start_idx:start_idx + T]
            return tensor.squeeze(0), selected_actions

        elif gripper is not None:
            sel_gripper_paths = gripper[start_idx:start_idx + T]
            gripper_frames = []
            for p in sel_gripper_paths:
                try:
                    g = np.load(p, allow_pickle=False)
                    gripper_frames.append(torch.from_numpy(g))
                except Exception:
                    return None
            if len(gripper_frames) == 0:
                return None
            tensor_gripper = torch.stack(gripper_frames, dim=1)
            return tensor.squeeze(0), tensor_gripper.squeeze(0)

        # Simple case with neither action nor gripper inputs
        return tensor.squeeze(0)
    
    def get_fps_for_path(self, image_tokens_path):
        for key in self.dataset_fps.keys():
            if key in image_tokens_path[0]:
                return self.dataset_fps[key]
        # Default return value if no key matches
        return None  # or some default FPS value
    
    def pad_tensor(self, tensor, max_length, pad_value):
        """Pads a tensor to a specified maximum length."""
        current_length = tensor.shape[-1]
        if current_length < max_length:
            pad_length = max_length - current_length
            padding = torch.full((pad_length,), fill_value=pad_value, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=-1)
        return tensor

    def __getitem__(self, index: int):
        """
        __getitem__ with retry support:
        - If a sample fails in random_frames_to_tensor / np.load / similar steps,
          retry with another index up to max_retry times.
        """
        max_retry = 5
        orig_index = index
        idx = index

        for attempt in range(max_retry):
            try:
                scene = self.data[idx]

                if self.cfg:
                    p_prob = random.random()
                    if p_prob < self.args.null_prompt_prob:
                        prompt = ""
                    else:
                        prompt = scene["text"]
                else:
                    prompt = scene["text"]

                image_tokens_path = scene["image"]

                # handle different dataset fps for post training
                fps = self.get_fps_for_path(image_tokens_path)
                if fps is not None:
                    self.action_frames = fps
                
                if self.T > 1 and self.video_format == "interleave":
                    if len(image_tokens_path) > self.T * self.action_frames:
                        frames_num = self.T * self.action_frames
                    else:
                        frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
                else:
                    frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
                
                # ---------- use action information ----------
                if self.actions:
                    action = scene["action"]  
                    if self.use_gripper:
                        gripper = scene["gripper_image"]
                        out = self.random_frames_to_tensor(
                            image_tokens_path, frames_num,
                            action_prompt=action, gripper=gripper
                        )
                        if out is None:
                            raise RuntimeError("random_frames_to_tensor returned None (actions+gripper)")
                        image_tokens, action_tokens, gripper_tokens = out
                    else:
                        out = self.random_frames_to_tensor(
                            image_tokens_path, frames_num,
                            action_prompt=action
                        )
                        if out is None:
                            raise RuntimeError("random_frames_to_tensor returned None (actions only)")
                        image_tokens, action_tokens = out
                    
                    if self.video_format == "interleave":
                        if self.actions_format == "fast":
                            if isinstance(action_tokens, list):
                                tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                                # Concatenate tensors along the first dimension
                                action_tokens = torch.cat(tensor_list, dim=0)
                            action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                            action_ids = self.action_tokenizer(action_tokens)
                            self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                            action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                        else:
                            raise ValueError(f"Invalid actions_format: {self.actions_format}")
                    else:
                        if self.actions_format == "openvla":
                            action_tokens = action_tokens.flatten()
                            action_ids = self.action_tokenizer(action_tokens)
                        elif self.actions_format == "text":
                            action_str = "\n".join(",".join(f"{num:.2f}" for num in row) for row in action_tokens)
                            action_prompt = self.act_template.format(action_prompt=action_str)
                        elif self.actions_format == "continuous":
                            action_continuous = action_tokens
                        elif self.actions_format == "fast":
                            if isinstance(action_tokens, list):
                                tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                                action_tokens = torch.cat(tensor_list, dim=0)
                            action_ids = self.action_tokenizer(action_tokens)[0]
                            self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                            action_ids = [self.last_vocab_idx - id for id in action_ids]
                        else:
                            raise ValueError(f"Invalid actions_format: {self.actions_format}")
                else:
                    if self.use_gripper:
                        gripper = scene["gripper_image"]
                        out = self.random_frames_to_tensor(
                            image_tokens_path, frames_num,
                            gripper=gripper
                        )
                        if out is None:
                            raise RuntimeError("random_frames_to_tensor returned None (gripper only)")
                        image_tokens, gripper_tokens = out
                    else:
                        out = self.random_frames_to_tensor(image_tokens_path, frames_num)
                        if out is None:
                            raise RuntimeError("random_frames_to_tensor returned None (image only)")
                        image_tokens = out

                # ---------------- video VLA ----------------
                if self.video_format == "interleave":
                    text_prompt = self.tokenizer.bos_token + prompt
                    image_tokens = image_tokens[0::self.action_frames,...]
                    if self.use_gripper:
                        gripper_tokens = gripper_tokens[0::self.action_frames,...]
                    
                    sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                    sample_input_ids = sample_text["input_ids"][0]
                    sample_attention_mask = sample_text["attention_mask"][0]

                    labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
                    for i in range(len(image_tokens)):
                        image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                        if self.use_gripper:
                            gripper_prompt = self.format_video_prompt(gripper_tokens[i:i+1])
                            image_prompt += gripper_prompt
                        sample_img = self.tokenizer(image_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                        image_input_ids = sample_img["input_ids"][0]
                        image_attention_mask = sample_img["attention_mask"][0]
                        if self.actions:
                            if self.actions_format == "fast":
                                action_sample = self.wrap_action_sequence(action_ids[i].tolist()) 
                                sample_input_ids = torch.cat([sample_input_ids, image_input_ids, action_sample], dim=-1)  
                                sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1) 
                                action_start = len(sample_input_ids) - len(action_sample)
                                action_end = len(sample_input_ids)
                                if self.args.apply_loss_on_only_action:  
                                    labels[action_start:action_end] = action_sample
                                else:
                                    labels[action_start-len(image_input_ids):action_start] = image_input_ids  
                                    labels[action_start:action_end] = action_sample 
                        else:
                            sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                            sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                            labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids
                    
                    sample = self.tokenizer.pad(
                            {
                                "input_ids": sample_input_ids,
                                "attention_mask": sample_attention_mask,
                                "labels": labels
                            },
                            padding="max_length",
                            return_tensors="pt"
                        )
                    for k, v in sample.items():
                        sample[k] = v.squeeze(0)

                # ---------------- VLA Baseline (Img) ----------------
                else:
                    image_tokens = image_tokens[0:self.T,...]
                    image_prompt = self.format_video_prompt(image_tokens)

                    if self.use_gripper:
                        gripper_tokens = gripper_tokens[0:self.T,...]
                        gripper_prompt = self.format_video_prompt(gripper_tokens)
                        image_prompt = image_prompt + gripper_prompt  

                    if self.VL:
                        p_prob_order = random.random()
                        if p_prob_order < 0.5:
                            input = self.tokenizer.bos_token + prompt + image_prompt + self.tokenizer.eos_token
                        else:
                            input = self.tokenizer.bos_token + self.chat_template.format(image_prompt=image_prompt, text_prompt=prompt) + self.tokenizer.eos_token
                    else:
                        input = self.tokenizer.bos_token + prompt + image_prompt 
                    # Delay padding and handle it later in a single place
                    sample = self.tokenizer(
                        input,
                        padding=False,
                        return_token_type_ids=False,
                        return_tensors="pt",
                    )
                    labels = sample["input_ids"]

                    # only use vision loss
                    if self.args.apply_loss_on_only_vision:
                        labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)

                    sample["labels"] = labels
                    for k, v in sample.items():
                        sample[k] = v.squeeze(0)

                    # based on the actions_format, append the action information to the sample
                    if self.actions:
                        if self.actions_format == "openvla":
                            action_sample = self.wrap_action_sequence(action_ids)
                            sample["input_ids"] = torch.cat([sample["input_ids"], action_sample], dim=-1)

                            # Update attention_mask
                            action_mask = torch.ones_like(action_sample, dtype=torch.long)
                            sample["attention_mask"] = torch.cat([sample["attention_mask"], action_mask], dim=-1)

                            action_labels = action_sample.clone()
                            sample["labels"] = torch.cat([sample["labels"], action_labels], dim=-1)
                        
                        elif self.actions_format == "fast":
                            if self.args.apply_loss_on_only_action:
                                sample['labels'] = torch.full_like(sample['labels'], self.args.ignore_index)
                            sample = self.append_action_to_sample(sample, action_ids)
                        
                        elif self.actions_format == "continuous":
                            boa_token_id = self.tokenizer.encode(self.tokenizer.boa_token)[0]
                            sample = self.append_boa_to_sample(sample, [boa_token_id])
                            sample["action"] = action_continuous
                    
                    # finally, do padding
                    sample = self.tokenizer.pad(
                        sample,
                        padding="max_length",
                        return_tensors="pt"
                    )

                    for k, v in sample.items():
                        sample[k] = v.squeeze(0)

                    if "labels" in sample:
                        sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)

                # Reaching this point means the current idx succeeded; return the sample directly
                return sample

            except Exception as e:
                # Current idx failed; switch to a new idx and continue
                # Keep a lightweight log for later debugging
                print(f"[Emu3SFTDataset] __getitem__ failed at idx={idx}, attempt={attempt+1}/{max_retry}, error={e}")
                idx = random.randint(0, len(self.data) - 1)

        # Raise after repeated retries still fail
        raise RuntimeError(f"Emu3SFTDataset __getitem__ failed for orig_index={orig_index} after {max_retry} retries")

    def append_action_to_sample(self, sample, action_ids):
        """
        Process action_ids and append them to sample, including input_ids, attention_mask, and labels.
        """
        action_sample = self.wrap_action_sequence(action_ids)
        action_mask = torch.ones_like(action_sample, dtype=torch.long)

        for key, value in zip(["input_ids", "attention_mask", "labels"], [action_sample, action_mask, action_sample.clone()]):
            sample[key] = torch.cat([sample[key], value], dim=-1)

        return sample
    
    def append_boa_to_sample(self, sample, action_ids):

        action_sample = torch.tensor(action_ids, dtype=torch.long)
        action_mask = torch.ones_like(action_sample, dtype=torch.long)

        for key, value in zip(["input_ids", "attention_mask", "labels"], [action_sample, action_mask, action_sample.clone()]):
            sample[key] = torch.cat([sample[key], value], dim=-1)

        return sample

    def wrap_action_sequence(self, action_ids: List[int]) -> torch.Tensor:
        """
        Wraps a sequence of action token IDs with special tokens (beginning and end).
        """
        action_begin = self.tokenizer.encode(self.tokenizer.boa_token)[0]
        action_end = self.tokenizer.encode(self.tokenizer.eoa_token)[0]
        eos = self.tokenizer.encode(self.tokenizer.eos_token)[0]

        wrapped_action = [action_begin] + action_ids + [action_end]
        return torch.tensor(wrapped_action, dtype=torch.long)

    def format_video_prompt(self, video_tokens):
        # Assume video_tokens has shape [frames, height, width]
        frames, h, w = video_tokens.shape
        videostr = self.to_videostr(video_tokens)

        video_prompt = (
            self.tokenizer.boi_token +
            f"{frames}*{h}*{w}" +
            self.tokenizer.img_token +
            videostr +
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token
        )
        return video_prompt

    def to_videostr(self, video_tokens):
        frame_str_list = []
        for frame in video_tokens:
            frame_token_str = [
                self.args.visual_token_pattern.format(token_id=token_id)
                for token_id in frame.flatten()
            ]
            frame_str = "".join(frame_token_str)
            frame_str_list.append(frame_str)
        videostr = self.tokenizer.eof_token.join(frame_str_list)
        return videostr

    def format_image_prompt(self, image_tokens):
        h, w = image_tokens.shape
        imgstr = self.to_imgstr(image_tokens)

        image_prompt = (
            self.tokenizer.boi_token +
            f"{h}*{w}" +
            self.tokenizer.img_token +
            imgstr +
            self.tokenizer.eol_token +
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token
        )
        return image_prompt

    def to_imgstr(self, image_tokens):
        image_token_str = [
            [
                self.args.visual_token_pattern.format(token_id=token_id)
                for token_id in token_row
            ]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr



class Emu3WorldModelDataset(Emu3SFTDataset):    

    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__(args, tokenizer=tokenizer)
        # weights
        dataset_weights = {
            'rt1': 0.3,
            'droid_fast': 0.2,
            'oxembodiment/bridge': 1.0,
            'oxembodiment/toto': 1.0,
            'oxembodiment/taco_play': 1.0,
            'oxembodiment/fmb': 1.0,
            'oxembodiment/maniskill': 0.5,
            'oxembodiment/kuka': 0.1,
            'oxembodiment/berkeley_autolab_ur5': 1.0,
            'calvin': 0.8,
            'libero': 1.0,
        }
        self.datasets_weight = args.datasets_weight
        if self.datasets_weight:
            self.sample_weights = [dataset_weights.get(d["dataset"], 1.0) for d in self.data]
        self.without_text = args.without_text

    def __getitem__(self, index: int):

        scene = self.data[index]

        if self.without_text:
            prompt = ""
        else:
            prompt = scene["text"]

        image_tokens_path = scene["image"]

        # handle different dataset fps for post training
        fps = self.get_fps_for_path(image_tokens_path)
        if fps is not None:
            self.action_frames = fps
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        
        if self.use_gripper and "gripper_image" in scene:
            gripper = scene["gripper_image"]
            image_tokens, gripper_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, gripper=gripper)
        else:
            image_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num) 
        
        # video VLA
        if self.video_format == "interleave":
            text_prompt = self.tokenizer.bos_token + prompt
            image_tokens = image_tokens[0::self.action_frames,...]
            if self.use_gripper and "gripper_image" in scene:
                gripper_tokens = gripper_tokens[0::self.action_frames,...]
            
            sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                if self.use_gripper and "gripper_image" in scene:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[i:i+1])
                    image_prompt += gripper_prompt
                sample_img = self.tokenizer(image_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                
                sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids
            
            if self.args.apply_loss_on_only_vision:
                labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)
            
            sample = self.tokenizer.pad(
                    {
                        "input_ids": sample_input_ids,
                        "attention_mask": sample_attention_mask,
                        "labels": labels
                    },
                    padding="max_length",
                    return_tensors="pt"
                )
            for k, v in sample.items():
                sample[k] = v.squeeze(0)
        
        else:
            raise NotImplementedError("Only interleave video format is supported for world model dataset.")
        return sample
    
class Emu3RealRobotDataset(Emu3SFTDataset):    

    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__(args, tokenizer=tokenizer)
        self.use_views = ['cam_high','cam_left_wrist','cam_right_wrist']
    
    def random_frames_to_tensor(self, img_list, T, action_prompt=None, wrist=None):
        
        start_idx = random.randint(0, len(img_list) - T)

        selected_frames = [np.load(img_path) for img_path in img_list[start_idx:start_idx + T]]
        tensor_frames = [torch.from_numpy(frame) for frame in selected_frames]
        tensor = torch.stack(tensor_frames, dim=1)

        wrist_left = wrist["cam_left_wrist"]
        wrist_right = wrist["cam_right_wrist"]

        select_wrist_left = [torch.from_numpy(np.load(img_path)) for img_path in wrist_left[start_idx:start_idx + T]]
        select_wrist_right = [torch.from_numpy(np.load(img_path)) for img_path in wrist_right[start_idx:start_idx + T]]

        tensor_wrist_left = torch.stack(select_wrist_left, dim=1)
        tensor_wrist_right = torch.stack(select_wrist_right, dim=1)

        if action_prompt is None:
            return tensor.squeeze(0), tensor_wrist_left.squeeze(0), tensor_wrist_right.squeeze(0)

        selected_actions = action_prompt[start_idx:start_idx + T]
        return tensor.squeeze(0), tensor_wrist_left.squeeze(0), tensor_wrist_right.squeeze(0), selected_actions
    
    def __getitem__(self, index: int):

        scene = self.data[index]

        prompt = scene["text"]

        image_tokens_path = scene["cam_high"]
        
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        
        # use action information
        if self.actions:
            action = scene["action"] 
            image_tokens, wrist_left_token, wrist_right_token, action_tokens= self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, wrist=scene)
            
            if self.video_format == "interleave":
                if self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                    action_ids = self.action_tokenizer(action_tokens)
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                    
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
            else:
                if self.actions_format == "openvla":
                    action_tokens = action_tokens.flatten()
                    action_ids = self.action_tokenizer(action_tokens)
                elif self.actions_format == "text":
                    action_str = "\n".join(",".join(f"{num:.2f}" for num in row) for row in action_tokens)
                    action_prompt = self.act_template.format(action_prompt=action_str)
                elif self.actions_format == "continuous":
                    action_continuous = action_tokens
                elif self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_ids = self.action_tokenizer(action_tokens)[0]
                    # action_decode = self.action_tokenizer.decode([action_ids])
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - id for id in action_ids]
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
        else:
            image_tokens, wrist_left_token, wrist_right_token = self.random_frames_to_tensor(image_tokens_path, frames_num, wrist=scene)
        
        # video VLA
        if self.video_format == "interleave":
            text_prompt = self.tokenizer.bos_token + prompt
            image_tokens = image_tokens[0::self.action_frames,...]
            wrist_left_token = wrist_left_token[0::self.action_frames,...]
            wrist_right_token = wrist_right_token[0::self.action_frames,...]
            
            sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                wrist_left_prompt = self.format_video_prompt(wrist_left_token[i:i+1])
                wrist_right_prompt = self.format_video_prompt(wrist_right_token[i:i+1])
                image_prompt += wrist_left_prompt + wrist_right_prompt
                sample_img = self.tokenizer(image_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                if self.actions:
                    if self.actions_format == "fast":
                        action_sample = self.wrap_action_sequence(action_ids[i].tolist()) 
                        sample_input_ids = torch.cat([sample_input_ids, image_input_ids, action_sample], dim=-1)  
                        sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1) 
                        action_start = len(sample_input_ids) - len(action_sample)
                        action_end = len(sample_input_ids)
                        if self.args.apply_loss_on_only_action:  
                            labels[action_start:action_end] = action_sample
                        else:  # Otherwise, fill both vision and action parts in the labels
                            labels[action_start-len(image_input_ids):action_start] = image_input_ids  
                            labels[action_start:action_end] = action_sample 
                else:
                    sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                    sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                    labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids
            sample = self.tokenizer.pad(
                    {
                        "input_ids": sample_input_ids,
                        "attention_mask": sample_attention_mask,
                        "labels": labels
                    },
                    padding="max_length",
                    return_tensors="pt"
                )
            for k, v in sample.items():
                sample[k] = v.squeeze(0)
        # VLA Baseline (Img)
        else:
            image_tokens = image_tokens[0:self.T,...]
            image_prompt = self.format_video_prompt(image_tokens)

            wrist_left_tokens = wrist_left_token[0:self.T,...]
            wrist_right_tokens = wrist_right_token[0:self.T,...]
            wrist_left_prompt = self.format_video_prompt(wrist_left_tokens)
            wrist_right_prompt = self.format_video_prompt(wrist_right_tokens)
            image_prompt = image_prompt + wrist_left_prompt + wrist_right_prompt
            
            input = self.tokenizer.bos_token + prompt + image_prompt 

            sample = self.tokenizer(
                input,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            labels = sample["input_ids"]

            # only use vision loss
            if self.args.apply_loss_on_only_vision:
                labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)

            sample["labels"] = labels
            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            # based on the actions_format, append the action information to the sample
            if self.actions:
                if self.actions_format == "openvla":
                    action_sample = self.wrap_action_sequence(action_ids)
                    sample["input_ids"] = torch.cat([sample["input_ids"], action_sample], dim=-1)

                    # Update attention_mask
                    action_mask = torch.ones_like(action_sample, dtype=torch.long)
                    sample["attention_mask"] = torch.cat([sample["attention_mask"], action_mask], dim=-1)

                    action_labels = action_sample.clone()  # Clone action_sample for labels
                    sample["labels"] = torch.cat([sample["labels"], action_labels], dim=-1)
                
                # FAST
                elif self.actions_format == "fast":
                    if 'state' in scene.keys():
                        state = scene['state'].reshape(1, 1, -1)
                        state_tokens = self.action_tokenizer(state)[0]
                        state_ids = [self.last_vocab_idx - id for id in state_tokens]
                        state_tensor = torch.tensor(state_ids, dtype=sample["input_ids"].dtype, device=sample["input_ids"].device)

                        sample["input_ids"] = torch.cat([sample["input_ids"], state_tensor], dim=-1)

                        state_label_tensor = torch.full_like(state_tensor, fill_value=-100)  # -100 means ignored in loss
                        sample["labels"] = torch.cat([sample["labels"], state_label_tensor], dim=-1)

                        state_mask = torch.ones_like(state_tensor)
                        sample["attention_mask"] = torch.cat([sample["attention_mask"], state_mask], dim=-1)
                    
                    if self.args.apply_loss_on_only_action:
                        sample['labels'] = torch.full_like(sample['labels'], self.args.ignore_index)
                    sample = self.append_action_to_sample(sample, action_ids)
                # Flow Matching
                elif self.actions_format == "continuous":
                    boa_token_id = self.tokenizer.encode(self.tokenizer.boa_token)[0]
                    sample = self.append_boa_to_sample(sample, [boa_token_id])
                    sample["action"] = action_continuous
            
            # finally, do padding
            sample = self.tokenizer.pad(
                sample,
                padding="max_length",
                return_tensors="pt"
            )

            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            if "labels" in sample:
                sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)
        return sample

class Emu3CoTDataset(Emu3SFTDataset):    

    def __init__(self, args: "DataArguments", tokenizer):
        super().__init__(args, tokenizer=tokenizer)
    
    def random_frames_to_tensor(self, img_list, T, action_prompt=None, reason_prompt=None):
        start_idx = random.randint(0, len(img_list) - T)

        selected_frames = [np.load(img_path) for img_path in img_list[start_idx:start_idx + T]]
        tensor_frames = [torch.from_numpy(frame) for frame in selected_frames]
        tensor = torch.stack(tensor_frames, dim=1)

        selected_actions = action_prompt[start_idx:start_idx + T]
        selected_reason = reason_prompt[start_idx:start_idx + T]
        return tensor.squeeze(0), selected_actions, selected_reason
    
    def __getitem__(self, index: int):

        scene = self.data[index]
        prompt = scene["text"]
        image_tokens_path = scene["image"]
        
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        

        action = scene["action"] 
        image_tokens, action_tokens, reason_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, reason_prompt=scene["reasoning"])
        
        if self.video_format == "interleave":
            if self.actions_format == "fast":
                if isinstance(action_tokens, list):
                    tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                    # Concatenate tensors along the first dimension
                    action_tokens = torch.cat(tensor_list, dim=0)
                action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                action_ids = self.action_tokenizer(action_tokens)
                self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                
            else:
                raise ValueError(f"Invalid actions_format: {self.actions_format}")
        else:
            if self.actions_format == "openvla":
                action_tokens = action_tokens.flatten()
                action_ids = self.action_tokenizer(action_tokens)

                # Debugging
                # action_debug = self.action_tokenizer.decode_token_ids_to_actions(action_ids)
                # error = action_tokens - action_debug
            elif self.actions_format == "text":
                action_str = "\n".join(",".join(f"{num:.2f}" for num in row) for row in action_tokens)
                action_prompt = self.act_template.format(action_prompt=action_str)
            elif self.actions_format == "continuous":
                action_continuous = action_tokens
            elif self.actions_format == "fast":
                if isinstance(action_tokens, list):
                    tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                    # Concatenate tensors along the first dimension
                    action_tokens = torch.cat(tensor_list, dim=0)
                action_ids = self.action_tokenizer(action_tokens)[0]
                # action_decode = self.action_tokenizer.decode([action_ids])
                self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                action_ids = [self.last_vocab_idx - id for id in action_ids]
            else:
                raise ValueError(f"Invalid actions_format: {self.actions_format}")
        
        # video VLA
        if self.video_format == "interleave":
            text_prompt = self.tokenizer.bos_token + prompt
            image_tokens = image_tokens[0::self.action_frames,...]
            if self.use_gripper:
                gripper_tokens = gripper_tokens[0::self.action_frames,...]
            
            sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                if self.use_gripper:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[i:i+1])
                    image_prompt += gripper_prompt
                sample_img = self.tokenizer(image_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                if self.actions:
                    if self.actions_format == "fast":
                        action_sample = self.wrap_action_sequence(action_ids[i].tolist()) 
                        sample_input_ids = torch.cat([sample_input_ids, image_input_ids, action_sample], dim=-1)  
                        sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1) 
                        action_start = len(sample_input_ids) - len(action_sample)
                        action_end = len(sample_input_ids)
                        if self.args.apply_loss_on_only_action:  
                            labels[action_start:action_end] = action_sample
                        else:  # Otherwise, fill both vision and action parts in the labels
                            labels[action_start-len(image_input_ids):action_start] = image_input_ids  
                            labels[action_start:action_end] = action_sample 
                else:
                    sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                    sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                    labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids
            
            sample = self.tokenizer.pad(
                    {
                        "input_ids": sample_input_ids,
                        "attention_mask": sample_attention_mask,
                        "labels": labels
                    },
                    padding="max_length",
                    return_tensors="pt"
                )
            for k, v in sample.items():
                sample[k] = v.squeeze(0)
                
        # VLA Baseline (Img)
        else:
            image_tokens = image_tokens[0:self.T,...]
            image_prompt = self.format_video_prompt(image_tokens)

            reason_tokens = reason_tokens[0:self.T]

            input = self.tokenizer.bos_token + prompt + image_prompt + self.tokenizer.bot_token + reason_tokens[0]['reasoning'] + self.tokenizer.eot_token

            sample = self.tokenizer(
                input,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            labels = sample["input_ids"]

            # not use vision loss
            labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), self.args.ignore_index, labels)

            sample["labels"] = labels
            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            # based on the actions_format, append the action information to the sample
            if self.actions:
                if self.args.apply_loss_on_only_action:
                    sample['labels'] = torch.full_like(sample['labels'], self.args.ignore_index)
                sample = self.append_action_to_sample(sample, action_ids)
            
            # finally, do padding
            sample = self.tokenizer.pad(
                sample,
                padding="max_length",
                return_tensors="pt"
            )

            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            if "labels" in sample:
                sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)
        return sample




class Emu3PlannerDataset(Emu3SFTDataset):
    """
    Online planner with per-keystep sampling:
      - Expand __len__/__getitem__ from per-episode to per-(episode, keystep) so each pass covers all keysteps
      - For a keystep k_j, sample backward every action_frames until the previous keystep (exclusive) or the sequence start
      - Apply labels only on the target keystep frame; context remains unsupervised and concatenated in temporal order
      - If no keystep exists, fall back to pseudo keysteps at the middle and final frame
      - Preserve the original interleave / visual-token marking / truncation / right-padding logic and remove loss_weight/context weighting
      - When raw_image=True, VQ encoding can be batched to reduce repeated encode overhead
    """

    # ---------- Keystep mapping load (unchanged) ----------
    @staticmethod
    def _load_keystep_list(path: str) -> Dict[str, List[int]]:
        p = str(path)
        if p.endswith(".csv"):
            ks_map = {}
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ep = str(row.get("episode_id", "")).strip()
                    if not ep or ("step_keystep" not in row):
                        continue
                    try:
                        k = int(row["step_keystep"])
                    except Exception:
                        continue
                    ks_map.setdefault(ep, set()).add(k)
            return {k: sorted(v) for k, v in ks_map.items()}
        if p.endswith(".json"):
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
        elif p.endswith(".pkl") or p.endswith(".pickle"):
            with open(p, "rb") as f:
                raw = pickle.load(f)
        elif p.endswith(".txt"):
            raw = {}
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if (not line) or line.startswith("#"):
                        continue
                    if ":" in line:
                        ep, v = line.split(":", 1)
                    else:
                        ep, v = line.split(None, 1)
                    nums = [int(x) for x in re.findall(r"-?\d+", v)]
                    raw[str(ep).strip()] = nums
        else:
            raise ValueError(f"Unsupported keysteps file type: {p}")
        return {str(k): sorted(set(int(x) for x in v)) for k, v in raw.items()}

    @staticmethod
    def _scene_key(scene: dict, key_from: str, key_field: Optional[str], scene_idx: int):
        if key_from == "index":
            return str(scene_idx)
        if key_from == "field":
            if (key_field is None) or (key_field not in scene):
                return None
            return str(scene[key_field])
        if key_from == "stem":
            img_list = scene.get("image", None)
            if not img_list:
                return None
            p = Path(img_list[0])
            return p.parent.name if p.parent.name not in ("", ".", "/") else p.stem
        return None

    # ---------- Construction ----------
    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__(args, tokenizer=tokenizer)
        assert self.video_format == "interleave", "Planner only supports interleave"
        # 1) Planner must use full-length episodes; SFT sliding-window expansion is not allowed
        if not getattr(self, "random_frame_sampling", True):
            raise ValueError(
                "Emu3PlannerDataset requires args.random_frame_sampling=True; "
                "otherwise SFT sliding_window_sampling will fragment episodes and break keystep alignment."
            )

        # 2) Optional overfit mode: keep only the first N episodes
        cap = getattr(args, "planner_overfit_first_n", None)
        if cap is not None and int(cap) > 0:
            self.data = self.data[:cap]

        # 3) Optional I/O acceleration: mmap + LRU cache
        self.use_mmap = bool(getattr(args, "planner_use_mmap", True))
        self.npy_cache_cap = int(getattr(args, "planner_npy_cache_cap", 4096))
        self._npy_cache = OrderedDict()

        # [MOD] T frames: only used as an upper bound; the actual context length comes from backward sampling and may be < T
        self.T = max(2, int(getattr(args, "frames", 2)))
        self.ctx_len = self.T - 1

        # [MOD] Keystep alignment config and mapping
        self.key_from  = getattr(args, "keystep_key_from", "index")
        self.key_field = getattr(args, "keystep_key_field", "episode_id")
        ks_file = getattr(args, "keystep_path", None)
        if not ks_file:
            raise ValueError("planner requires --keystep_path (CSV/JSON/PKL/TXT are all supported)")
        self.ep2K_raw = self._load_keystep_list(ks_file)
        self.expand_by_offset = bool(getattr(args, "planner_expand_by_offset", False))
        self.max_groups = int(getattr(args, "max_groups_per_keystep", int(self.action_frames)))
        
        # NEW: Switch for sampling only one group per episode in each epoch
        self.one_group_per_episode = bool(getattr(args, "planner_one_group_per_episode", True))
        
        # [MOD] Build a flat index (ep_idx, keystep_rank) to guarantee every keystep is covered each pass
        self.flat_index = []   # list of (scene_idx, j, g)
        self.ep_keysteps = []
        for si, scene in enumerate(self.data):
            N = len(scene["image"])
            ep_key = self._scene_key(scene, self.key_from, self.key_field, si)
            K = sorted([kk for kk in self.ep2K_raw.get(str(ep_key), []) if 0 <= kk < N]) if ep_key is not None else []
            K = sorted(set(K))
            if not K:
                mid = max(0, min(N-1, N // 2))
                last = max(0, N - 1)
                K = sorted(set([mid, last]))

            self.ep_keysteps.append(K)

            for j in range(len(K)):
                if not self.expand_by_offset:
                    # Original behavior: each keystep produces only one sample
                    self.flat_index.append((si, j, 0))
                    continue

                k = K[j]
                right_bound = K[j+1] if j+1 < len(K) else N  # Exclusive right bound
                # Expand by offset: g=0..max_groups-1, with tgt=k+g kept strictly before right_bound
                for g in range(self.max_groups):
                    tgt = k + g
                    if tgt >= right_bound or tgt >= N:
                        break
                    self.flat_index.append((si, j, g))
        if self.expand_by_offset:
            print(f"[planner] expand_by_offset=True, max_groups={self.max_groups}, "
                f"flat_index_len={len(self.flat_index)}", flush=True)
        
        # NEW: Bucket indices by episode: episode -> indices of all candidate samples in flat_index
        from collections import defaultdict
        self._ep_buckets = defaultdict(list)
        for idx, (si, j, g) in enumerate(self.flat_index):
            self._ep_buckets[si].append(idx)

        # NEW: Record the list of valid episodes (at least one candidate)
        self._episode_ids = sorted([si for si, bucket in self._ep_buckets.items() if len(bucket) > 0])        
        # [MOD] Optional raw_image batch encoding switch (enabled by default)
        self.batch_encode = bool(getattr(args, "planner_batch_encode", True))
    
    def __len__(self):
        # NEW: If "one group per episode" is enabled, length = number of episodes; otherwise use the full flat sample count
        return len(self._episode_ids) if self.one_group_per_episode else len(self.flat_index)

    # ---------- Optional batched VQ encoding (active when raw_image=True) ----------
    def _load_frames_tokens_batch(self, img_list: List[str], indices: List[int]) -> List[torch.Tensor]:
        """
        Return List[1,H,W] aligned with indices; batch VQ-encode once when raw_image=True.
        When raw_image=False, fall back to per-frame loading, equivalent to _load_single_frame_tokens.
        """
        if not getattr(self, "raw_image", False):
            return [self._load_single_frame_tokens(img_list, i) for i in indices]

        self.image_tokenizer.eval()
        imgs = []
        for i in indices:
            img = Image.open(img_list[i])
            px = self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
            imgs.append(px)
        pixels = torch.stack(imgs, dim=0)  # [B,C,H,W]
        with torch.no_grad():
            codes = self.image_tokenizer.encode(pixels)  # Expected to return [B, H, W] or a compatible shape
        codes = torch.as_tensor(codes)
        if codes.dim() == 3:
            codes = codes.unsqueeze(1)  # [B,1,H,W]
        elif codes.dim() == 4 and codes.size(1) != 1:
            # Use the most conservative reshape when the result is not [B,1,H,W]
            B = codes.size(0)
            codes = codes.permute(0, 2, 3, 1).reshape(B, 1, codes.size(2), -1)
        out = [codes[b] for b in range(codes.size(0))]
        return out

    # ---------- Compute the backward window for a given keystep ----------
    # [MOD] Core rule: step backward every action_frames; stop before the previous keystep or at the sequence start; preserve temporal order
    # ==== Modified: _window_for_keystep supports an externally provided step (default: self.action_frames) ====
    def _window_for_keystep(self, N: int, K: List[int], j: int, step: Optional[int] = None, tgt_offset: int = 0) -> List[int]:
        assert 0 <= j < len(K)
        k0 = K[j]
        k = k0 + int(tgt_offset)                  # Target frame = keystep + offset
        left_bound  = K[j-1] if j-1 >= 0 else -1
        right_bound = K[j+1] if j+1 < len(K) else N
        if step is None:
            step = max(1, int(self.action_frames))

        # Soft-protect invalid target frames, although __init__ should already have filtered them
        if not (left_bound < k < right_bound) or not (0 <= k < N):
            k = min(max(k, 0), min(right_bound-1, N-1))

        # Still walk backward from the target, subtracting step each time until left_bound is crossed
        ctx = []
        t = k - step
        while t > left_bound and t >= 0:
            ctx.append(t)
            t -= step
        ctx = sorted(set([x for x in ctx if 0 <= x < k]))

        win = ctx + [k]
        if len(win) > self.T:
            win = win[-self.T:]
        win = sorted(win)
        if win[-1] != k:
            if k not in win:
                win.append(k)
            win = sorted(win)
        return win

    # ---------- Normalize single-frame visual codes into [1,H,W] ----------
    # ==== New: npy loader with LRU caching ====
    def _load_npy_cached(self, path: str):
        if not self.use_mmap:
            return np.load(path, allow_pickle=False)
        arr = self._npy_cache.get(path)
        if arr is not None:
            self._npy_cache.move_to_end(path)
            return arr
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        self._npy_cache[path] = arr
        if len(self._npy_cache) > self.npy_cache_cap:
            self._npy_cache.popitem(last=False)
        return arr
    # [MOD] Single-frame loader (kept consistent with the previous version; skip if already present)
    def _load_single_frame_tokens(self, img_list, frame_idx):
        if getattr(self, "raw_image", False):
            self.image_tokenizer.eval()
            img_path = img_list[frame_idx]
            img = Image.open(img_path)
            px = self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
            with torch.no_grad():
                code = self.image_tokenizer.encode(px.unsqueeze(0))
            code = torch.as_tensor(code).squeeze()
            if code.dim() == 2:
                code = code.unsqueeze(0)                # [1,H,W]
            elif code.dim() == 3:
                if code.shape[0] == 1:
                    pass
                elif code.shape[-1] == 1:
                    code = code.permute(2, 0, 1)        # [H,W,1] -> [1,H,W]
                else:
                    code = code.permute(1, 2, 0).reshape(code.shape[1], -1).unsqueeze(0)
            else:
                code = code.view(1, 1, -1)
            return code
        else:
            arr = self._load_npy_cached(img_list[frame_idx])   # <<< updated here
            if not arr.flags.writeable:
                arr = arr.copy()  # Make a writable copy
            ten = torch.from_numpy(arr).squeeze()
            if ten.dim() == 2:
                ten = ten.unsqueeze(0)
            elif ten.dim() == 3:
                if ten.shape[0] == 1:
                    pass
                elif ten.shape[-1] == 1:
                    ten = ten.permute(2, 0, 1)
                else:
                    ten = ten.permute(1, 2, 0).reshape(ten.shape[1], -1).unsqueeze(0)
            else:
                ten = ten.view(1, 1, -1)
            return ten

    # [MOD] Batch tokenization: send multiple strings into the tokenizer at once and return per-segment token sequences
    def _batch_tokenize_segments(self, segments: List[str]):
        enc = self.tokenizer(
            segments,
            add_special_tokens=False,         # Do not insert special tokens inside segments
            padding=False,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors=None
        )
        return enc["input_ids"], enc["attention_mask"]   # list[list[int]], list[list[int]]


    # ---------- Assemble one sample ----------
    def __getitem__(self, index: int):
        # NEW: Select (si, j, g)
        if self.one_group_per_episode:
            # index is the episode slot, not the scene_idx value; fetch the true scene_idx first
            si = self._episode_ids[index]

            bucket = self._ep_buckets[si]
            # Simple approach: randomly pick one candidate on each access
            # For stricter reproducibility, replace this with hash(seed, epoch, si) once epoch is injected
            pick_flat = bucket[random.randrange(len(bucket))]
            scene_idx, j, g = self.flat_index[pick_flat]
        else:
            scene_idx, j, g = self.flat_index[index]
        scene = self.data[scene_idx]
        img_paths = scene["image"]
        N = len(img_paths)

        # [NEW] Soft-enable gripper input: require self.use_gripper=True and "gripper_image" to exist in the sample
        use_grip = bool(getattr(self, "use_gripper", False)) and ("gripper_image" in scene)
        grip_paths = scene.get("gripper_image", None) if use_grip else None
        # Soft fallback on length mismatch to avoid out-of-range access
        if use_grip and (not isinstance(grip_paths, (list, tuple)) or len(grip_paths) != N):
            use_grip, grip_paths = False, None

        # 1) Keep the text portion unchanged
        if getattr(self, "without_text", False):
            prompt = ""
        else:
            if self.cfg and random.random() < self.args.null_prompt_prob:
                prompt = ""
            else:
                prompt = scene.get("text", "")

        # 2) Compute the backward step for this episode (aligned with SFT)
        local_step = max(1, int(self.action_frames))
        if getattr(self, "post_training", False) and hasattr(self, "get_fps_for_path"):
            fps = self.get_fps_for_path(img_paths)
            if isinstance(fps, int) and fps > 0:
                local_step = fps

        # 3) Build the window (explicitly pass step)
        K = self.ep_keysteps[scene_idx]
        window = self._window_for_keystep(N, K, j, step=local_step, tgt_offset=g)
        ctx_idx, tgt_idx = window[:-1], window[-1]


        # ====== [FIXED] Batch tokenization + dynamic labels + tail padding (avoid shape mismatch) ======

        # 1) Collect window indices: context + target
        all_idx = ctx_idx + [tgt_idx]

        # 2) Load frame VQ tokens
        if getattr(self, "raw_image", False) and getattr(self, "batch_encode", True) and len(all_idx) > 0:
            frames_tokens = self._load_frames_tokens_batch(img_paths, all_idx)  # List[1,H,W]
            if use_grip:
                grip_tokens = self._load_frames_tokens_batch(grip_paths, all_idx)  # List[1,H,W]
        else:
            frames_tokens = [self._load_single_frame_tokens(img_paths, ii) for ii in all_idx]
            if use_grip:
                grip_tokens = [self._load_single_frame_tokens(grip_paths, ii) for ii in all_idx] # List[1,H,W]

        # === 3) Build segment strings (same as the current behavior) ===
        seg_text = self.tokenizer.bos_token + (prompt or "")
        seg_ctx_prompts = []
        for t in range(len(all_idx) - 1):
            p = self.format_video_prompt(frames_tokens[t])
            if use_grip:
                p += self.format_video_prompt(grip_tokens[t])
            seg_ctx_prompts.append(p)
        seg_tgt_prompt = self.format_video_prompt(frames_tokens[-1])
        if use_grip:
            seg_tgt_prompt += self.format_video_prompt(grip_tokens[-1])
        #seg_ctx_prompts = [self.format_video_prompt(frames_tokens[t]) for t in range(len(all_idx) - 1)]
        #seg_tgt_prompt  = self.format_video_prompt(frames_tokens[-1])
        segments = [seg_text] + seg_ctx_prompts + [seg_tgt_prompt]

        # === 4) Batch tokenize (same as the current behavior) ===
        seg_input_ids_list, seg_attn_list = self._batch_tokenize_segments(segments)  # list[list[int]]

        # === 5) Concatenate into one sequence (same as the current behavior) ===
        input_ids_list = []
        attention_mask_list = []
        for ids, attn in zip(seg_input_ids_list, seg_attn_list):
            input_ids_list.append(torch.tensor(ids,  dtype=torch.long))
            attention_mask_list.append(torch.tensor(attn, dtype=torch.long))

        input_ids      = torch.cat(input_ids_list, dim=0)      # [L]
        attention_mask = torch.cat(attention_mask_list, dim=0) # [L]

        # === [NEW] 5.1 Compute start/end offsets for each segment ===
        seg_lens = [len(x) for x in seg_input_ids_list]
        offsets  = [0]
        for L in seg_lens[:-1]:
            offsets.append(offsets[-1] + L)

        # Segment types: 0=text, 1..N-2=context segments, N-1=target segment
        text_span = (offsets[0], offsets[0] + seg_lens[0])
        ctx_spans = []
        for i in range(1, len(seg_lens) - 1):
            s, e = offsets[i], offsets[i] + seg_lens[i]
            ctx_spans.append((s, e))
        tgt_span = (offsets[-1], offsets[-1] + seg_lens[-1])

        # === [MOD] 5.2 Generate labels: whether to supervise context depends on args.supervise_context ===
        labels = torch.full((input_ids.size(0),), fill_value=self.args.ignore_index, dtype=torch.long)
        # The target segment is always supervised
        labels[tgt_span[0]:tgt_span[1]] = input_ids[tgt_span[0]:tgt_span[1]]
        # If context supervision is enabled, also label all context segments
        if bool(getattr(self.args, "supervise_context", False)):
            for (s, e) in ctx_spans:
                labels[s:e] = input_ids[s:e]

        # === [NEW] 5.3 Position weights: use different weights for target and context segments ===
        # Preserve the default behavior: when context supervision is off, context weight stays 0 and only the target uses weight 1 (or keystep_loss_weight)
        w_ctx = float(getattr(self.args, "ctx_loss_weight", 1.0))
        w_tgt = float(getattr(self.args, "keystep_loss_weight", 1.0))
        loss_weight = torch.zeros_like(input_ids, dtype=torch.float32)

        # Target-segment weight
        loss_weight[tgt_span[0]:tgt_span[1]] = w_tgt
        # Context-segment weight (only active when context supervision is enabled)
        if bool(getattr(self.args, "supervise_context", False)):
            for (s, e) in ctx_spans:
                loss_weight[s:e] = w_ctx

        # === 6) Optional vision-only filtering (same as the current behavior) ===
        if self.args.apply_loss_on_only_vision:
            vis_mask = torch.logical_and(input_ids >= self.bov, input_ids <= self.eov)
            labels = torch.where(vis_mask, labels, self.args.ignore_index)

        # Important: zero out weights on non-supervised positions so they do not contribute to loss
        loss_weight = loss_weight * (labels != self.args.ignore_index).to(loss_weight.dtype)

        # === 7) Left-truncate overlong sequences (same as current behavior, also truncate loss_weight) ===
        max_len = self.tokenizer.model_max_length
        overflow = input_ids.size(0) - max_len
        if overflow > 0:
            input_ids      = input_ids[-max_len:]
            attention_mask = attention_mask[-max_len:]
            labels         = labels[-max_len:]
            loss_weight    = loss_weight[-max_len:]
        # === 8) Right-pad to max_length (same as current behavior, pad loss_weight with 0 as well) ===
        sample = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="max_length", return_tensors="pt"
        )
        sample["input_ids"]      = sample["input_ids"].squeeze(0)
        sample["attention_mask"] = sample["attention_mask"].squeeze(0)

        if labels.size(0) < max_len:
            pad_len = max_len - labels.size(0)
            labels = torch.cat([labels, torch.full((pad_len,), self.args.ignore_index, dtype=labels.dtype)], dim=0)
            loss_weight = torch.cat([loss_weight, torch.zeros((pad_len,), dtype=loss_weight.dtype)], dim=0)

        sample["labels"] = labels
        sample["loss_weight"] = loss_weight  # Key: pass per-position weights to the model

        if index == 0:
            print(f"[sample] ctx_idx={ctx_idx}, tgt_idx={tgt_idx}", flush=True)
        return sample
        # ====== [FIXED] END ======
