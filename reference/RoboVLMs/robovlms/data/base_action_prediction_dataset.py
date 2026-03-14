from dataclasses import dataclass
import numpy as np
from PIL import Image
from typing import Any, Dict, Callable, List, Tuple, Union, Literal, Optional, Sequence

import torch
import torch.nn.functional as F

from transformers import PreTrainedTokenizerBase

from robovlms.data.base_task_dataset import BaseTaskDataset, IGNORE_INDEX

from robovlms.data.data_utils import (
    get_prompt_builder,
    get_tensor_chunk,
    mu_law_companding,
    normalize_action,
    pad_sequences,
    regularize_action,
)
from robovlms.model.policy_head.action_tokenizer import ActionTokenizer


@dataclass
class ActionPredictionBatchTransform:
    """
    Transform one item of dataset
    """

    model_name: str
    tokenizer: PreTrainedTokenizerBase
    text_fn: Callable
    image_fn: Callable[[List[Image.Image]], torch.Tensor]

    window_size: int
    fwd_pred_next_n: int
    predict_stop_token: bool

    organize_type: str
    image_history: bool
    action_history: bool
    discrete: bool
    action_tokenizer: Optional[ActionTokenizer]
    special_history_id: int
    mode: str

    norm_action: bool
    norm_min: float
    norm_max: float
    x_mean: float
    x_std: float
    regular_action: bool
    use_mu_law: bool
    min_action: float
    max_action: float

    @staticmethod
    def refine_action_at_gripper_dim(
        action: Union[np.ndarray, torch.Tensor], value: int = -1, status: bool = False
    ):
        """
        make the open gripper action state as value (0 or 1)
        """
        if isinstance(action, np.ndarray):
            action = action.copy()
        elif isinstance(action, torch.Tensor):
            action = action.clone()
        else:
            raise TypeError("The type of action must be ndarray or tensor")
        gripper_action = action[..., -1]
        if status:
            gripper_action[gripper_action == 1] = value
        else:
            gripper_action[gripper_action != 1] = value
        return action

    def convert_image(
        self,
        images: Optional[np.ndarray],
        image_mask: torch.Tensor,
        static: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if images is None:
            return None, None, None

        if not self.image_history:
            image_tensors = self.image_fn(
                [Image.fromarray(images[self.window_size - 1])], static=static
            )
            return image_tensors, None, None

        image_tensors = self.image_fn(
            [Image.fromarray(each_image) for each_image in images], static=static
        )

        # you can't get chunk image in the segment dataset because segment dataset will padding in the left side
        if self.organize_type == "segment":
            return image_tensors, None, None

        left_pad_index = self.window_size - image_mask[: self.window_size].sum()
        image_tensors[:left_pad_index] = image_tensors[left_pad_index]

        # this chunk is to predict next fwd_pred_next_n images, it is based on one image, so we need to skip the first one which including image0
        image_chunk = get_tensor_chunk(image_tensors, self.fwd_pred_next_n)[1:]
        image_chunk_mask = get_tensor_chunk(image_mask, self.fwd_pred_next_n)[1:]

        image_tensors = image_tensors[: self.window_size]
        return image_tensors, image_chunk, image_chunk_mask

    def convert_action(self, action: np.ndarray, action_mask: torch.Tensor):
        # ACTION
        if self.mode == "train":
            # the act step set to fwd_pred_next_n + 1, it will get one more action which we should drop it
            action = action[:-1]
            action_mask = action_mask[:-1]
        else:
            # in inference, this mask will be give by the image mask, which is one more than action action (we have current image but don't know current action)
            action_mask = action_mask[:-1]

        if self.norm_action:
            action = normalize_action(
                action, self.norm_min, self.norm_max, maintain_last=True
            )
        if self.regular_action:
            action = regularize_action(action, self.x_mean, self.x_std)
        if self.use_mu_law:
            action = mu_law_companding(action)
        action = self.refine_action_at_gripper_dim(action, value=0)
        action = torch.tensor(action)
        if self.mode != "train":
            return action, action_mask, None, None

        action_chunk = get_tensor_chunk(action, self.fwd_pred_next_n)
        action_chunk_mask = get_tensor_chunk(action_mask, self.fwd_pred_next_n)
        return action, action_mask, action_chunk, action_chunk_mask

    def get_right_pad_len(self, action_chunk_mask: np.ndarray, action_dim: int):
        right_chunk_mask = action_chunk_mask[-self.fwd_pred_next_n :]
        return (right_chunk_mask.shape[0] - right_chunk_mask.sum()) * action_dim

    def cat_input_ids_and_action_ids(
        self,
        input_ids: List[int],
        action_ids: List[int],
        eos_token_id: Optional[int],
        right_pad_len: int,
    ):
        eos_offset = 1 if eos_token_id is not None else 0
        input_ids = (
            input_ids[: len(input_ids) - eos_offset]
            + action_ids
            + input_ids[len(input_ids) - eos_offset :]
        )

        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        attention_masks = torch.ones_like(input_ids, dtype=bool)

        action_id_start_index = len(input_ids) - len(action_ids) - eos_offset
        action_id_end_index = len(input_ids) - eos_offset
        action_pad_start_index = len(input_ids) - right_pad_len - eos_offset
        action_pad_end_index = len(input_ids) - eos_offset

        labels[:action_id_start_index] = IGNORE_INDEX
        attention_masks[action_id_start_index:action_id_end_index] = False
        labels[action_pad_start_index:action_pad_end_index] = IGNORE_INDEX

        if (right_pad_len != 0) or (
            not self.predict_stop_token and self.tokenizer.eos_token
        ):
            labels[-1] = IGNORE_INDEX
        return input_ids, labels, attention_masks

    def wrap_instruction_and_action_interleave(
        self, task_description: str, action: torch.Tensor, action_mask: torch.Tensor
    ):
        if self.mode == "train":
            assert action.shape[0] == self.window_size + self.fwd_pred_next_n - 1
            window_size = self.window_size
        else:
            window_size = action.shape[0] + 1

        prompt_builder = get_prompt_builder(
            self.model_name, eos=self.tokenizer.eos_token, bos=self.tokenizer.bos_token
        )
        action_mask = action_mask.bool()
        action_dim = action.shape[1]
        action = self.refine_action_at_gripper_dim(
            action, value=self.min_action, status=False
        )
        action = self.refine_action_at_gripper_dim(
            action, value=self.max_action, status=True
        )
        action = action.flatten()
        conversation = [
            {
                "from": "human",
                "value": (
                    f"What action should the robot take to {task_description}?"
                    if self.fwd_pred_next_n == 1
                    else f"What {self.fwd_pred_next_n} step actions should the robot take to {task_description}?"
                ),
            },
            {"from": "gpt", "value": ""},
        ]

        input_ids = []
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        if (
            self.tokenizer.eos_token is not None
            and self.tokenizer.eos_token_id != input_ids[-1]
        ):
            input_ids = input_ids + [self.tokenizer.eos_token_id]

        if not self.discrete:
            all_input_ids = torch.tensor(input_ids)
            all_labels = all_input_ids
            return all_input_ids, all_labels, torch.ones_like(all_input_ids, dtype=bool)

        action_ids = self.action_tokenizer.encode_actions_to_token_ids(action)

        all_input_ids = []
        all_labels = []
        all_masks = []

        for i in range(window_size):
            start_action_index = i
            end_action_index = i + self.fwd_pred_next_n
            tmp_action_ids = action_ids[
                start_action_index * action_dim : end_action_index * action_dim
            ]
            tmp_action_mask = action_mask[start_action_index:end_action_index]
            right_pad_len = self.get_right_pad_len(tmp_action_mask, action_dim)

            tmp_input_ids, tmp_labels, tmp_masks = self.cat_input_ids_and_action_ids(
                input_ids, tmp_action_ids, self.tokenizer.eos_token_id, right_pad_len
            )
            all_input_ids.append(tmp_input_ids)
            all_labels.append(tmp_labels)
            all_masks.append(tmp_masks)

        all_input_ids = torch.stack(all_input_ids)
        all_labels = torch.stack(all_labels)
        all_masks = torch.stack(all_masks)
        return all_input_ids, all_labels, all_masks

    def wrap_instruction_and_action_segment(
        self, task_description, action: torch.Tensor, action_mask: torch.Tensor
    ):
        prompt_builder = get_prompt_builder(
            self.model_name, eos=self.tokenizer.eos_token, bos=self.tokenizer.bos_token
        )
        # if pass in multi-step actions, we concat them
        if self.mode == "train":
            assert action.shape[0] == self.fwd_pred_next_n + self.window_size - 1
            window_size = self.window_size
        else:
            # assert action_chunk.shape[0] == self.window_size - 1
            window_size = action.shape[0] + 1

        if self.action_history:
            history_action = action[: window_size - 1]
            history_mask = action_mask[: window_size - 1]
            history_action = history_action[history_mask]
            history_len = history_mask.sum()
        else:
            history_action = np.zeros((0, action.shape[1]))
            history_len = 0

        next_action = action[window_size - 1 :]
        next_action_mask = action_mask[window_size - 1 :]

        action_dim = action.shape[1]
        history_action = history_action.flatten()
        next_action = next_action.flatten()

        if history_len == 1:
            history_prompt = (
                f"Here is {history_len} step action that the robot has taken: "
            )
        elif history_len > 1:
            history_prompt = (
                f"Here are {history_len} step actions that the robot has taken: "
            )
        else:
            history_prompt = ""

        if self.fwd_pred_next_n == 1:
            question_prompt = (
                f"What action should the robot take to {task_description}?"
            )
        else:
            question_prompt = f"What {self.fwd_pred_next_n} step actions should the robot take to {task_description}?"

        conversation = [
            {"from": "human", "value": history_prompt + question_prompt},
            {"from": "gpt", "value": ""},
        ]

        input_ids = []
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        prompt: str = prompt_builder.get_prompt()
        add_special_tokens = True

        if history_len > 0:
            prefix_index = prompt.find(history_prompt) + len(history_prompt)
            prefix_prompt = prompt[:prefix_index]
            prompt = prompt[prefix_index:]
            input_ids = self.tokenizer(
                prefix_prompt, add_special_tokens=add_special_tokens
            ).input_ids
            add_special_tokens = False
            if self.discrete:
                history_action_ids = self.action_tokenizer.encode_actions_to_token_ids(
                    history_action
                )
            else:
                history_action_ids = [
                    self.special_history_id for _ in range(history_len)
                ]
            input_ids += history_action_ids

        input_ids += self.tokenizer(
            prompt, add_special_tokens=add_special_tokens
        ).input_ids
        if (
            self.tokenizer.eos_token is not None
            and self.tokenizer.eos_token_id != input_ids[-1]
        ):
            input_ids = input_ids + [self.tokenizer.eos_token_id]

        if self.discrete:
            next_action_ids = self.action_tokenizer.encode_actions_to_token_ids(
                next_action
            )
            right_pad_len = self.get_right_pad_len(next_action_mask, action_dim)
            input_ids, labels, _ = self.cat_input_ids_and_action_ids(
                input_ids, next_action_ids, self.tokenizer.eos_token, right_pad_len
            )
        else:
            input_ids = torch.tensor(input_ids)
            labels = input_ids

        if self.mode != "train" and self.tokenizer.eos_token is not None:
            input_ids = input_ids[:-1]
            labels = labels[:-1]

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        return input_ids, labels, torch.ones_like(input_ids, dtype=bool)

    def __call__(
        self,
        task_description: str,
        action: np.ndarray,
        episode_mask: np.ndarray,
        images: np.ndarray,
        gripper_images: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Converts an item to the format expected by collator/models."""
        episode_mask = torch.tensor(episode_mask)

        # Pad in Image and action tensors
        image_tensors, image_chunk, image_chunk_mask = self.convert_image(
            images, episode_mask
        )
        gripper_image_tensors, gripper_image_chunk, _ = self.convert_image(
            gripper_images, episode_mask, static=False
        )

        # ACTION TENSORS
        action, action_mask, action_chunk, action_chunk_mask = self.convert_action(
            action, episode_mask
        )

        # INPUT IDS (OPTIONAL WITH DISCRETE ACTION IDS)
        if self.organize_type == "interleave":
            (
                input_ids,
                labels,
                attention_mask,
            ) = self.wrap_instruction_and_action_interleave(
                task_description, action, action_mask
            )
        elif self.organize_type == "segment":
            (
                input_ids,
                labels,
                attention_mask,
            ) = self.wrap_instruction_and_action_segment(
                task_description, action, action_mask
            )
        else:
            raise TypeError("The organize type must be interleave or segment")

        return dict(
            image_tensors=image_tensors,
            image_chunk=image_chunk,
            image_chunk_mask=image_chunk_mask,
            gripper_image_tensors=gripper_image_tensors,
            gripper_image_chunk=gripper_image_chunk,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            action_tensors=action,
            action_mask=action_mask,
            action_chunk=action_chunk,
            action_chunk_mask=action_chunk_mask,
        )


@dataclass
class ActionPredictionPaddedCollator:
    pad_token_id: int
    fwd_pred_next_n: int
    window_size: int
    organize_type: str
    discrete: bool = False

    def __call__(
        self, instances: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        (
            image_tensors,
            image_chunk,
            image_chunk_mask,
            gripper_image_tensors,
            gripper_image_chunk,
            input_ids,
            labels,
            attention_mask,
            action_tensors,
            action_mask,
            action_chunk,
            action_chunk_mask,
        ) = tuple(
            [instance[key] for instance in instances]
            for key in (
                "image_tensors",
                "image_chunk",
                "image_chunk_mask",
                "gripper_image_tensors",
                "gripper_image_chunk",
                "input_ids",
                "labels",
                "attention_mask",
                "action_tensors",
                "action_mask",
                "action_chunk",
                "action_chunk_mask",
            )
        )
        input_ids = pad_sequences(input_ids, self.pad_token_id)
        labels = pad_sequences(labels, IGNORE_INDEX)
        attention_mask = pad_sequences(attention_mask, False)

        image_tensors = torch.stack(image_tensors)
        gripper_image_tensors = (
            torch.stack(gripper_image_tensors)
            if gripper_image_tensors[0] is not None
            else None
        )
        image_chunk = torch.stack(image_chunk) if image_chunk[0] is not None else None
        image_chunk_mask = (
            torch.stack(image_chunk_mask) if image_chunk_mask[0] is not None else None
        )
        gripper_image_chunk = (
            torch.stack(gripper_image_chunk)
            if gripper_image_chunk[0] is not None
            else None
        )
        action_tensors = torch.stack(action_tensors)
        action_mask = torch.stack(action_mask)

        # if not self.organize_type == "segment":
        #     action_chunk = action_tensors[:, -self.fwd_pred_next_n:]
        #     action_chunk_mask = action_mask[:, -self.fwd_pred_next_n:]
        # else:
        #     action_chunk = torch.stack(action_chunk)
        #     action_chunk_mask = torch.stack(action_chunk_mask)
        # import pdb; pdb.set_trace()
        action_chunk = torch.stack(action_chunk)
        action_chunk_mask = torch.stack(action_chunk_mask)

        output = {
            "rgb": image_tensors,
            "hand_rgb": gripper_image_tensors,
            "fwd_rgb_chunck": image_chunk,
            "fwd_hand_rgb_chunck": gripper_image_chunk,
            "fwd_mask": image_chunk_mask,
            "text": input_ids,
            "text_mask": attention_mask,
            "action": action_tensors,
            "action_mask": action_mask,
            "action_chunck": action_chunk,
            "chunck_mask": action_chunk_mask,
            "instr_and_action_ids": input_ids,
            "instr_and_action_labels": labels,
            "instr_and_action_mask": attention_mask,
        }
        return output


class ActionPredictionDataset(BaseTaskDataset):
    """
    Abstract dataset base class.

    Args:
        num_workers: Number of dataloading workers for this dataset.
        batch_size: Batch size.
    """

    def __init__(
        self,
        model_name: str = "flamingo",
        mode: Literal["train", "inference"] = "train",
        organize_type: Literal["interleave", "segment"] = "interleave",
        discrete: bool = True,
        action_history: bool = True,
        image_history: bool = True,
        predict_stop_token: bool = True,
        special_history_id: int = IGNORE_INDEX,
        window_size: int = 16,
        fwd_pred_next_n: int = 2,
        n_bin=256,
        min_action=-1,
        max_action=1,
        norm_action: bool = False,
        norm_min: int = -1,
        norm_max: int = 1,
        regular_action: bool = False,
        x_mean: int = 0,
        x_std: int = 1,
        use_mu_law: bool = False,
        **kwargs,
    ):
        """
        Args:
            model_name: this value will use to build different prompt builder for different model, it will pass to get_prompt_builder function
            mode: the mode of this dataset, "train" or "inference", it will cause different data flow
            organize_type: the type you organize your output data, if you set interleave, it will be [batch size, window size, language token length + action token length(optional)],
                           else it will be [batch size, history image token length + language token length + history action token length(optional) + next action token length]
            discrete: set True if you want discrete the action to language token space
            action_history: only valid when the organize_type='segment', and if you set it False, you output data will not contain history action token
            image_history: only valid when the organize_type='segment', and if you set it False, you output data will only contain one image, else the image number will equal to window size
            predict_stop_token: only valid when the discrete=True, set True if you want the model to predict the <eos> token
            special_history_id: only valid when discrete=False and organize_type=segment, it will be the placement of the action embeding in the instr_and_action_ids

            window_size: the history length of the image / action
            fwd_pred_next_n: we need to predict fwd_pred_next_n images / actions

            n_bin: How many bins is the interval of action divided into
            min_action: the min action numerical value, if any action is lower than this, we will set these action to min_action
            max_action: the max action numerical value.

            norm_action: set True if you want to normalize the action space
            norm_min: the min action value in normalize space
            norm_max: the max action value in normalize space
            regular_action: set True if you want to regularize the action space
            x_mean: the mean action value of regular action space
            x_std: the std value of regular action space
            use_mu_law: set True if you want to use mu_law
        """
        (
            self.model_name,
            self.mode,
            self.organize_type,
            self.discrete,
            self.image_history,
            self.action_history,
            self.predict_stop_token,
            self.special_history_id,
        ) = (
            model_name,
            mode,
            organize_type,
            discrete,
            image_history,
            action_history,
            predict_stop_token,
            special_history_id,
        )

        self.window_size, self.fwd_pred_next_n = window_size, fwd_pred_next_n

        (
            self.norm_action,
            self.norm_min,
            self.norm_max,
            self.regular_action,
            self.x_mean,
            self.x_std,
            self.use_mu_law,
        ) = (norm_action, norm_min, norm_max, regular_action, x_mean, x_std, use_mu_law)

        self.n_bin, self.min_action, self.max_action = n_bin, min_action, max_action
        kwargs["task_type"] = "action"
        super().__init__(**kwargs)

    def init_batch_transform(self):
        if self.discrete:
            self.action_tokenizer = ActionTokenizer(
                self.tokenizer,
                bins=self.n_bin,
                min_action=self.min_action,
                max_action=self.max_action,
            )
        else:
            self.action_tokenizer = None

        return ActionPredictionBatchTransform(
            action_tokenizer=self.action_tokenizer,
            special_history_id=self.special_history_id,
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            text_fn=self.text_fn,
            image_fn=self.image_fn,
            window_size=self.window_size,
            fwd_pred_next_n=self.fwd_pred_next_n,
            predict_stop_token=self.predict_stop_token,
            organize_type=self.organize_type,
            discrete=self.discrete,
            image_history=self.image_history,
            action_history=self.action_history,
            mode=self.mode,
            norm_action=self.norm_action,
            norm_min=self.norm_min,
            norm_max=self.norm_max,
            x_mean=self.x_mean,
            x_std=self.x_std,
            regular_action=self.regular_action,
            use_mu_law=self.use_mu_law,
            min_action=self.min_action,
            max_action=self.max_action,
        )

    def init_collater_fn(self):
        # use or to avoid the attr exists but the value is None
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        return ActionPredictionPaddedCollator(
            pad_token_id=pad_token_id,
            window_size=self.window_size,
            fwd_pred_next_n=self.fwd_pred_next_n,
            discrete=self.discrete,
            organize_type=self.organize_type,
        )
