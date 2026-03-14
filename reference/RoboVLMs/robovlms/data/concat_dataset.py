import warnings
from typing import List, Dict
import numpy as np
import bisect

import torch
from torch.utils.data import ConcatDataset as _ConcatDataset

import robovlms
from robovlms.utils.dist_train import get_rank
from robovlms.data.data_utils import generate_chunck_data


class ConcatDataset(_ConcatDataset):
    def __init__(
        self, datasets: List[Dict], min_sampled_num: int = 0, is_training=True, **kwargs
    ):
        """
        min_sampled_num: the minimum sample number for each dataset during each epoch.
        """
        self.dataset_configs = datasets
        self.min_num_sample = min_sampled_num
        self.is_training = is_training
        # these args will be shared across all datasets in this ConcatDataset
        self.kwargs = kwargs

        self._init_datasets()
        super().__init__(self.datasets)
        # overwrite the default cumulative_sizes
        self.cumulative_sizes = self.cumsum(self.datasets, self.min_num_sample)

    @staticmethod
    def cumsum(sequence, min_sampled_num=0):
        r, s = [], 0
        for e in sequence:
            l = max(len(e), min_sampled_num)
            r.append(l + s)
            s += l
        return r

    def __str__(self):
        info_str = ""
        for dataset in self.datasets:
            info_str += f"{dataset}\n"
        info_str += f"Minimum #sample per dataset: {self.min_num_sample}\n"
        info_str += (
            f"Cumulative sizes (the last item is the length): {self.cumulative_sizes}"
        )
        return info_str

    def _init_datasets(self) -> None:
        self.datasets = []
        for configs in self.dataset_configs:
            name = configs.pop("type")
            configs["is_training"] = self.is_training

            # update configs by self.kwargs
            for k in self.kwargs:
                if k in configs:
                    if get_rank() == 0:
                        warnings.warn(
                            f"Keyword args already specified: \n\t{k}: {configs[k]}. "
                            f"Not changed by the shared args."
                        )
                else:
                    configs[k] = self.kwargs[k]

            self.datasets.append(getattr(robovlms.data, name)(**configs))

        if self.min_num_sample > 0:
            self.num_samples = np.array(
                [max(len(d), self.min_num_sample) for d in self.datasets], dtype=int
            )
        else:
            self.num_samples = np.array([len(d) for d in self.datasets], dtype=int)
            self.min_num_sample = min(self.num_samples)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % len(self.datasets[dataset_idx])
        return self.datasets[dataset_idx][sample_idx]

    def collater(self, data):
        # action_tensors = torch.from_numpy(np.array([np.stack(s["action"]) for s in data]))
        # print(data)
        # return self.datasets[0].collater(data)
        action_tensors = (
            torch.stack([s["action"] for s in data], dim=0)
            if data[0]["action"] is not None
            else None
        )
        image_tensors = torch.stack([s["rgb"] for s in data])
        image_mask = torch.stack([s["attention_mask"] for s in data])
        gripper_tensors = (
            torch.stack([s["hand_rgb"] for s in data])
            if data[0]["hand_rgb"] is not None
            else None
        )

        fwd_rgb_chunck = generate_chunck_data(
            image_tensors, self.window_size, self.fwd_pred_next_n
        )
        fwd_hand_rgb_chunck = generate_chunck_data(
            gripper_tensors, self.window_size, self.fwd_pred_next_n
        )
        chunck_mask = generate_chunck_data(
            image_mask, self.window_size, self.fwd_pred_next_n
        )

        action_chunck = generate_chunck_data(
            action_tensors, self.window_size, self.fwd_pred_next_n
        )

        stacked_language = [s["raw_text"] for s in data]
        text_tensors, text_mask = self.text_fn(stacked_language)

        res = {
            "rgb": image_tensors,
            "attention_mask": image_mask,
            "hand_rgb": gripper_tensors,
            "action": action_tensors,
            "text": text_tensors,
            "text_mask": text_mask,
            "fwd_rgb_chunck": fwd_rgb_chunck,
            "fwd_hand_rgb_chunck": fwd_hand_rgb_chunck,
            "action_chunck": action_chunck,
            "chunck_mask": chunck_mask,
        }

        # return image_tensors, (text_tensors, text_mask), action_tensors, gripper_tensors, image_mask,\
        #     fwd_rgb_chunck, fwd_hand_rgb_chunck, action_chunk
        return res
