import math
from typing import Optional, Iterator, Sequence
import random
import numpy as np

import torch.distributed as dist
from torch.utils.data import Sampler, DistributedSampler

from robovlms.data.concat_dataset import ConcatDataset


class DistributedWeightedSampler(Sampler[int]):
    r"""
    Modified from torch.utils.DistributedSampler.

    This dataloader can only be used together with the ConcatDataset defined in
    decision_transformer.data.concat_dataset. By using this sampler, datasets defined
    in the ConcatDataset will be loaded using the specified weights.

    The dataloader is designed to follow the principles:
    1. In each epoch, each dataset should be sampled by at least 'min_num_samples' times,
       to avoid the situation that if a dataset weight is set to a small value, it can be
       hardly sampled.
    2. Every sample in the dataset will be sampled as uniformly as possible.
    3. When evaluation, this sampler can degenerate to the normal DistributedSampler.

    Args:

    sample_per_epoch (int)
        The number of samples to format one training epoch. If not specified, it will be inferred
        from the ConcatDataset and specified sampling weights.

    no_shuffle_no_weighted (bool)
        If set to True, the datasets in the ConcatDataset will not be weighted
        during sampling as long as the 'shuffle' is False.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        weights: Optional[Sequence[float]] = None,
        sample_per_epoch: Optional[int] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        is_training=True,
    ) -> None:
        super().__init__(None)

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )

        self.is_training = is_training
        self.sample_per_epoch = sample_per_epoch
        self.weights = None
        if weights is not None:
            assert len(weights) == len(dataset.datasets)
            self.weights = np.array(weights) / np.array(weights).sum()
        self.dataset = dataset
        self.num_datasets = len(self.dataset.datasets)
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self._init_num_samples()

    def _refine_num_samples_per_dataset(
        self, original_samples_per_dataset: Sequence[int]
    ):
        """
        The datasets may contain number of samples that can not be perfectly divided by self.num_replicas.
        This function will refine the number of samples per datasets so that the total size can fulfill
        this condition.
        """
        if self.is_training and self.sample_per_epoch is not None:
            # scaling the number of samples per dataset according to the expected sample number per epoch
            total_num_samples = sum(original_samples_per_dataset)
            original_samples_per_dataset = [
                math.ceil(n * self.sample_per_epoch / total_num_samples)
                for n in original_samples_per_dataset
            ]

        total_num_samples = sum(original_samples_per_dataset)

        if self.drop_last and total_num_samples % self.num_replicas != 0:  # type: ignore[arg-type]
            total_drop_num_samples = total_num_samples % self.num_replicas
            n_drop = math.floor(total_drop_num_samples / self.num_datasets)
            # for the first total_drop_num_samples % self.num_datasets datasets, drop 1 more sample
            num_samples = [
                (
                    n - n_drop - 1
                    if i < total_drop_num_samples % self.num_datasets
                    else n - n_drop
                )
                for i, n in enumerate(original_samples_per_dataset)
            ]
            assert sum(num_samples) == total_num_samples - total_drop_num_samples

        else:
            total_add_num_samples = (
                self.num_replicas - total_num_samples % self.num_replicas
            )
            n_add = math.floor(total_add_num_samples / self.num_datasets)
            # for the first total_add_num_samples % self.num_datasets datasets, add 1 more sample
            num_samples = [
                (
                    n + n_add + 1
                    if i < total_add_num_samples % self.num_datasets
                    else n + n_add
                )
                for i, n in enumerate(original_samples_per_dataset)
            ]
            assert sum(num_samples) == total_num_samples + total_add_num_samples

        return num_samples

    def _init_num_samples_unweighted(self):
        num_samples = self.dataset.num_samples
        num_samples = self._refine_num_samples_per_dataset(num_samples)
        self.num_samples_per_dataset = num_samples
        self.total_size = sum(self.num_samples_per_dataset)
        assert self.total_size % self.num_replicas == 0
        self.num_samples = int(self.total_size / self.num_replicas)

    def _init_num_samples_weighted(self):
        min_weight = np.min(self.weights)
        min_weight_idx = np.argmin(self.weights)
        min_num_sample = self.dataset.num_samples[min_weight_idx]
        # make sure the smallest dataset gets
        total_num_samples = min_num_sample / min_weight
        num_samples = [int(total_num_samples * weight) for weight in self.weights]
        # make sure that the total_num_samples is an integer.
        num_samples = self._refine_num_samples_per_dataset(num_samples)
        self.num_samples_per_dataset = num_samples
        self.total_size = sum(self.num_samples_per_dataset)
        assert self.total_size % self.num_replicas == 0
        self.num_samples = int(self.total_size / self.num_replicas)

    def _init_num_samples(self):
        if (not self.is_training and not self.shuffle) or self.weights is None:
            self._init_num_samples_unweighted()
        else:
            self._init_num_samples_weighted()

        if self.rank == 0:
            print("=" * 40)
            print("DISTRIBUTED WEIGHTED SAMPLER:")
            print(f"Samples per epoch: {self.sample_per_epoch}")
            print(f"Datasets:")
            print(self.dataset)
            print(f"Sampling weights: {self.weights}")
            print(f"Number of samples by dataset: {self.num_samples_per_dataset}")
            print("=" * 40)

    def _add_idx_shift(self, indices, dataset_idx):
        if dataset_idx == 0:
            shift = 0
        else:
            shift = self.dataset.cumulative_sizes[dataset_idx - 1]
        return [i + shift for i in indices]

    def _get_indices(self):
        indices_per_dataset = []
        for dataset_idx, (num_sample, data_len) in enumerate(
            zip(self.num_samples_per_dataset, self.dataset.num_samples)
        ):
            repeat = int(num_sample / data_len)
            indices = list(range(data_len)) * repeat
            # padding for sampled weight
            padding_size = num_sample % data_len
            if padding_size > 0:
                if self.shuffle:
                    indices += random.sample(range(data_len), padding_size)
                else:
                    indices += list(range(padding_size))
            indices = self._add_idx_shift(indices, dataset_idx)
            indices_per_dataset.extend(indices)

        if self.shuffle:
            # shuffle the indices
            random.shuffle(indices_per_dataset)

        return indices_per_dataset

    def __iter__(self) -> Iterator[int]:
        # set random seed
        _seed = self.seed + self.epoch
        random.seed(_seed)
        indices = self._get_indices()
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


if __name__ == "__main__":
    data_num = [50, 1000, 10000]
    weights = [3.0, 1.0, 1.0]

    dataset_configs = [
        dict(
            type="DummyDataset",
            num_samples=data_num[i],
        )
        for i in range(3)
    ]

    concat_dataset = ConcatDataset(dataset_configs)
    dataloader = DistributedWeightedSampler(
        dataset=concat_dataset, num_replicas=8, rank=0, shuffle=False
    )

    indices = []
    for i in dataloader:
        indices.append(i)
    print(sorted(indices))
    print(len(indices))
