import random
import numpy as np

from lightning.pytorch.utilities.combined_loader import (
    CombinedLoader,
    _SUPPORTED_MODES,
    _Sequential,
)


class WeightedCombinedLoader(CombinedLoader):
    def __init__(self, loaders, mode="max_size_cycle", weights=None):
        # Use the list directly as loaders
        super().__init__(loaders, mode)

        if weights is None:
            # raise ValueError("You must provide weights for the DataLoaders.")
            weights = [1] * len(loaders)

        # Normalize the weights to sum to 1
        self.weights = np.array(weights) / np.sum(weights)
        self.loader_iters = None

    def __iter__(self):
        cls = _SUPPORTED_MODES[self._mode]["iterator"]
        iterator = cls(self.flattened, self._limits)
        iter(iterator)
        self._iterator = iterator

        # Initialize each DataLoader's iterator directly from the list
        self.loader_iters = [iter(loader) for loader in self.flattened]
        return self

    def __next__(self):
        if self.loader_iters is None or not self.loader_iters:
            raise StopIteration

        # Randomly choose a DataLoader index based on the given weights
        selected_loader_idx = random.choices(
            range(len(self.loader_iters)), weights=self.weights, k=1
        )[0]
        selected_loader_iter = self.loader_iters[selected_loader_idx]

        try:
            # Get the next batch from the selected DataLoader
            batch = next(selected_loader_iter)
        except StopIteration:
            # # If the selected DataLoader is exhausted, remove it and try again
            # del self.loader_iters[selected_loader_idx]
            # del self.weights[selected_loader_idx]
            # if not self.loader_iters:
            #     raise StopIteration
            # self.weights = np.array(self.weights) / np.sum(self.weights)  # Normalize remaining weights
            # return self.__next__()
            self.loader_iters[selected_loader_idx] = iter(
                self.flattened[selected_loader_idx]
            )
            batch = next(self.loader_iters[selected_loader_idx])

        # Format the output to match the expected structure
        if isinstance(self._iterator, _Sequential):
            return batch

        batch_idx = 0  # Placeholder value, update as needed
        # return batch, batch_idx, selected_loader_idx
        return batch
