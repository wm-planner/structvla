from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """
    only for debugging the samplers
    """

    def __init__(self, num_samples=0, mode="train"):
        self.data = list(range(num_samples))
        self.data_length = len(self.data)

    def __len__(self):
        return self.data_length

    def __getitem__(self, item):
        return self.data[item]
