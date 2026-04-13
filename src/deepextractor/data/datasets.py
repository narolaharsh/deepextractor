import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Dataset for 1-D time-series arrays stored as .npy files.

    Each array is expected to have shape ``(N, L)`` where *N* is the number
    of samples and *L* is the signal length.  Items are returned as
    ``(1, L)`` float32 tensors (channel dimension added).
    """

    def __init__(self, input_npy, target_npy, transform=None):
        self.inputs = np.load(input_npy)
        self.targets = np.load(target_npy)
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).float().unsqueeze(0)
        y = torch.from_numpy(self.targets[idx]).float().unsqueeze(0)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y


class SpectrogramDataset(Dataset):
    """Dataset for 2-D spectrogram arrays stored as .npy files.

    Each array is expected to have shape ``(N, H, W)``.  Items are returned
    as ``(1, H, W)`` float32 tensors (channel dimension added).
    """

    def __init__(self, input_npy, target_npy, transform=None):
        self.inputs = np.load(input_npy)
        self.targets = np.load(target_npy)
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).float().unsqueeze(0)
        y = torch.from_numpy(self.targets[idx]).float().unsqueeze(0)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y
