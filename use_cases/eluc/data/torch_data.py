"""
A simple custom PyTorch dataset is created here. This is used to keep our
datasets standard between models. It is used in both Torch prescription
and Neural Network training.
"""

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

class TorchDataset(Dataset):
    """
    Simple custom torch dataset.
    :param X: data
    :param y: labels
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, device="cpu"):
        super().__init__()   
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, device=device)
        assert len(self.X) == len(self.y), "X and y must have the same length"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]