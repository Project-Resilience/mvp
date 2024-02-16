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

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]