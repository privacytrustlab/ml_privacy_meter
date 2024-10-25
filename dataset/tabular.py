import torch
from torch.utils.data import Dataset
import numpy as np


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The target labels.
        """
        self.data = X
        self.targets = y

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.targets)

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int or torch.Tensor): The index or tensor index of the sample to retrieve.
        Returns:
            A tuple (data, target), where both are converted to PyTorch tensors.
        """
        # Handle torch.Tensor index for compatibility with random_split
        if isinstance(idx, torch.Tensor):
            idx = idx.item()  # Get the integer value of a single-element tensor

        # Convert the data and target to tensors for PyTorch usage
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.long)

        return data, target
