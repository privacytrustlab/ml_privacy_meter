"""This file contains functions for loading the dataset"""
import math
import os
import pickle
from ast import List

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from fast_train import get_batches, get_cifar10_data
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, X, y):
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        self.data = X
        self.targets = y

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.data[idx], self.targets[idx]]


class InfiniteRepeatDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


def get_dataset(dataset_name: str, data_dir: str):
    """Load the dataset from the pickle file or download it from the internet.
    Args:
        dataset_name (str): Dataset name
        data_dir (str): Indicate the log directory for loading the dataset

    Raises:
        NotImplementedError: Check if the dataset has been implemented.

    Returns:
        torchvision.datasets: Whole dataset.
    """
    path = f"{data_dir}/{dataset_name}"
    if os.path.exists(f"{path}.pkl"):
        with open(f"{path}.pkl", "rb") as file:
            all_data = pickle.load(file)
        print(f"Load data from {path}.pkl")

    else:
        if dataset_name == "cifar10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            all_data = torchvision.datasets.CIFAR10(
                root=path, train=True, download=True, transform=transform
            )
            test_data = torchvision.datasets.CIFAR10(
                root=path, train=False, download=True, transform=transform
            )
            all_features = np.concatenate([all_data.data, test_data.data], axis=0)
            all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
            all_data.data = all_features
            all_data.targets = all_targets
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            print(f"Save data to {path}.pkl")
        elif dataset_name == "cifar100":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            all_data = torchvision.datasets.CIFAR100(
                root=path, train=True, download=True, transform=transform
            )
            test_data = torchvision.datasets.CIFAR100(
                root=path, train=False, download=True, transform=transform
            )
            all_features = np.concatenate([all_data.data, test_data.data], axis=0)
            all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
            all_data.data = all_features
            all_data.targets = all_targets
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            print(f"Save data to {path}.pkl")
        elif dataset_name == "purchase100":
            if os.path.exists("../data/purchase/dataset_purchase"):
                df = pd.read_csv(
                    "../data/purchase/dataset_purchase", header=None, encoding="utf-8"
                ).to_numpy()
                y = df[:, 0] - 1
                X = df[:, 1:].astype(np.float32)
                all_data = TabularDataset(X, y)
                with open(f"{path}.pkl", "wb") as file:
                    pickle.dump(all_data, file)
                print(f"Save data to {path}.pkl")
            else:
                raise NotImplementedError(
                    f"{dataset_name} is not installed correctly in ../data/purchase"
                )
        elif dataset_name == "texas100":
            if os.path.exists("../data/texas/texas/100/feats"):
                X = (
                    pd.read_csv(
                        "../data/texas/texas/100/feats", header=None, encoding="utf-8"
                    )
                    .to_numpy()
                    .astype(np.float32)
                )
                y = (
                    pd.read_csv(
                        "../data/texas/texas/100/labels", header=None, encoding="utf-8"
                    )
                    .to_numpy()
                    .reshape(-1)
                    - 1
                )

                all_data = TabularDataset(X, y)
                with open(f"{path}.pkl", "wb") as file:
                    pickle.dump(all_data, file)
                print(f"Save data to {path}.pkl")
            else:
                raise NotImplementedError(
                    f"{dataset_name} is not installed correctly in ../data/texas"
                )
        else:
            raise NotImplementedError(f"{dataset_name} is not implemented")

    print(f"the whole dataset size: {len(all_data)}")
    return all_data


def get_dataset_subset(
    dataset: torchvision.datasets, index: List(int), model_name="CNN", device="cuda"
):
    """Get a subset of the dataset.

    Args:
        dataset (torchvision.datasets): Whole dataset.
        index (list): List of index.
        model_name (str): name of the model.
    """
    assert max(index) < len(dataset) and min(index) >= 0, "Index out of range"
    if model_name != "speedyresnet":  ## check if we need to change
        data_loader = get_dataloader(
            torch.utils.data.Subset(dataset, index),
            batch_size=len(index),
            shuffle=False,
        )
        for data, targets in data_loader:
            return data, targets
    else:
        data = get_cifar10_data(dataset, index[:1], index, device=device)
        input_list = []
        targets_list = []

        MAX_BATCH_SIZE = 5000  # to avoid OOM
        size = len(index)
        list_divisors = list(
            set(
                factor
                for i in range(1, int(math.sqrt(size)) + 1)
                if size % i == 0
                for factor in (i, size // i)
                if factor < MAX_BATCH_SIZE
            )
        )
        batch_size = max(list_divisors)

        for inputs, targets in get_batches(
            data, key="eval", batchsize=batch_size, shuffle=False, device=device
        ):
            input_list.append(inputs)
            targets_list.append(targets)
        inputs = torch.cat(input_list, dim=0)
        targets = torch.cat(targets_list, dim=0).max(dim=1)[1]
        return inputs, targets


def get_dataloader(
    dataset: torchvision.datasets,
    batch_size: int,
    loader_type="torch",
    shuffle: bool = True,
):
    if loader_type == "torch":
        repeated_data = InfiniteRepeatDataset(dataset)
        return torch.utils.data.DataLoader(
            repeated_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=16,
        )
