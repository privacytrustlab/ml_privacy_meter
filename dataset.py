"""This file contains functions for loading the dataset"""
import os
import pickle
from ast import List

import numpy as np
import torch
import torchvision
from torchvision import transforms


def get_dataset(dataset_name: str, data_dir: str) -> torchvision.datasets:
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
            transform = transforms.Compose([transforms.ToTensor()])
            all_data = torchvision.datasets.CIFAR10(
                root=path, train=True, download=True, transform=transform
            )
            test_data = torchvision.datasets.CIFAR10(
                root=path, train=False, download=True, transform=transform
            )
            all_features = np.concatenate(
                [all_data.data, test_data.data], axis=0)
            all_targets = np.concatenate(
                [all_data.targets, test_data.targets], axis=0)

            all_data.data = all_features
            all_data.targets = all_targets
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            print(f"Save data to {path}.pkl")
        else:
            raise NotImplementedError(f"{dataset_name} is not implemented")

    print(f"the whole dataset size: {len(all_data)}")
    return all_data


def get_dataset_subset(dataset: torchvision.datasets, index: List(int)):
    """Get a subset of the dataset.

    Args:
        dataset (torchvision.datasets): Whole dataset.
        index (list): List of index.
    """
    assert max(index) < len(dataset) and min(index) >= 0, "Index out of range"
    data = (
        torch.from_numpy(dataset.data[index]).float().permute(0, 3, 1, 2) / 255
    )  # channel first
    targets = list(np.array(dataset.targets)[index])
    targets = torch.tensor(targets, dtype=torch.long)
    return data, targets
