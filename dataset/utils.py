"""This file contains functions for loading the dataset"""

import logging
import os
import pickle
from typing import Any, List, Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset


class InfinitelyIndexableDataset(Dataset[Any]):
    """
    A PyTorch Dataset that is able to index the given dataset infinitely.
    This is a helper class to allow easier and more efficient computation later when repeatedly indexing the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be indexed repeatedly.
    """

    def __init__(self, dataset: Dataset[Any]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset) #type:ignore

    def __getitem__(self, idx: int) -> Any:
        # If the index is out of range, wrap it around
        return self.dataset[idx % len(self.dataset)]#type:ignore


def get_dataset(dataset_name: str, data_dir: str, logger: logging.Logger, **kwargs: Any) -> Any:
    """
    Function to load the dataset from the pickle file or download it from the internet.

    Args:
        dataset_name (str): Dataset name.
        data_dir (str): Indicate the log directory for loading the dataset.
        logger (logging.Logger): Logger object for the current run.

    Raises:
        NotImplementedError: If the dataset is not implemented.

    Returns:
        Any: Loaded dataset.
    """
    path = f"{data_dir}/{dataset_name}"
    if os.path.exists(f"{path}.pkl"):
        assert os.path.exists(f"{path}_population.pkl")
        with open(f"{path}.pkl", "rb") as file:
            all_data = pickle.load(file)
        logger.info(f"Data loaded from {path}.pkl")
        with open(f"{path}_population.pkl", "rb") as file:
            test_data = pickle.load(file)
        logger.info(f"Population data loaded from {path}_population.pkl")
    else:
        if dataset_name not in ["cifar10", "cifar100"]:
            raise NotImplementedError(f"{dataset_name} is not implemented")
        loader = torchvision.datasets.CIFAR10 if dataset_name == "cifar10" else torchvision.datasets.CIFAR100

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        all_data = loader(root=path, train=True, download=True, transform=transform)
        test_data = loader(root=path, train=False, download=True, transform=transform)

        with open(f"{path}.pkl", "wb") as file:
            pickle.dump(all_data, file)
        logger.info(f"Save data to {path}.pkl")
        with open(f"{path}_population.pkl", "wb") as file:
            pickle.dump(test_data, file)
        logger.info(f"Save population data to {path}_population.pkl")

    logger.info(f"The whole dataset size: {len(all_data)}")
    return all_data, test_data


def load_dataset_subsets(
    dataset: Dataset[Any],
    index: List[int],
    model_type: str,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function to divide dataset into subsets and load them into device (GPU).

    Args:
        dataset (torchvision.datasets): The whole dataset.
        index (List[int]): List of sample indices.
        model_type (str): Type of the model.
        batch_size (int): Batch size for getting signals.
        device (str): Device used for loading models.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Loaded samples and their labels.
    """
    assert max(index) < len(dataset) and min(index) >= 0, "Index out of range" #type:ignore
    input_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    data_loader = get_dataloader(
            Subset(dataset, index),
            batch_size=batch_size,
            shuffle=False,
            num_workers=(0 if batch_size == 1 else 4),
        )

    for inputs, targets in data_loader:
        input_list.append(inputs)
        targets_list.append(targets)
    inputs = torch.cat(input_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    return inputs, targets


def get_dataloader(
    dataset: Dataset[Any],
    batch_size: int,
    loader_type: str = "torch",
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader[Any]:
    """
    Function to get DataLoader.

    Args:
        dataset (torchvision.datasets): The whole dataset.
        batch_size (int): Batch size for getting signals.
        loader_type (str): Loader type.
        shuffle (bool): Whether to shuffle dataset or not.

    Returns:
        DataLoader: DataLoader object.
    """
    if loader_type not in ["torch"]:
        raise NotImplementedError(f"{loader_type} is not supported")

    repeated_data = InfinitelyIndexableDataset(dataset)
    return DataLoader(
        repeated_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=16 if num_workers > 0 else None,
    )
