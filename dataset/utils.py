"""This file contains functions for loading the dataset"""

import math
import os
import pickle
import subprocess
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from dataset import TabularDataset, TextDataset, load_agnews
from trainers.fast_train import get_batches, load_cifar10_data


class InfinitelyIndexableDataset(Dataset):
    """
    A PyTorch Dataset that is able to index the given dataset infinitely.
    This is a helper class to allow easier and more efficient computation later when repeatedly indexing the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be indexed repeatedly.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        # If the index is out of range, wrap it around
        return self.dataset[idx % len(self.dataset)]


def get_dataset(dataset_name: str, data_dir: str, logger: Any, **kwargs: Any) -> Any:
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
        with open(f"{path}.pkl", "rb") as file:
            all_data = pickle.load(file)
        logger.info(f"Load data from {path}.pkl")
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
            logger.info(f"Save data to {path}.pkl")
        elif dataset_name == "cifar10_canary":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            all_data = torchvision.datasets.CIFAR10(
                root=path.replace('cifar10_canary', 'cifar10'), train=True, download=True, transform=transform
            )
            test_data = torchvision.datasets.CIFAR10(
                root=path.replace('cifar10_canary', 'cifar10'), train=False, download=True, transform=transform
            )
            all_features = np.concatenate([all_data.data, test_data.data], axis=0)
            all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
            all_data.data = all_features
            all_data.targets = all_targets
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
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
            logger.info(f"Save data to {path}.pkl")
        elif dataset_name == "purchase100":
            if not os.path.exists(f"{data_dir}/dataset_purchase"):
                logger.info(
                    f"{dataset_name} not found in {data_dir}/dataset_purchase. Downloading to /data..."
                )
                try:
                    # Download the dataset to /data
                    subprocess.run(
                        [
                            "wget",
                            "https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz",
                            "-P",
                            f"./{data_dir}",
                        ],
                        check=True,
                    )
                    # Extract the dataset to /data
                    subprocess.run(
                        [
                            "tar",
                            "-xf",
                            f"./{data_dir}/dataset_purchase.tgz",
                            "-C",
                            f"./{data_dir}",
                        ],
                        check=True,
                    )
                    logger.info(
                        "Dataset downloaded and extracted to /data successfully."
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error during download or extraction: {e}")
                    raise RuntimeError("Failed to download or extract the dataset.")

            df = pd.read_csv(
                f"{data_dir}/dataset_purchase", header=None, encoding="utf-8"
            ).to_numpy()
            y = df[:, 0] - 1
            X = df[:, 1:].astype(np.float32)
            all_data = TabularDataset(X, y)
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
        elif dataset_name == "texas100":
            if not os.path.exists(f"{data_dir}/dataset_texas/feats"):
                logger.info(
                    f"{dataset_name} not found in {data_dir}/dataset_purchase. Downloading to /data..."
                )
                try:
                    # Download the dataset to /data
                    subprocess.run(
                        [
                            "wget",
                            "https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz",
                            "-P",
                            f"./{data_dir}",
                        ],
                        check=True,
                    )
                    # Extract the dataset to /data
                    subprocess.run(
                        [
                            "tar",
                            "-xf",
                            f"./{data_dir}/dataset_texas.tgz",
                            "-C",
                            "./data",
                        ],
                        check=True,
                    )
                    logger.info(
                        "Dataset downloaded and extracted to /data successfully."
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error during download or extraction: {e}")
                    raise RuntimeError("Failed to download or extract the dataset.")

            X = (
                pd.read_csv(
                    f"{data_dir}/dataset_texas/feats", header=None, encoding="utf-8"
                )
                .to_numpy()
                .astype(np.float32)
            )
            y = (
                pd.read_csv(
                    f"{data_dir}/dataset_texas/labels",
                    header=None,
                    encoding="utf-8",
                )
                .to_numpy()
                .reshape(-1)
                - 1
            )
            all_data = TabularDataset(X, y)
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
        elif dataset_name == "agnews":
            tokenizer = kwargs.get("tokenizer")
            if tokenizer is None:
                agnews = load_agnews(tokenize=False)
            else:
                agnews = load_agnews(
                    tokenize=True,
                    tokenizer=AutoTokenizer.from_pretrained(
                        tokenizer, clean_up_tokenization_spaces=True
                    ),
                )
            all_data = TextDataset(agnews, target_column="labels", text_column="text")
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
        else:
            raise NotImplementedError(f"{dataset_name} is not implemented")

    logger.info(f"The whole dataset size: {len(all_data)}")
    return all_data


def load_dataset_subsets(
    dataset: torchvision.datasets,
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
    assert max(index) < len(dataset) and min(index) >= 0, "Index out of range"
    input_list = []
    targets_list = []
    if model_type != "speedyresnet":
        if batch_size == 1:
            # This happens with range dataset. Need to set num_workers to 0 to avoid CUDA error
            data_loader = get_dataloader(
                torch.utils.data.Subset(dataset, index),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            data_loader = get_dataloader(
                torch.utils.data.Subset(dataset, index),
                batch_size=batch_size,
                shuffle=False,
            )
        for inputs, targets in data_loader:
            input_list.append(inputs)
            targets_list.append(targets)
        inputs = torch.cat(input_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
    else:
        data = load_cifar10_data(dataset, index[:1], index, device=device)
        size = len(index)
        list_divisors = list(
            set(
                factor
                for i in range(1, int(math.sqrt(size)) + 1)
                if size % i == 0
                for factor in (i, size // i)
                if factor < batch_size
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
    loader_type: str = "torch",
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
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
    if loader_type == "torch":
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
    else:
        raise NotImplementedError(f"{loader_type} is not supported")
