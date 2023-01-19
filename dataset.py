from ast import List
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import copy
import pickle
import os


def get_dataset(dataset_name: str, data_dir: str) -> torchvision.datasets:
    """Load the dataset 

    Args:
        dataset_name (str): Dataset name
        data_dir (str): Indicate the log directory for loading the dataset

    Raises:
        NotImplementedError: Check if the dataset has been implemented.

    Returns:
        torchvision.datasets: Whole dataset.
    """
    path = f'{data_dir}/{dataset_name}.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            all_data = pickle.load(f)
        print(f"Load data from {path}")

    else:
        if dataset_name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor()]
            )
            all_data = torchvision.datasets.CIFAR10(
                root=f'{data_dir}/{dataset_name}', train=True, download=True, transform=transform)
            test_data = torchvision.datasets.CIFAR10(
                root=f'{data_dir}/{dataset_name}', train=False, download=True, transform=transform)
            X = np.concatenate([all_data.data, test_data.data], axis=0)
            Y = np.concatenate([all_data.targets, test_data.targets], axis=0)

            all_data.data = X
            all_data.targets = Y
            with open(f'{path}', 'wb') as f:
                pickle.dump(all_data, f)
            print(f"Save data to {path}")
        else:
            raise NotImplementedError(f"{dataset_name} is not implemented")

    print(f"the whole dataset size: {len(all_data)}")
    return all_data


def get_cifar10_subset(dataset: torchvision.datasets.cifar.CIFAR10, index: List(int), is_tensor: bool = False):
    """Get a subset of the cifar10 dataset

    Args:
        dataset (torchvision.datasets.cifar.CIFAR10): Whole dataset
        index (list): List of index
        is_tensor (bool, optional): Whether to return tensors of the data. Defaults to False.
    """
    assert type(dataset) == torchvision.datasets.cifar.CIFAR10, ValueError(
        "Input the correct dataset")
    assert max(index) < 60000 and min(
        index) >= 0, ValueError("Input the correct index")

    data = dataset.data[index]
    targets = list(np.array(dataset.targets)[index].astype(np.int32))
    if is_tensor:
        data = torch.from_numpy(data).float().permute(
            0, 3, 1, 2)/255  # channel first
        targets = torch.tensor(targets,dtype=torch.long)

    return data, targets
