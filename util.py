from ast import List
import numpy as np
import torch.optim as optim
import torch


def get_optimizer(model: torch.nn.Module, configs: dict) -> torch.optim.Optimizer:
    """Get the optimizer for the given model

    Args:
        model (torch.nn.Module): The model we want to optimize
        configs (dict): Configurations for the optimizer

    Raises:
        NotImplementedError: Check if the optimizer is implemented.

    Returns:
        optim: Optimizer for the given model
    """
    optimizer = configs.get('optimizer', 'SGD')
    lr = configs.get('learning_rate', 0.001)
    wd = configs.get('weight_decay', 0)

    if optimizer in ['SGD', 'Adam']:
        if optimizer == 'SGD':
            momentum = configs.get('momentum', 0)
            print(
                f"Load the optimizer {optimizer} with learning rate {lr}, weight decay {wd} and momentum {momentum}")
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        else:
            print(
                f"Load the optimizer {optimizer} with learning rate {lr}, weight decay {wd}")
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise NotImplementedError(
            f"Optimizer {optimizer} has not been implemented. Please choose from SGD or Adam")


def get_split(all_index: List(int), used_index:  List(int), size: int, split_method: str) -> np.ndarray:
    """Select points based on the splitting methods

    Args:
        all_index (list): All the possible dataset index list
        used_index (list): Index list of used points
        size (int): Size of the points needs to be selected
        split_method (str): Splitting (selection) method

    Raises:
        NotImplementedError: If the splitting the methods isn't implemented
        ValueError: If there aren't enough points to select
    Returns:
        np.ndarray: List of index
    """
    if split_method in 'no_overlapping':
        selected_index = np.array(
            [i for i in all_index if i not in used_index])
        if size <= len(selected_index):
            selected_index = np.random.choice(
                selected_index, size, replace=False)
        else:
            raise ValueError("Not enough remaining data points.")
    elif split_method == 'uniform':
        if size <= len(all_index):
            selected_index = np.random.choice(all_index, size, replace=False)
        else:
            raise ValueError("Not enough remaining data points.")
    else:
        raise NotImplementedError(
            f"{split_method} is not implemented. The only supported methods are uniform and no_overlapping.")

    return selected_index
