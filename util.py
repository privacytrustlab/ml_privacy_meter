"""This file contains information about the utility functions."""
from ast import List
import numpy as np
import torch


def check_configs(configs: dict):
    """Check if the configs are valid.

    Args:
        configs (dict): Configs provided by the user.
    """
    privacy_game = configs["audit"]["privacy_game"]
    supported_games = [
        "avg_privacy_loss_training_algo",
        "privacy_loss_model",
        "privacy_loss_sample",
    ]
    if privacy_game not in supported_games:
        raise NotImplementedError(
            f"{privacy_game} is not supported."
            + f"Please choose from {supported_games}"
        )
    if privacy_game in ["privacy_loss_model", "avg_privacy_loss_training_algo"]:
        num_target_model = configs["train"]["num_target_model"]
        if privacy_game == "privacy_loss_model" and num_target_model != 1:
            raise ValueError(
                "privacy_loss_model game only supports one target model")
        if privacy_game == "avg_privacy_loss_training_algo":
            if num_target_model <= 1:
                raise ValueError(
                    "avg_privacy_loss_training_algo game"
                    + "needs more than one target model"
                )


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
    optimizer = configs.get("optimizer", "SGD")
    learning_rate = configs.get("learning_rate", 0.001)
    weight_decay = configs.get("weight_decay", 0)

    print(f"Load the optimizer {optimizer}: ", end=" ")
    print(f"learning rate {learning_rate}", end=" ")
    print(f"weight decay {weight_decay} ")

    if optimizer == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer == "Adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    else:
        raise NotImplementedError(
            f"Optimizer {optimizer} has not been implemented. Please choose from SGD or Adam"
        )


def get_split(
    all_index: List(int), used_index: List(int), size: int, split_method: str
) -> np.ndarray:
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
    if split_method in "no_overlapping":
        selected_index = np.array(
            [i for i in all_index if i not in used_index])
        if size <= len(selected_index):
            selected_index = np.random.choice(
                selected_index, size, replace=False)
        else:
            raise ValueError("Not enough remaining data points.")
    elif split_method == "uniform":
        if size <= len(all_index):
            selected_index = np.random.choice(all_index, size, replace=False)
        else:
            raise ValueError("Not enough remaining data points.")
    else:
        raise NotImplementedError(
            f"{split_method} is not implemented. The only supported methods are uniform and no_overlapping."
        )

    return selected_index


def load_models_by_conditions(
    model_metadata_dict: dict,
    conditions: dict,
    num_models: int,
    exclude_idx: List(int) = [],
) -> List(int):
    """Load existing models metadata index based on the conditions

    Args:
        model_metadata_dict (dict): Model metadata dict.
        conditions (dict): Conditions to match.
        num_models (int): Number of models needed.
        exclude_idx (List, optional): Metadata index list that are excluded.

    Returns:
        List: List of metadata index which match the conditions.
    """
    assert isinstance(conditions, dict)
    if len(conditions) == 0:
        return []
    matched_idx = []
    for meta_idx, meta_data in model_metadata_dict["model_metadata"].items():
        if meta_idx in exclude_idx:
            continue
        if len(matched_idx) >= num_models:
            return matched_idx
        for key, item in conditions.items():
            if key in meta_data:
                if meta_data[key] != item:
                    is_matched = False
                    break
                else:
                    is_matched = True
            else:
                is_matched = False
                break
        if is_matched:
            matched_idx.append(meta_idx)
    return matched_idx


def load_models_by_model_idx(
    model_metadata_dict: dict, model_idx_list: List(int)
) -> List(int):
    """Load existing models metadata index based on the model index.

    Args:
        model_metadata_dict (dict): Model metadata dict.
        model_idx_list (List[int]): Model index list.

    Returns:
        List[int]: List of metadata index which match the model index.
    """
    assert all(isinstance(index, int) for index in model_idx_list)
    assert isinstance(model_metadata_dict, dict)
    assert set(model_idx_list).issubset(
        set(model_metadata_dict["model_metadata"].keys())
    )
    return model_idx_list


def load_models_with_data_idx_list(
    model_metadata_dict: dict, data_idx_list: List(int)
) -> List(int):
    """Load existing metadata index of models which are trained on the data index list.

    Args:
        model_metadata_dict (dict): Model metadata dict.
        data_idx_list (List(int)): Data index list.

    Returns:
        List(int): List of metadata index which match the conditions.
    """
    assert all(isinstance(index, int) for index in data_idx_list)
    assert isinstance(model_metadata_dict, dict)
    if not data_idx_list:
        raise ValueError("data_idx_list is empty.")
    matched_idx = [
        meta_idx
        for meta_idx, meta_data in model_metadata_dict["model_metadata"].items()
        if (set(data_idx_list).issubset(set(meta_data["train_split"])))
    ]
    return matched_idx


def load_models_without_data_idx_list(
    model_metadata_dict: dict, data_idx_list: List(int)
) -> List(int):
    """Load existing metadata index of models which are not trained on the data index list.

    Args:
        model_metadata_dict (dict): Model metadata dict.
        data_idx_list (List(int)): Data index list.

    Returns:
        List(int): List of metadata index which match the conditions.
    """
    assert all(isinstance(index, int) for index in data_idx_list)
    assert isinstance(model_metadata_dict, dict)
    if not data_idx_list:
        raise ValueError("data_idx_list is empty.")
    matched_idx = [
        meta_idx
        for meta_idx, meta_data in model_metadata_dict["model_metadata"].items()
        if set(data_idx_list).isdisjoint(meta_data["train_split"])
    ]
    return matched_idx


def load_models_with_data_idx_and_cond_list(
    model_metadata_dict: dict,
    conditions: dict,
    data_idx_list: List(int),
    exclude_idx: List(int) = [],
) -> List(int):
    # TODO
    return []


def load_models_without_data_idx_and_cond_list(
    model_metadata_dict: dict,
    conditions: dict,
    data_idx_list: List(int),
    exclude_idx: List(int) = [],
) -> List(int):
    # TODO
    return []
