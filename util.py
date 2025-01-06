"""This file contains information about the utility functions."""

import logging
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from dataset import get_dataset


def check_configs(configs: Dict[str, Any]) -> None:
    """
    Function to check if the configs are valid.

    Args:
        configs (Dict[str, Any]): Configs provided by the user.
    """
    privacy_game = configs["audit"]["privacy_game"]
    supported_games = ["privacy_loss_model"]
    if privacy_game not in supported_games:
        raise NotImplementedError(
            f"{privacy_game} is not supported. Please choose from {supported_games}"
        )
    mia_algorithm = configs["audit"]["algorithm"]
    num_models = configs["audit"]["num_ref_models"]
    if mia_algorithm == "RMIA" and (num_models is None or num_models < 1):
        raise ValueError("The game should have at least 2 models")


def setup_log(report_dir: str, name: str, save_file: bool) -> logging.Logger:
    """
    Function to generate the logger for the current run.

    Args:
        report_dir (str): Folder name of the audit.
        name (str): Logging file name.
        save_file (bool): Flag about whether to save to file.

    Returns:
        logging.Logger: Logger object for the current run.
    """
    my_logger = logging.getLogger(name)
    my_logger.setLevel(logging.INFO)

    log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    my_logger.addHandler(console_handler)

    if save_file:
        filename = f"{report_dir}/log_{name}.log"

        if not Path(filename).is_file():
            open(filename, "w+").close()

        log_handler = logging.FileHandler(filename, mode="w")
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(log_format)
        my_logger.addHandler(log_handler)

    return my_logger


def initialize_seeds(seed: int) -> None:
    """
    Initialize seeds for reproducibility.

    Args:
        seed (int): Seed value for random number generators.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_directories(paths: Dict[str, str]) -> None:
    """
    Create necessary directories if they do not exist.

    Args:
        paths (Dict[str, str]): Dictionary of directory paths to create.
    """
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)


def load_dataset(configs: Dict[str, Any], data_dir: str, logger: logging.Logger) -> Any:
    """
    Load the dataset based on the configuration.

    Args:
        configs (Dict[str, Any]): Configuration dictionary.
        data_dir (str): Directory where the data is stored.
        logger (logging.Logger): Logger object for logging information.

    Returns:
        Any: Loaded dataset.
    """
    if not configs['data'].get("tokenize", False):
        return get_dataset(configs['data']['dataset'], data_dir, logger)
    return get_dataset(
        configs['data']['dataset'],
        data_dir,
        logger,
        tokenizer=configs['data']['tokenizer'],
    )


def load_canary_dataset(configs: Dict[str, Any], data_dir: str, logger: logging.Logger) -> Any:
    """
    Load the dataset based on the configuration.

    Args:
        configs (Dict[str, Any]): Configuration dictionary.
        data_dir (str): Directory where the data is stored.
        logger (logging.Logger): Logger object for logging information.

    Returns:
        Any: Loaded dataset.
    """
    if not configs['dp_audit'].get("tokenize", False):
        return get_dataset(configs['dp_audit']['canary_dataset'], data_dir, logger)
    return get_dataset(
        configs['dp_audit']['canary_dataset'],
        data_dir,
        logger,
        tokenizer=configs['dp_audit']['tokenizer'],
    )

def split_dataset_for_training_poisson(dataset_size, num_model_pairs):
    """
    Split dataset into training and test partitions for model pairs.
    Args:
        dataset_size (int): Total number of samples in the dataset.
        num_model_pairs (int): Number of model pairs to be trained, with each pair trained on different halves of the dataset.
    Returns:
        data_split (list): List of dictionaries containing training and test split indices for each model.
        master_keep (np.array): D boolean array indicating the membership of samples in each model's training set.
    """
    data_splits = []
    indices = np.arange(dataset_size)
    split_index = len(indices) // 2
    master_keep = np.full((2 * num_model_pairs, dataset_size), True, dtype=bool)
    for i in range(num_model_pairs):
        # only the target model is sampled via poisson sampling
        if i == 0:
            keep = np.random.choice(2, dataset_size, p=[0.5, 0.5]).astype(bool)
            master_keep[i * 2, :] = keep
            master_keep[i * 2 + 1, :] = keep
            train_indices = np.where(keep)[0]
            test_indices = np.where(~keep)[0]
        else:
            # rest of the models are sampled to ensure there are equal number of in/out models
            np.random.shuffle(indices)
            master_keep[i * 2, indices[split_index:]] = False
            master_keep[i * 2 + 1, indices[:split_index]] = False
            keep = master_keep[i * 2, :]
            train_indices = np.where(keep)[0]
            test_indices = np.where(~keep)[0]
        data_splits.append(
            {
                "train": train_indices,
                "test": test_indices,
            }
        )
        data_splits.append(
            {
                "train": test_indices,
                "test": train_indices,
            }
        )
    return data_splits, master_keep