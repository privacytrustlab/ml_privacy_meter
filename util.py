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
