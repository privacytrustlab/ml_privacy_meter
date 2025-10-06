"""This file contains information about the utility functions."""

import argparse
import functools
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch
import yaml


class timer:
    def __init__(self, logger: logging.Logger | None, text: str = "", end_messages: list[str] = []):
        self.logger = logger
        self.text = text
        self.end_messages = end_messages

    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args: Any):
        self.end = time.time()
        self.interval = self.end - self.start
        printer = self.logger.info if self.logger is not None else print

        for msg in self.end_messages:
            printer(msg)
        printer(f"{self.text} took {self.interval:.5f} seconds")



    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                result = func(*args, **kwargs)
            return result
        return wrapper

    def set_logger(self, logger: logging.Logger) -> None:
        self.logger = logger

    def add_msg(self, msg: str) -> None:
        self.end_messages.append(msg)

def setup():
    """
        Setup function to initialize configurations, directories, and logger.
    """

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run RMIA.")
    parser.add_argument(
        "--cf",
        type=str,
        default="configs/cifar10.yaml",
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()

    # Load configuration file
    with open(args.cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    # Validate configurations
    __check_configs(configs)

    # Initialize seeds for reproducibility
    __initialize_seeds(configs["run"]["random_seed"])

    # Create necessary directories
    log_dir = configs["run"]["log_dir"]
    directories: dict[str, str] = {
        "log_dir": log_dir,
        "report_dir": f"{log_dir}/report",
        "signal_dir": f"{log_dir}/signals",
        "data_dir": configs["data"]["data_dir"],
    }
    __create_directories(directories)

    # Set up logger
    logger = __setup_log(
        directories["report_dir"], "time_analysis", configs["run"]["time_log"]
    )

    return configs, directories, log_dir, logger


def __check_configs(configs: Dict[str, Any]) -> None:
    """
    Function to check if the configs are valid.

    Args:
        configs (Dict[str, Any]): Configs provided by the user.
    """
    num_models = configs["audit"]["num_ref_models"]
    if (num_models is None or num_models < 1):
        raise ValueError("The game should have at least 2 models")

def __initialize_seeds(seed: int) -> None:
    """
    Initialize seeds for reproducibility.

    Args:
        seed (int): Seed value for random number generators.
    """
    torch.manual_seed(seed) #type:ignore
    np.random.seed(seed)
    random.seed(seed)

def __create_directories(paths: Dict[str, str]) -> None:
    """
    Create necessary directories if they do not exist.

    Args:
        paths (Dict[str, str]): Dictionary of directory paths to create.
    """
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)

def __setup_log(report_dir: str, name: str, save_file: bool) -> logging.Logger:
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

