"""This module defines functions for model handling, including model definition, loading, and training."""

import copy
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, Subset

from dataset.utils import get_dataloader
from models.models import CNN, MLP, AlexNet, WideResNet
from trainers.default_trainer import inference, train
from utils.util import timer

INPUT_OUTPUT_SHAPE = {
    "cifar10": [3072, 10],
    "cifar100": [3072, 100]
}


def __get_model(model_type: str, dataset_name: str):
    num_classes = INPUT_OUTPUT_SHAPE[dataset_name][1]
    in_shape = INPUT_OUTPUT_SHAPE[dataset_name][0]
    if model_type == "CNN":
        return CNN(num_classes=num_classes)
    elif model_type == "alexnet":
        return AlexNet(num_classes=num_classes)
    elif model_type == "wrn28-1":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=28, width=1)
    elif model_type == "wrn28-2":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=28, width=2)
    elif model_type == "wrn28-10":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=28, width=10)
    elif model_type == "mlp":  # for purchase dataset
        return MLP(in_shape=in_shape, num_classes=num_classes)
    elif model_type == "vgg16":
        return torchvision.models.vgg16(pretrained=False)
    else:
        raise NotImplementedError(f"{model_type} is not implemented")

def __load_existing_model(
    model_metadata: dict[Any, Any], config: dict[str, Any]
):
    """Load an existing model from disk based on the provided metadata.

    Args:
        model_metadata (dict): Metadata dictionary containing information about the model.
        dataset (Dataset[Any]): Dataset object used to instantiate the model.
        device (str): The device on which to load the model, such as 'cpu' or 'cuda'.
        config (dict): Configuration dictionary containing information about the model.
    Returns:
        model (torch.nn.Module): Loaded model object with weights restored from disk.
    """
    model_name = model_metadata["model_name"]
    dataset_name = model_metadata["dataset"]

    model = __get_model(model_name, dataset_name)

    model_checkpoint_extension = os.path.splitext(model_metadata["model_path"])[1]
    if model_checkpoint_extension == ".pkl":
        with open(model_metadata["model_path"], "rb") as file:
            model_weight = pickle.load(file)
        model.load_state_dict(model_weight)
    elif model_checkpoint_extension == ".pt" or model_checkpoint_extension == ".pth":
        model.load_state_dict(torch.load(model_metadata["model_path"])) #type:ignore
    else:
        raise ValueError(f"Model path is invalid.")
    return model

def __load_models(log_dir: str, dataset: Dataset[Any], num_models: int, configs: dict[str, Any], logger: logging.Logger):
    """
    Load trained models from disk if available.

    Args:
        log_dir (str): Path to the directory containing model logs and metadata.
        dataset (Dataset[Any]): Dataset object used for model training.
        num_models (int): Number of models to be loaded from disk.
        configs (dict): Dictionary of configuration settings, including device information.
        logger (logging.Logger): Logger object for logging the model loading process.

    Returns:
        model_list (list of nn.Module): List of loaded model objects.
        all_memberships (np.array): Membership matrix for all loaded models, indicating training set membership.
    """
    experiment_dir = f"{log_dir}/models"
    if os.path.exists(f"{experiment_dir}/models_metadata.json"):
        with open(f"{experiment_dir}/models_metadata.json", "r") as f:
            model_metadata_dict = json.load(f)
        all_memberships = np.load(f"{experiment_dir}/memberships.npy")
        if len(model_metadata_dict) < num_models:
            return None, None
    else:
        return None, None

    model_list: list[nn.Module] = []
    for model_idx in range(len(model_metadata_dict)):
        logger.info(f"Loading model {model_idx}")
        model_obj = __load_existing_model(
            model_metadata_dict[str(model_idx)],
            configs
        )
        model_list.append(model_obj)
        if len(model_list) == num_models:
            break
    return model_list, all_memberships

def __train_models(log_dir: str, dataset: Dataset[Any], data_split_info: list[Any], all_memberships: npt.NDArray[Any], configs: dict[str, Any], logger: logging.Logger):
    """
    Train models based on the dataset split information.

    Args:
        log_dir (str): Path to the directory where models and logs will be saved.
        dataset (Dataset[Any]): Dataset object used for training the models.
        data_split_info (list): List of dictionaries containing training and test split information for each model.
        all_memberships (np.array): Membership matrix indicating which samples were used in training each model.
        configs (dict): Configuration dictionary containing training settings.
        logger (logging.Logger): Logger object for logging the training process.

    Returns:
        model_list (list of nn.Module): List of trained model objects.
    """
    experiment_dir = f"{log_dir}/models"
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Training {len(data_split_info)} models")

    np.save(f"{experiment_dir}/memberships.npy", all_memberships)

    model_metadata_dict: dict[int, dict[str, Any]] = {}
    model_list: list[nn.Module] = []

    # for split, split_info in enumerate(data_split_info):
    for split in range(len(data_split_info)):
        split_info = data_split_info[split]

        with timer(logger, f"Training model {split}"):
            logger.info(50 * "-")
            logger.info(
                f"Training model {split}: Train size {len(split_info['train'])}, Test size {len(split_info['test'])}"
            )

            model_name, dataset_name, batch_size, device = (
                configs["train"]["model_name"],
                configs["data"]["dataset"],
                configs["train"]["batch_size"],
                configs["train"]["device"],
            )

            train_loader = get_dataloader(
                Subset(dataset, split_info["train"]),
                batch_size=batch_size,
                shuffle=True,
            )
            test_loader = get_dataloader(
                Subset(dataset, split_info["test"]),
                batch_size=batch_size,
            )
            model = train(
                __get_model(model_name, dataset_name),
                train_loader,
                configs["train"],
                test_loader,
                f"{experiment_dir}/model_{split}"
            )
            test_loss, test_acc = inference(model, test_loader, device)
            train_loss, train_acc = inference(model, train_loader, device)
            logger.info(f"Train accuracy {train_acc}, Train Loss {train_loss}")
            logger.info(f"Test accuracy {test_acc}, Test Loss {test_loss}")

            model_list.append(copy.deepcopy(model))




        with open(f"{experiment_dir}/model_{split}.pkl", "wb") as f:
            pickle.dump(model.state_dict(), f)

        model_metadata_dict[split] = {
            "num_train": len(split_info["train"]),
            "optimizer": configs["train"]["optimizer"],
            "batch_size": batch_size,
            "epochs": configs["train"]["epochs"],
            "model_name": model_name,
            "learning_rate": configs["train"]["learning_rate"],
            "weight_decay": configs["train"]["weight_decay"],
            "model_path": f"{log_dir}/model_{split}.pkl",
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "dataset": dataset_name,
        }

    with open(f"{experiment_dir}/models_metadata.json", "w") as f:
        json.dump(model_metadata_dict, f, indent=4)


    return model_list

def __split_dataset_for_training(dataset_size: int, num_model_pairs: int):
    """
    Split dataset into training and test partitions for model pairs.

    Args:
        dataset_size (int): Total number of samples in the dataset.
        num_model_pairs (int): Number of model pairs to be trained, with each pair trained on different halves of the dataset.

    Returns:
        data_split (list): List of dictionaries containing training and test split indices for each model.
        master_keep (np.array): D boolean array indicating the membership of samples in each model's training set.
    """
    data_splits: list[dict[str, Any]] = []
    indices = np.arange(dataset_size)
    split_index = len(indices) // 2
    master_keep = np.full((2 * num_model_pairs, dataset_size), True, dtype=bool)

    for i in range(num_model_pairs):
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


def get_models(log_dir: str,
               dataset: Dataset[Any],
               num_model_pairs: int,
               configs: dict[str, Any],
               logger: logging.Logger):

    models_list, memberships = __load_models(
        log_dir, dataset, num_model_pairs * 2, configs, logger
    )
    if models_list is None:
        # Split dataset for training two models per pair
        data_splits, memberships = __split_dataset_for_training(
            len(dataset), num_model_pairs #type:ignore
        )
        models_list = __train_models(
            log_dir, dataset, data_splits, memberships, configs, logger
        )
    return models_list, memberships
