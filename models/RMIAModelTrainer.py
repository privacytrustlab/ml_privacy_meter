import copy
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torchvision

from trainers.train_transformers import *
from peft import get_peft_model

from models.utils import get_model


class RMIAModelTrainer:
    """
    Handles the entire process of training models for RMIA, including dataset splitting,
    model training, and saving the artifacts.
    """

    def __init__(
        self,
        log_dir: str,
        dataset: torchvision.datasets,
        num_model_pairs: int,
        configs: dict,
        logger,
    ):
        """
        Initializes the trainer.

        Args:
            log_dir (str): Base directory to save logs, models, and metadata.
            dataset (torchvision.datasets): The dataset to use for training.
            num_model_pairs (int): The number of model pairs (k) to train.
            configs (dict): Configuration dictionary.
            logger: Logger object.
        """
        self.log_dir = f"{log_dir}/models"
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.num_model_pairs = num_model_pairs
        self.configs = configs
        self.logger = logger
        self.indices = np.arange(self.dataset_size)

    def _split_dataset(self):
        """
        Splits the dataset for training 2*k models for RMIA.

        Returns:
            A tuple containing:
            - data_splits (list): A list of dictionaries for each model with 'train' and 'test' indices.
            - memberships (np.ndarray): A boolean array indicating training set membership for each model.
        """
        data_splits = []
        num_models = 2 * self.num_model_pairs
        memberships = np.zeros((num_models, self.dataset_size), dtype=bool)
        split_index = self.dataset_size // 2

        for i in range(self.num_model_pairs):
            np.random.shuffle(self.indices)
            model1_idx = 2 * i
            model2_idx = 2 * i + 1

            train1_indices = self.indices[:split_index]
            test1_indices = self.indices[split_index:]
            memberships[model1_idx, train1_indices] = True
            data_splits.append({"train": train1_indices, "test": test1_indices})

            train2_indices = test1_indices
            test2_indices = train1_indices
            memberships[model2_idx, train2_indices] = True
            data_splits.append({"train": train2_indices, "test": test2_indices})

        return data_splits, memberships

    def train_and_save_models(self):
        """
        Orchestrates the model training process. It splits the data, trains the models,
        saves them to disk, and returns the trained model objects and membership information.

        Returns:
            A tuple containing:
            - model_list (list): A list of trained model objects.
            - all_memberships (np.ndarray): The membership matrix.
        """
        data_split_info, all_memberships = self._split_dataset()

        np.save(f"{self.log_dir}/memberships.npy", all_memberships)

        self.logger.info(f"Training {len(data_split_info)} models")

        model_metadata_dict = {}
        model_list = []

        for split, split_info in enumerate(data_split_info):
            baseline_time = time.time()
            self.logger.info(50 * "-")
            self.logger.info(
                f"Training model {split}: Train size {len(split_info['train'])}, Test size {max(len(split_info['test']), 1000)}"
            )

            model_name = self.configs["train"]["model_name"]
            dataset_name = self.configs["data"]["dataset"]
            batch_size = self.configs["train"]["batch_size"]

            train_configs = self.configs["train"]

            if model_name == "gpt2":
                hf_dataset = self.dataset.hf_dataset
                model, train_loss, test_loss = train_transformer(
                    hf_dataset.select(split_info["train"]),
                    get_model(model_name, dataset_name, self.configs),
                    self.configs,
                    hf_dataset.select(split_info["test"][:1000]),
                    model_idx=split
                )
                train_acc, test_acc = None, None
            else:
                raise ValueError(
                    f"The {model_name} is not supported for the {dataset_name}"
                )

            model_list.append(copy.deepcopy(model))
            self.logger.info(
                f"Training model {split} took {time.time() - baseline_time} seconds"
            )

            model_idx = split
            with open(f"{self.log_dir}/model_{model_idx}.pkl", "wb") as f:
                pickle.dump(model.state_dict(), f)

            model_metadata_dict[model_idx] = {
                "num_train": len(split_info["train"]),
                "optimizer": self.configs["train"]["optimizer"],
                "batch_size": batch_size,
                "epochs": self.configs["train"]["epochs"],
                "model_name": model_name,
                "learning_rate": self.configs["train"]["learning_rate"],
                "weight_decay": self.configs["train"]["weight_decay"],
                "model_path": f"{self.log_dir}/model_{model_idx}.pkl",
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "dataset": dataset_name,
            }

        with open(f"{self.log_dir}/models_metadata.json", "w") as f:
            json.dump(model_metadata_dict, f, indent=4)

        return model_list, all_memberships