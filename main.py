"""This file is the main entry point for running the priavcy auditing."""
import argparse
import logging
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from torch import nn

from core import (load_dataset_for_existing_models, load_existing_models,
                  load_existing_target_model, prepare_datasets,
                  prepare_datasets_for_sample_privacy_risk,
                  prepare_information_source, prepare_models,
                  prepare_priavcy_risk_report)
from dataset import get_dataset, get_dataset_subset
from privacy_meter.audit import Audit
from privacy_meter.model import PytorchModelTensor
from util import (check_configs, load_models_with_data_idx_list,
                  load_models_without_data_idx_list)


def setup_log(name: str) -> logging.Logger:
    """Generate the logger for the current run.
    Args:
        name (str): Logging file name.

    Returns:
        logging.Logger: Logger object for the current run.
    """
    my_logger = logging.getLogger(name)
    my_logger.setLevel(logging.INFO)
    log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    filename = f"log_{name}.log"
    log_handler = logging.FileHandler(filename, mode="w")
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(log_format)
    my_logger.addHandler(log_handler)
    return my_logger


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cf",
        type=str,
        default="config_models.yaml",
        help="Yaml file which contains the configurations",
    )

    # Load the parameters
    args = parser.parse_args()
    with open(args.cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    check_configs(configs)
    # Set the random seed, log_dir and inference_game
    torch.manual_seed(configs["run"]["random_seed"])
    np.random.seed(configs["run"]["random_seed"])

    log_dir = configs["run"]["log_dir"]
    inference_game_type = configs["audit"]["privacy_game"].upper()

    # Set up the logger
    logger = setup_log("time_analysis")

    # Create folders for saving the logs if they do not exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    report_dir = f"{log_dir}/{configs['audit']['report_log']}"
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Load or initialize models based on metadata
    if os.path.exists((f"{log_dir}/models_metadata.pkl")):
        with open(f"{log_dir}/models_metadata.pkl", "rb") as f:
            model_metadata_list = pickle.load(f)
    else:
        model_metadata_list = {"model_metadata": {}, "current_idx": 0}
    # Load the dataset
    baseline_time = time.time()
    dataset = get_dataset(configs["data"]["dataset"],
                          configs["data"]["data_dir"])

    privacy_game = configs["audit"]["privacy_game"]

    # Check the auditing game.
    if privacy_game in ["avg_privacy_loss_training_algo", "privacy_loss_model"]:
        # Load the trained models
        if model_metadata_list["current_idx"] > 0:
            target_model_idx_list = load_existing_target_model(
                len(dataset), model_metadata_list, configs
            )
            trained_target_models_list = load_existing_models(
                model_metadata_list,
                target_model_idx_list,
                configs["train"]["model_name"],
            )
            trained_target_dataset_list = load_dataset_for_existing_models(
                len(dataset),
                model_metadata_list,
                target_model_idx_list,
                configs["data"],
            )
            num_target_models = configs["train"]["num_target_model"] - \
                len(trained_target_dataset_list)
        else:
            target_model_idx_list = []
            trained_target_models_list = []
            trained_target_dataset_list = []
            num_target_models = configs["train"]["num_target_model"]

        # Prepare the datasets
        print(25 * ">" + "Prepare the the datasets")
        data_split_info = prepare_datasets(
            len(dataset), num_target_models, configs["data"]
        )

        logger.info(
            "Prepare the datasets costs %0.5f seconds", time.time() - baseline_time
        )

        # Prepare the target models
        print(25 * ">" + "Prepare the the target models")
        baseline_time = time.time()

        new_model_list, model_metadata_list, new_target_model_idx_list = prepare_models(
            log_dir, dataset, data_split_info, configs["train"], model_metadata_list
        )

        model_list = [*new_model_list, *trained_target_models_list]
        data_split_info["split"] = [
            *data_split_info["split"], *trained_target_dataset_list]
        target_model_idx_list = [
            *new_target_model_idx_list, *target_model_idx_list]

        logger.info(
            "Prepare the target model costs %0.5f seconds", time.time() - baseline_time
        )

        # Prepare the information sources
        print(25 * ">" + "Prepare the information source, including attack models")
        baseline_time = time.time()
        (
            target_info_source,
            reference_info_source,
            metrics,
            log_dir_list,
            model_metadata_list,
        ) = prepare_information_source(
            log_dir,
            dataset,
            data_split_info,
            model_list,
            configs["audit"],
            model_metadata_list,
            target_model_idx_list,
        )
        logger.info(
            "Prepare the information source costs %0.5f seconds",
            time.time() - baseline_time,
        )

        # Call core of privacy meter
        print(25 * ">" + "Auditing the privacy risk")
        baseline_time = time.time()
        audit_obj = Audit(
            metrics=metrics,
            inference_game_type=inference_game_type,
            target_info_sources=target_info_source,
            reference_info_sources=reference_info_source,
            fpr_tolerances=None,
            logs_directory_names=log_dir_list,
        )
        audit_obj.prepare()
        audit_results = audit_obj.run()
        logger.info(
            "Prepare the privacy meter result costs %0.5f seconds",
            time.time() - baseline_time,
        )

        # Generate the privacy risk report
        print(25 * ">" + "Generating privacy risk report")
        baseline_time = time.time()
        prepare_priavcy_risk_report(
            log_dir,
            audit_results,
            configs["audit"],
            save_path=f"{log_dir}/{configs['audit']['report_log']}",
        )
        print(100 * "#")

        logger.info(
            "Prepare the plot for the privacy risk report costs %0.5f seconds",
            time.time() - baseline_time,
        )

        logger.info(
            "Run the priavcy meter for the all steps costs %0.5f seconds",
            time.time() - start_time,
        )

    # Auditing the priavcy risk for an individual data point
    elif configs["audit"]["privacy_game"] == "privacy_loss_sample":

        # Load existing models that match the requirement
        matched_in_idx = load_models_with_data_idx_list(
            model_metadata_list, [configs["train"]["idx"]]
        )
        matched_out_idx = load_models_without_data_idx_list(
            model_metadata_list, [configs["train"]["idx"]]
        )

        # Train additional models if the existing models are not enough
        if len(matched_in_idx) < configs["train"]["num_in_models"]:
            data_split_info_in = prepare_datasets_for_sample_privacy_risk(
                len(dataset),
                configs["train"]["num_in_models"],
                configs["train"]["num_in_models"] - len(matched_in_idx),
                configs["train"]["idx"],
                configs["data"],
                "include",
                model_metadata_list,
            )
            in_model_list, model_metadata_list, matched_in_idx = prepare_models(
                log_dir,
                dataset,
                data_split_info_in,
                configs["train"],
                model_metadata_list,
                matched_in_idx,
            )
        else:
            in_model_list, model_metadata_list, matched_in_idx = prepare_models(
                log_dir,
                dataset,
                {"split": []},
                configs["train"],
                model_metadata_list,
                matched_in_idx[: configs["train"]["num_in_models"]],
            )

        if len(matched_out_idx) < configs["train"]["num_out_models"]:
            data_split_info_out = prepare_datasets_for_sample_privacy_risk(
                len(dataset),
                configs["train"]["num_out_models"],
                configs["train"]["num_out_models"] - len(matched_out_idx),
                configs["train"]["idx"],
                configs["data"],
                "exclude",
                model_metadata_list,
            )
            out_model_list, model_metadata_list, matched_out_idx = prepare_models(
                log_dir,
                dataset,
                data_split_info_out,
                configs["train"],
                model_metadata_list,
                matched_out_idx,
            )
        else:
            out_model_list, model_metadata_list, matched_out_idx = prepare_models(
                log_dir,
                dataset,
                {"split": []},
                configs["train"],
                model_metadata_list,
                matched_out_idx[: configs["train"]["num_out_models"]],
            )

        # Obtain models trained on train.idx and without it
        in_model_list_pm = [
            PytorchModelTensor(
                model_obj=model, loss_fn=nn.CrossEntropyLoss(), batch_size=1000
            )
            for model in in_model_list
        ]
        out_model_list_pm = [
            PytorchModelTensor(
                model_obj=model, loss_fn=nn.CrossEntropyLoss(), batch_size=1000
            )
            for model in out_model_list
        ]

        # Test the models' performance on the data indicated by the audit.idx
        data, targets = get_dataset_subset(dataset, [configs["audit"]["idx"]])
        in_signal = np.array(
            [model.get_loss(data, targets).item()
             for model in in_model_list_pm]
        )
        out_signal = np.array(
            [model.get_loss(data, targets).item()
             for model in out_model_list_pm]
        )

        # Rescale the loss
        in_signal = in_signal + 1e-17  # avoid nan
        in_signal = np.log(
            np.divide(np.exp(-in_signal), (1 - np.exp(-in_signal))))
        out_signal = out_signal + 1e-17  # avoid nan
        out_signal = np.log(
            np.divide(np.exp(-out_signal), (1 - np.exp(-out_signal))))

        # Generate the privacy risk report
        labels = np.concatenate(
            [np.ones(in_signal.shape[0]), np.zeros(out_signal.shape[0])]
        )
        histogram = sns.histplot(
            data=pd.DataFrame(
                {
                    "Signal": np.concatenate([in_signal, out_signal]),
                    "Membership": [
                        f"In ({configs['train']['idx']})"
                        if y == 1
                        else f"Out ({configs['train']['idx']})"
                        for y in labels
                    ],
                }
            ),
            x="Signal",
            hue="Membership",
            element="step",
            kde=True,
        )
        plt.grid()
        plt.xlabel("Signal value")
        plt.ylabel("Number of Models")
        plt.title(f"Signal histogram for data point {configs['audit']['idx']}")
        plt.savefig(
            f"{log_dir}/{configs['audit']['report_log']}/individual_pr_{configs['train']['idx']}_{configs['audit']['idx']}.png"
        )
