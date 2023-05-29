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
from argument import get_signal_on_argumented_data
from core import (
    load_dataset_for_existing_models,
    load_existing_models,
    load_existing_target_model,
    prepare_datasets,
    prepare_datasets_for_online_attack,
    prepare_datasets_for_sample_privacy_risk,
    prepare_information_source,
    prepare_models,
    prepare_priavcy_risk_report,
)
from dataset import get_dataset, get_dataset_subset
from plot import plot_roc, plot_signal_histogram
from scipy.stats import norm
from sklearn.metrics import auc, roc_curve
from torch import nn
from util import (
    check_configs,
    load_leave_one_out_models,
    load_models_with_data_idx_list,
    load_models_without_data_idx_list,
    sweep,
)

from privacy_meter.audit import Audit
from privacy_meter.model import PytorchModelTensor

torch.backends.cudnn.benchmark = True


def setup_log(name: str, save_file: bool):
    """Generate the logger for the current run.
    Args:
        name (str): Logging file name.
        save_file (bool): Flag about whether to save to file.
    Returns:
        logging.Logger: Logger object for the current run.
    """
    my_logger = logging.getLogger(name)
    my_logger.setLevel(logging.INFO)
    if save_file:
        log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
        filename = f"log_{name}.log"
        log_handler = logging.FileHandler(filename, mode="w")
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(log_format)
        my_logger.addHandler(log_handler)

    return my_logger

def metric_results(fpr_list, tpr_list):
    acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
    roc_auc = auc(fpr_list, tpr_list)

    one_percent = tpr_list[np.where(fpr_list < 0.01)[0][-1]]
    tenth_percent = tpr_list[np.where(fpr_list < 0.001)[0][-1]]
    hundredth_percent = tpr_list[np.where(fpr_list < 0.0001)[0][-1]]

    return roc_auc, acc, one_percent, tenth_percent, hundredth_percent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cf",
        type=str,
        default="experiments/config_models_online.yaml",
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
    logger = setup_log("time_analysis", configs["run"]["time_log"])

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
    dataset = get_dataset(configs["data"]["dataset"], configs["data"]["data_dir"])

    privacy_game = configs["audit"]["privacy_game"]

    ############################
    # Privacy auditing for a model or an algorithm
    ############################
    if (
        privacy_game in ["avg_privacy_loss_training_algo", "privacy_loss_model"]
        and "online" not in configs["audit"]["algorithm"]
    ):
        # Load the trained models from disk
        if model_metadata_list["current_idx"] > 0:
            target_model_idx_list = load_existing_target_model(
                len(dataset), model_metadata_list, configs
            )
            trained_target_dataset_list = load_dataset_for_existing_models(
                len(dataset),
                model_metadata_list,
                target_model_idx_list,
                configs["data"],
            )

            trained_target_models_list = load_existing_models(
                model_metadata_list,
                target_model_idx_list,
                configs["train"]["model_name"],
                # trained_target_dataset_list,
                dataset,
            )
            num_target_models = configs["train"]["num_target_model"] - len(
                trained_target_dataset_list
            )
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
            *data_split_info["split"],
            *trained_target_dataset_list,
        ]
        target_model_idx_list = [*new_target_model_idx_list, *target_model_idx_list]

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
            configs["train"]["model_name"],
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

    ############################
    # Privacy auditing for a sample
    ############################
    elif configs["audit"]["privacy_game"] == "privacy_loss_sample":
        # Load existing models that match the requirement
        assert (
            "data_idx" in configs["train"]
        ), "data_idx in config.train is not specified"
        assert (
            "data_idx" in configs["audit"]
        ), "data_idx in config.audit is not specified"

        in_model_idx_list = load_models_with_data_idx_list(
            model_metadata_list, [configs["train"]["data_idx"]]
        )
        model_in_list = load_existing_models(
            model_metadata_list,
            in_model_idx_list,
            configs["train"]["model_name"],
            dataset,
        )
        # Train additional models if the existing models are not enough
        if len(in_model_idx_list) < configs["train"]["num_in_models"]:
            data_split_info_in = prepare_datasets_for_sample_privacy_risk(
                len(dataset),
                configs["train"]["num_in_models"] - len(in_model_idx_list),
                configs["train"]["data_idx"],
                configs["data"],
                "include",
                "leave_one_out",
                model_metadata_list,
            )
            new_in_model_list, model_metadata_list, new_matched_in_idx = prepare_models(
                log_dir,
                dataset,
                data_split_info_in,
                configs["train"],
                model_metadata_list,
            )
            model_in_list = [*new_in_model_list, *model_in_list]
            in_model_idx_list = [*new_matched_in_idx, *in_model_idx_list]
        in_model_list_pm = [
            PytorchModelTensor(
                model_obj=model,
                loss_fn=nn.CrossEntropyLoss(),
                batch_size=1000,
                device=configs["audit"]["device"],
            )
            for model in model_in_list
        ]
        if configs["data"]["split_method"] == "uniform":
            out_model_idx_list = load_models_without_data_idx_list(
                model_metadata_list, [configs["train"]["data_idx"]]
            )
        elif configs["data"]["split_method"] == "leave_one_out":
            out_model_idx_list = load_leave_one_out_models(
                model_metadata_list, [configs["train"]["data_idx"]], in_model_idx_list
            )
        else:
            raise ValueError("The split method is not supported")

        model_out_list = load_existing_models(
            model_metadata_list,
            out_model_idx_list,
            configs["train"]["model_name"],
            dataset,
        )
        # Train additional models if the existing models are not enough
        if len(out_model_idx_list) < configs["train"]["num_out_models"]:
            data_split_info_out = prepare_datasets_for_sample_privacy_risk(
                len(dataset),
                configs["train"]["num_out_models"] - len(out_model_idx_list),
                configs["train"]["data_idx"],
                configs["data"],
                "exclude",
                configs["data"]["split_method"],
                model_metadata_list,
                in_model_idx_list,
            )
            (
                new_out_model_list,
                model_metadata_list,
                new_matched_out_idx,
            ) = prepare_models(
                log_dir,
                dataset,
                data_split_info_out,
                configs["train"],
                model_metadata_list,
            )
            model_out_list = [*new_out_model_list, *model_out_list]
            out_model_idx_list = [*new_matched_out_idx, *out_model_idx_list]

        out_model_list_pm = [
            PytorchModelTensor(
                model_obj=model,
                loss_fn=nn.CrossEntropyLoss(),
                batch_size=1000,
                device=configs["audit"]["device"],
            )
            for model in model_out_list
        ]

        # Test the models' performance on the data indicated by the audit.idx
        data, targets = get_dataset_subset(
            dataset, [configs["audit"]["data_idx"]], configs["audit"]["model_name"], device=configs["audit"]["device"]
        )
        in_signal = np.array(
            [
                model.get_rescaled_logits(data, targets).item()
                for model in in_model_list_pm
            ]
        )
        out_signal = np.array(
            [
                model.get_rescaled_logits(data, targets).item()
                for model in out_model_list_pm
            ]
        )

        # Generate the privacy risk report
        plot_signal_histogram(
            in_signal,
            out_signal,
            configs["train"]["data_idx"],
            configs["audit"]["data_idx"],
            f"{log_dir}/{configs['audit']['report_log']}/individual_pr_{configs['train']['data_idx']}_{configs['audit']['data_idx']}.png",
        )
        fpr_list, tpr_list, roc_auc = sweep(in_signal, out_signal)
        plot_roc(
            fpr_list,
            tpr_list,
            roc_auc,
            f"{log_dir}/{configs['audit']['report_log']}/individual_pr_roc_{configs['train']['data_idx']}_{configs['audit']['data_idx']}.png",
        )

    ############################
    # Privacy auditing for an model with online attack (i.e., adversary trains models with/without each target points)
    ############################
    elif "online" in configs["audit"]["algorithm"]:
        # The following code is modified from the original code in the repo: https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021
        baseline_time = time.time()
        p_ratio = configs["data"]["keep_ratio"]
        dataset_size = configs["data"]["dataset_size"]
        number_of_models_lira = configs["train"]["num_in_models"] + configs["train"]["num_out_models"] + configs["train"]["num_target_model"]
        data_split_info, keep_matrix = prepare_datasets_for_online_attack(
            dataset_size,
            num_models=(
                number_of_models_lira
            ),
            keep_ratio=p_ratio,
            is_uniform=False,
        )
        data, targets = get_dataset_subset(
            dataset, np.arange(dataset_size), configs["train"]["model_name"], device=configs["train"]["device"]
        )  # only the train dataset we want to attack
        logger.info(
            "Prepare the datasets costs %0.5f seconds",
            time.time() - baseline_time,
        )
        baseline_time = time.time()
        if model_metadata_list["current_idx"] == 0:
            (model_list, model_metadata_dict, trained_model_idx_list) = prepare_models(
                log_dir,
                dataset,
                data_split_info,
                configs["train"],
                model_metadata_list,
            )
            logger.info(
                "Prepare the models costs %0.5f seconds",
                time.time() - baseline_time,
            )
            baseline_time = time.time()
            signals = []
            for i, model in enumerate(model_list):
                model_init = lambda: PytorchModelTensor(
                    model_obj=model,
                    loss_fn=nn.CrossEntropyLoss(),
                    device=configs["audit"]["device"],
                    batch_size=int(configs["audit"]["audit_batch_size"]),
                )
                signals.append(
                    get_signal_on_argumented_data(
                        model_pm,
                        data, # but we query a smaller dataset..
                        targets,
                        method=configs["audit"]["argumentation"],
                    )
                )
            logger.info(
                "Prepare the signals costs %0.5f seconds",
                time.time() - baseline_time,
            )
        else:
            baseline_time = time.time()
            signals = []
            number_of_models_lira = configs["train"]["num_in_models"] + configs["train"]["num_out_models"] + configs["train"]["num_target_model"]
            for idx in range(number_of_models_lira): # we consider that we train lira online setting first.
                print("load the model and compute signals for model %d" % idx)
                model_pm = PytorchModelTensor(
                    model_obj=load_existing_models(
                        model_metadata_list,
                        [idx],
                        configs["train"]["model_name"],
                        dataset,
                        device=configs["audit"]["device"]
                    )[0],
                    loss_fn=nn.CrossEntropyLoss(),
                    device=configs["audit"]["device"],
                    batch_size=int(configs["audit"]["audit_batch_size"]),
                )
                signals.append(
                    get_signal_on_argumented_data(
                        model_pm,
                        data,
                        targets,
                        method=configs["audit"]["argumentation"],
                    )
                )
            logger.info(
                "Prepare the signals costs %0.5f seconds",
                time.time() - baseline_time,
            )
        baseline_time = time.time()
        signals = np.array(signals)
        target_signal = signals[-1:, :]
        reference_signals = signals[:-1, :]
        reference_keep_matrix = keep_matrix[:-1, :]
        membership = keep_matrix[-1:, :]

        in_signals = []
        out_signals = []

        for data_idx in range(dataset_size):
            in_signals.append(
                reference_signals[reference_keep_matrix[:, data_idx], data_idx]
            )
            out_signals.append(
                reference_signals[~reference_keep_matrix[:, data_idx], data_idx]
            )

        in_size = min(min(map(len, in_signals)), configs["train"]["num_in_models"])
        out_size = min(min(map(len, out_signals)), configs["train"]["num_out_models"])
        in_signals = np.array([x[:in_size] for x in in_signals]).astype("float32")
        out_signals = np.array([x[:out_size] for x in out_signals]).astype("float32")

        mean_in = np.median(in_signals, 1)
        mean_out = np.median(out_signals, 1)
        fix_variance = configs["audit"]["fix_variance"]
        if fix_variance:
            std_in = np.std(in_signals)
            std_out = np.std(in_signals)
        else:
            std_in = np.std(in_signals, 1)
            std_out = np.std(out_signals, 1)

        prediction = []
        answers = []
        for ans, sc in zip(membership, target_signal):
            if configs["audit"]["offline"]:
                pr_in = 0
            else:
                pr_in = -norm.logpdf(sc, mean_in, std_in + 1e-30)
            pr_out = -norm.logpdf(sc, mean_out, std_out + 1e-30)
            score = pr_in - pr_out
            if len(score.shape) == 2:  # the score is of size (data_size, num_arguments)
                prediction.extend(score.mean(1))
            else:
                prediction.extend(score)
            answers.extend(ans)

        prediction = np.array(prediction)
        answers = np.array(answers, dtype=bool)
        print(prediction.shape, answers.shape, prediction, np.isnan(prediction).sum())
        # Last step: compute the metrics
        fpr_list, tpr_list, _ = roc_curve(answers.ravel(), -prediction.ravel())
        acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
        roc_auc = auc(fpr_list, tpr_list)
        logger.info(
            "Prepare the privacy risks results costs %0.5f seconds",
            time.time() - baseline_time,
        )
        low = tpr_list[np.where(fpr_list < 0.001)[0][-1]]
        print("AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f" % (roc_auc, acc, low))
        plot_roc(
            fpr_list,
            tpr_list,
            roc_auc,
            f"{log_dir}/{configs['audit']['report_log']}/online_attack.png",
        )

    ############################
    # END
    ############################
    logger.info(
        "Run the priavcy meter for the all steps costs %0.5f seconds",
        time.time() - start_time,
    )
