"""This file is the main entry point for running the priavcy auditing."""
import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, "../experiments/")

import random

import numpy as np
import torch
import yaml
from augment import get_signal_on_augmented_data
from core import (
    load_existing_models,
    prepare_datasets_for_online_attack,
    prepare_models,
)
from dataset import get_dataset, get_dataset_subset
from plot import plot_compare_roc
from scipy.stats import norm
from sklearn.metrics import auc, roc_curve
from torch import nn
from util import check_configs

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
    random.seed(configs["run"]["random_seed"])

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

    baseline_time = time.time()
    p_ratio = configs["data"]["keep_ratio"]
    dataset_size = configs["data"]["dataset_size"]

    number_of_models_total = (
        configs["train"]["num_in_models"]
        + configs["train"]["num_out_models"]
        + configs["train"]["num_target_model"]
    )
    (
        data_split_info,
        keep_matrix,
        target_data_index,
    ) = prepare_datasets_for_online_attack(
        len(dataset),
        dataset_size,
        num_models=(number_of_models_total),
        keep_ratio=p_ratio,
        is_uniform=False,
    )
    data, targets = get_dataset_subset(
        dataset,
        target_data_index,
        configs["train"]["model_name"],
        device=configs["train"]["device"],
    )
    population_idx = np.setdiff1d(np.arange(len(dataset)), target_data_index)
    population_data, population_targets = get_dataset_subset(
        dataset,
        population_idx,
        configs["train"]["model_name"],
        device=configs["train"]["device"],
    )

    logger.info(
        "Prepare the datasets costs %0.5f seconds",
        time.time() - baseline_time,
    )
    baseline_time = time.time()
    signals = []
    population_signals = []
    if model_metadata_list["current_idx"] == 0:
        # check if the models are trained
        (model_list, model_metadata_dict, trained_model_idx_list) = prepare_models(
            log_dir,
            dataset,
            data_split_info,
            configs["train"],
            model_metadata_list,
            configs["data"]["dataset"],
        )
        logger.info(
            "Prepare the models costs %0.5f seconds",
            time.time() - baseline_time,
        )
        baseline_time = time.time()
        for model in model_list:
            model_pm = PytorchModelTensor(
                model_obj=model,
                loss_fn=nn.CrossEntropyLoss(),
                device=configs["audit"]["device"],
                batch_size=configs["audit"]["audit_batch_size"],
            )
            signals.append(
                get_signal_on_augmented_data(
                    model_pm,
                    data,
                    targets,
                    method=configs["audit"]["augmentation"],
                )
            )
            population_signals.append(
                get_signal_on_augmented_data(
                    model_pm,
                    population_data,
                    population_targets,
                    method=configs["audit"]["augmentation"],
                )
            )

        logger.info(
            "Prepare the signals costs %0.5f seconds",
            time.time() - baseline_time,
        )
    else:
        for idx in range(number_of_models_total):
            print("Load the model and compute signals for model %d" % idx)
            model_pm = PytorchModelTensor(
                model_obj=load_existing_models(
                    model_metadata_list,
                    [idx],
                    configs["train"]["model_name"],
                    dataset,
                    configs["data"]["dataset"],
                )[0],
                loss_fn=nn.CrossEntropyLoss(),
                device=configs["audit"]["device"],
                batch_size=10000,
            )
            signals.append(
                get_signal_on_augmented_data(
                    model_pm,
                    data,
                    targets,
                    method=configs["audit"]["augmentation"],
                )
            )
            population_signals.append(
                get_signal_on_augmented_data(
                    model_pm,
                    population_data,
                    population_targets,
                    method=configs["audit"]["augmentation"],
                )
            )
        logger.info(
            "Prepare the signals costs %0.5f seconds",
            time.time() - baseline_time,
        )
    baseline_time = time.time()
    signals = np.array(signals)
    population_signals = np.array(population_signals)
    # number of models we want to consider as test
    n_test = 1
    target_signal = signals[:n_test, :]
    reference_signals = signals[n_test:, :]
    reference_keep_matrix = keep_matrix[n_test:, :]
    membership = keep_matrix[:n_test, :]
    in_signals = []
    out_signals = []
    population_signals = population_signals[:n_test, :]

    for data_idx in range(len(target_data_index)):
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

    all_fpr_list = []
    all_tpr_list = []
    all_auc_list = []
    all_alg_list = [
        "reference_in_out_logit_pdf_fixed",
        "reference_in_out_logit_pdf",
        "reference_out_logits_percentile_fixed",
        "reference_out_logits_percentile",
        "reference_out_logits_pdf_fixed",
        "reference_out_logits_pdf",
        "population",
    ]

    print(100 * "#")

    for alg in all_alg_list:
        prediction = []
        answers = []
        for ans, sc in zip(membership, target_signal):
            if alg == "reference_in_out_logit_pdf_fixed":
                mean_in = np.median(in_signals, 1)
                mean_out = np.median(out_signals, 1)
                std_in = np.std(in_signals)
                std_out = np.std(in_signals)

                pr_in = -norm.logpdf(sc, mean_in, std_in + 1e-30)
                pr_out = -norm.logpdf(sc, mean_out, std_out + 1e-30)
                score = pr_in - pr_out
            elif alg == "reference_in_out_logit_pdf":
                mean_in = np.median(in_signals, 1)
                mean_out = np.median(out_signals, 1)
                std_in = np.std(in_signals, 1)
                std_out = np.std(out_signals, 1)
                pr_in = -norm.logpdf(sc, mean_in, std_in + 1e-30)
                pr_out = -norm.logpdf(sc, mean_out, std_out + 1e-30)
                score = pr_in - pr_out
            elif alg == "reference_out_logits_percentile_fixed":
                mean_out = np.median(out_signals, 1)
                std_out = np.std(out_signals)
                pr_out = norm.cdf(sc, mean_out, std_out + 1e-30)
                score = -pr_out
            elif alg == "reference_out_logits_percentile":
                mean_out = np.median(out_signals, 1)
                std_out = np.std(out_signals, 1)
                pr_out = norm.cdf(sc, mean_out, std_out + 1e-30)
                score = -pr_out
            elif alg == "population":
                mean_out = np.median(population_signals)
                std_out = np.std(population_signals)
                pr_out = norm.cdf(sc, mean_out, std_out + 1e-30)
                score = -pr_out
            elif alg == "reference_out_logits_pdf_fixed":
                mean_out = np.median(out_signals, 1)
                std_out = np.std(out_signals)
                pr_out = -norm.logpdf(sc, mean_out, std_out + 1e-30)
                score = -pr_out
            elif alg == "reference_out_logits_pdf":
                mean_out = np.median(out_signals, 1)
                std_out = np.std(out_signals, 1)
                pr_out = -norm.logpdf(sc, mean_out, std_out + 1e-30)
                score = -pr_out
            else:
                raise ValueError("Unknown algorithm")

            if len(score.shape) == 2:  # the score is of size (data_size, num_augments)
                prediction.extend(score.mean(1))
                fpr_list, tpr_list, _ = roc_curve(ans, -score.mean(1))
            else:
                prediction.extend(score)
                fpr_list, tpr_list, _ = roc_curve(ans, -score)
            answers.extend(ans)
            acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
            roc_auc = auc(fpr_list, tpr_list)

        prediction = np.array(prediction)
        answers = np.array(answers, dtype=bool)
        fpr_list, tpr_list, _ = roc_curve(answers.ravel(), -prediction.ravel())
        ref_in_acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
        ref_in_roc_auc = auc(fpr_list, tpr_list)
        ref_in_low = tpr_list[np.where(fpr_list < 0.001)[0][-1]]

        print(
            f"{alg}     AUC: %.4f, Accuracy: %.4f, TPR@0.1%%FPR: %.4f"
            % (ref_in_roc_auc, ref_in_acc, ref_in_low)
        )
        all_fpr_list.append(fpr_list)
        all_tpr_list.append(tpr_list)
        all_auc_list.append(ref_in_roc_auc)

    print(100 * "#")

    plot_compare_roc(
        all_fpr_list,
        all_tpr_list,
        all_auc_list,
        all_alg_list,
        f"{log_dir}/{configs['audit']['report_log']}/Combined_ROC.png",
        log_scale=False,
    )

    plot_compare_roc(
        all_fpr_list,
        all_tpr_list,
        all_auc_list,
        all_alg_list,
        f"{log_dir}/{configs['audit']['report_log']}/Combined_ROC_log_scaled.png",
        log_scale=True,
    )

    ############################
    # END
    ############################
    logger.info(
        "Run the priavcy meter for the all steps costs %0.5f seconds",
        time.time() - start_time,
    )
