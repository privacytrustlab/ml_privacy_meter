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
from scipy.stats import norm
from sklearn.metrics import auc, roc_curve
from torch import nn
from util import (
    check_configs,
    load_leave_one_out_models,
    load_models_with_data_idx_list,
    load_models_without_data_idx_list,
)

from privacy_meter.audit import Audit
from privacy_meter.model import PytorchModelTensor

torch.backends.cudnn.benchmark = True


def setup_log(name: str, save_file: bool) -> logging.Logger:
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
        default="config_models_online.yaml",
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

    # Check the auditing game.
    if (
        privacy_game in ["avg_privacy_loss_training_algo", "privacy_loss_model"]
        and "online" not in configs["audit"]["algorithm"]
    ):
        # Load the trained models from disk
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
        data, targets = get_dataset_subset(dataset, [configs["audit"]["data_idx"]])
        in_signal = np.array(
            [model.get_loss(data, targets).item() for model in in_model_list_pm]
        )
        out_signal = np.array(
            [model.get_loss(data, targets).item() for model in out_model_list_pm]
        )

        # Rescale the loss
        in_signal = in_signal + 1e-17  # avoid nan
        in_signal = - np.log(np.divide(np.exp(-in_signal), (1 - np.exp(-in_signal))))
        out_signal = out_signal + 1e-17  # avoid nan
        out_signal = - np.log(np.divide(np.exp(-out_signal), (1 - np.exp(-out_signal))))

        # Generate the privacy risk report
        labels = np.concatenate(
            [np.ones(in_signal.shape[0]), np.zeros(out_signal.shape[0])]
        )
        histogram = sns.histplot(
            data=pd.DataFrame(
                {
                    "Signal": np.concatenate([in_signal, out_signal]),
                    "Membership": [
                        f"In ({configs['train']['data_idx']})"
                        if y == 1
                        else f"Out ({configs['train']['data_idx']})"
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
        plt.title(f"Signal histogram for data point {configs['audit']['data_idx']}")
        plt.savefig(
            f"{log_dir}/{configs['audit']['report_log']}/individual_pr_{configs['train']['data_idx']}_{configs['audit']['data_idx']}.png"
        )

        # Generate the ROC
        all_signals = np.concatenate([in_signal, out_signal])
        all_signals.sort()
        tpr_list = []
        fpr_list = []
        for threshold in all_signals:
            tp = np.sum(in_signal < threshold)
            fp = np.sum(out_signal < threshold)
            tn = np.sum(out_signal >= threshold)
            fn = np.sum(in_signal >= threshold)
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        roc_auc = np.trapz(x=fpr_list, y=tpr_list)
        range01 = np.linspace(0, 1)
        plt.fill_between(fpr_list, tpr_list, alpha=0.15)
        plt.plot(range01, range01, "--", label="Random guess")
        plt.plot(fpr_list, tpr_list, label="ROC curve")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid()
        plt.legend()
        plt.xlabel("False positive rate (FPR)")
        plt.ylabel("True positive rate (TPR)")
        plt.title("ROC curve")
        plt.text(
            0.7,
            0.3,
            f"AUC = {roc_auc:.03f}",
            horizontalalignment="center",
            verticalalignment="center",
            bbox=dict(facecolor="white", alpha=0.5),
        )
        plt.savefig(
            fname=f"{log_dir}/{configs['audit']['report_log']}/individual_pr_roc_{configs['train']['data_idx']}_{configs['audit']['data_idx']}.png",
            dpi=1000,
        )
        plt.clf()

    elif "online" in configs["audit"]["algorithm"]:
        print("Online attack")
        # The following code is modified from the original code in the repo: https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021
        p_ratio = configs["data"]["keep_ratio"]
        dataset_size = configs["data"]["dataset_size"]
        data_split_info, keep_matrix = prepare_datasets_for_online_attack(
            dataset_size,
            num_models=(
                configs["train"]["num_in_models"]
                + configs["train"]["num_out_models"]
                + configs["train"]["num_target_model"]
            ),
            configs=configs["data"],
            keep_ratio=p_ratio,
            model_metadata_dict=model_metadata_list,
        )
        data, targets = get_dataset_subset(
            dataset, np.arange(dataset_size)
        )  # only the train dataset we want to attack

        (model_list, model_metadata_dict, trained_model_idx_list) = prepare_models(
            log_dir,
            dataset,
            data_split_info,
            configs["train"],
            model_metadata_list,
        )

        signals = []
        for model in model_list:
            model_pm = PytorchModelTensor(
                model_obj=model,
                loss_fn=nn.CrossEntropyLoss(),
                device=configs["audit"]["device"],
                batch_size=10000,
            )
            signals.append(
                get_signal_on_argumented_data(
                    model_pm, data, targets, method="argumented"
                )
            )
        # # # will remove later
        # signals = []
        # for idx in range(17):
        #     print("load the model")
        #     model_pm = PytorchModelTensor(
        #         model_obj=load_existing_models(
        #             model_metadata_list,
        #             [idx],
        #             configs["train"]["model_name"],
        #         )[0],
        #         loss_fn=nn.CrossEntropyLoss(),
        #         device=configs["audit"]["device"],
        #         batch_size=10000,
        #     )
        #     print("compute the signal")
        #     signals.append(
        #         get_signal_on_argumented_data(
        #             model_pm, data, targets, method="argumented"
        #         )
        #     )

        # # Get the logits for each model
        signals = np.array(signals)
        signals = signals + 1e-7
        signals = - np.log(np.divide(np.exp(-signals), (1 - np.exp(-signals))))

        # target model
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

        in_signals = np.array([x[:in_size] for x in in_signals])
        out_signals = np.array([x[:out_size] for x in out_signals])

        mean_in = np.median(in_signals, 1)
        mean_out = np.median(out_signals, 1)
        fix_variance = True
        if fix_variance:
            std_in = np.std(in_signals)
            std_out = np.std(in_signals)
        else:
            std_in = np.std(in_signals, 1)
            std_out = np.std(out_signals, 1)

        prediction = []
        answers = []
        for ans, sc in zip(membership, target_signal):
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

        # Last step: compute the metrics
        fpr_list, tpr_list, _ = roc_curve(answers, -prediction)
        acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
        roc_auc = auc(fpr_list, tpr_list)

        low = tpr_list[np.where(fpr_list < 0.001)[0][-1]]
        print("AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f" % (roc_auc, acc, low))
        range01 = np.linspace(0, 1)
        plt.fill_between(fpr_list, tpr_list, alpha=0.15)
        plt.plot(range01, range01, "--", label="Random guess")
        plt.plot(fpr_list, tpr_list, label="ROC curve")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid()
        plt.legend()
        plt.xlabel("False positive rate (FPR)")
        plt.ylabel("True positive rate (TPR)")
        plt.title("ROC curve")
        plt.text(
            0.7,
            0.3,
            f"AUC = {roc_auc:.03f}",
            horizontalalignment="center",
            verticalalignment="center",
            bbox=dict(facecolor="white", alpha=0.5),
        )
        plt.savefig(
            fname=f"{log_dir}/{configs['audit']['report_log']}/online_attack.png",
            dpi=1000,
        )
        plt.clf()
