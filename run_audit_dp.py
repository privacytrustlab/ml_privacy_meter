"""This file is the main entry point for running the privacy auditing tool."""

import argparse
import math
import time

import torch
import yaml
import numpy as np

from audit import (
    get_average_audit_results,
    audit_models,
    get_all_dp_audit_results,
    get_dp_audit_results_for_k_pos_k_neg,
)
from get_signals import get_model_signals
from models.utils import dp_load_models, train_models, dp_train_models
from util import (
    check_configs,
    setup_log,
    initialize_seeds,
    create_directories,
    load_dataset,
    load_canary_dataset,
    split_dataset_for_training_poisson,
)

# Enable benchmark mode in cudnn to improve performance when input sizes are consistent
torch.backends.cudnn.benchmark = True


def main():
    print(20 * "-")
    print("Privacy Meter Tool!")
    print(20 * "-")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run privacy auditing tool.")
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
    check_configs(configs)

    # Initialize seeds for reproducibility
    initialize_seeds(configs["run"]["random_seed"])

    # Create necessary directories
    log_dir = configs["run"]["log_dir"]
    directories = {
        "log_dir": log_dir,
        "report_dir": f"{log_dir}/report",
        "signal_dir": f"{log_dir}/signals",
        "data_dir": configs["data"]["data_dir"],
    }
    create_directories(directories)

    # Set up logger
    logger = setup_log(
        directories["report_dir"], "time_analysis", configs["run"]["time_log"]
    )

    start_time = time.time()

    # Load the canary dataset
    baseline_time = time.time()
    if configs["dp_audit"].get("canary_dataset", "none") == "none":
        dataset, population = load_dataset(configs, directories["data_dir"], logger)
        canary_dataset = torch.utils.data.Subset(
            dataset, np.arange(configs["dp_audit"]["canary_size"])
        )
    elif configs["dp_audit"].get("canary_dataset", "none") == "cifar10_canary":
        canary_dataset, _ = load_canary_dataset(
            configs, directories["data_dir"], logger
        )
        if configs["dp_audit"]["canary_size"] > len(canary_dataset):
            raise ValueError(
                "canary data size cannot be larger than the whole cifar10 dataset."
            )
        canary_dataset = torch.utils.data.Subset(
            canary_dataset, np.arange(configs["dp_audit"]["canary_size"])
        )
        clean_dataset, population = load_dataset(
            configs, directories["data_dir"], logger
        )
        # subsample clean dataset to ensure that the number of clean samples + the number of canary samples = size of the whole training dataset
        clean_dataset = torch.utils.data.Subset(
            clean_dataset,
            np.arange(configs["dp_audit"]["canary_size"], len(clean_dataset)),
        )
        dataset = torch.utils.data.ConcatDataset([canary_dataset, clean_dataset])
    else:
        raise NotImplementedError(
            f"canary dataset {configs['dp_audit']} is not supported"
        )

    logger.info("Loading dataset took %0.5f seconds", time.time() - baseline_time)

    # Define experiment parameters
    num_experiments = configs["run"]["num_experiments"]
    num_reference_models = configs["audit"]["num_ref_models"]
    num_model_pairs = max(math.ceil(num_experiments / 2.0), num_reference_models + 1)

    # Load or train models
    baseline_time = time.time()
    models_list, memberships = dp_load_models(
        log_dir, dataset, num_model_pairs * 2, configs, logger
    )
    if models_list is None:
        # Split dataset for training two models per pair
        data_splits, memberships = split_dataset_for_training_poisson(
            len(dataset), num_model_pairs
        )
        if configs["dp_audit"]["training_alg"] == "dp":
            models_list = dp_train_models(
                log_dir, dataset, data_splits, memberships, configs, logger
            )
        elif configs["dp_audit"]["training_alg"] == "nondp":
            models_list = train_models(
                log_dir, dataset, data_splits, memberships, configs, logger
            )
    logger.info(
        "Model loading/training took %0.1f seconds", time.time() - baseline_time
    )

    auditing_dataset = canary_dataset
    auditing_membership = memberships[:, : len(canary_dataset)].reshape(
        (memberships.shape[0], len(canary_dataset))
    )

    population = torch.utils.data.Subset(
        population,
        np.random.choice(
            len(population),
            configs["audit"].get("population_size", len(population)),
            replace=False,
        ),
    )

    # Generate signals (softmax outputs) for all models
    baseline_time = time.time()
    signals = get_model_signals(models_list, auditing_dataset, configs, logger)
    population_signals = get_model_signals(
        models_list, population, configs, logger, is_population=True
    )
    logger.info("Preparing signals took %0.5f seconds", time.time() - baseline_time)

    # Perform the privacy audit
    baseline_time = time.time()
    target_model_indices = list(range(num_experiments))
    mia_score_list, membership_list = audit_models(
        f"{directories['report_dir']}/exp",
        target_model_indices,
        signals,
        population_signals,
        auditing_membership,
        num_reference_models,
        logger,
        configs,
    )

    get_all_dp_audit_results(
        directories["report_dir"], mia_score_list, membership_list, logger
    )

    # k_neg = 900
    # k_pos = 59700
    # get_dp_audit_results_for_k_pos_k_neg(directories["report_dir"], mia_score_list, membership_list, logger, k_pos, k_neg)

    if len(target_model_indices) > 1:
        logger.info(
            "Auditing privacy risk took %0.1f seconds", time.time() - baseline_time
        )

    # Get average audit results across all experiments
    if len(target_model_indices) > 1:
        get_average_audit_results(
            directories["report_dir"], mia_score_list, membership_list, logger
        )

    logger.info("Total runtime: %0.5f seconds", time.time() - start_time)


if __name__ == "__main__":
    main()
