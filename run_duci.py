"""This file is the main entry point for running the privacy auditing tool."""

import argparse
import math
import time

import torch
import yaml
import numpy as np

from audit import get_average_audit_results, audit_models, sample_auditing_dataset
from get_signals import get_model_signals
from models.utils import load_models, train_models, split_dataset_for_training
from util import (
    check_configs,
    setup_log,
    initialize_seeds,
    create_directories,
    load_dataset,
)
from module_duci import DUCI

# Enable benchmark mode in cudnn to improve performance when input sizes are consistent
torch.backends.cudnn.benchmark = True


def main():
    print(20 * "-")
    print("Run Dataset Usage Cardinality Inference!")
    print(20 * "-")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run DUCI using Privacy Meter.")
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

    # Load the dataset
    baseline_time = time.time()
    dataset = load_dataset(configs, directories["data_dir"], logger)
    logger.info("Loading dataset took %0.5f seconds", time.time() - baseline_time)

    # Define experiment parameters
    num_experiments = configs["run"]["num_experiments"]
    num_reference_models = configs["audit"]["num_ref_models"]
    num_model_pairs = max(math.ceil(num_experiments / 2.0), num_reference_models + 1)

    # Load or train models
    baseline_time = time.time()
    models_list, memberships = load_models(
        log_dir, dataset, num_model_pairs * 2, configs, logger
    )
    if models_list is None:
        # Split dataset for training two models per pair
        data_splits, memberships = split_dataset_for_training(
            len(dataset), num_model_pairs
        )
        models_list = train_models(
            log_dir, dataset, data_splits, memberships, configs, logger
        )
    logger.info(
        "Model loading/training took %0.1f seconds", time.time() - baseline_time
    )

    auditing_dataset, auditing_membership = sample_auditing_dataset(
        configs, dataset, logger, memberships
    )

    # Generate signals (softmax outputs) for all models
    baseline_time = time.time()
    signals = get_model_signals(models_list, auditing_dataset, configs, logger) # num_models * num_samples
    logger.info("Preparing signals took %0.5f seconds", time.time() - baseline_time)

    # Perform DUCI
    baseline_time = time.time()
    target_model_indices = list(range(num_experiments))

    logger.info(f"Initiate DUCI for target models: {target_model_indices}")
    DUCI_instance = DUCI(num_reference_models)

    logger.info("Collecting membership prediction for each sample in the target dataset on target models and reference models.")
    mia_score_list, membership_list, offline_a = DUCI_instance.get_ind_signals(
        target_model_indices,
        signals,
        auditing_membership,
        num_reference_models,
        logger,
        configs,
    )

    logger.info("Predicting the proportion of dataset usage on target models.") #TODO: remove the baseline
    _, duci = DUCI_instance.pred_proportion(
        target_model_indices, 
        mia_score_list, 
        membership_list, 
        offline_a
    )

    if len(target_model_indices) > 1:
        logger.info(
            "DUCI %0.1f seconds", time.time() - baseline_time
        )

    # Get average prediction errors across all experiments
    if len(target_model_indices) > 1:
        logger.info(f"Average prediction errors: agg-MIA baseline {np.mean(agg_mia_baseline)} | DUCI: {np.mean(duci)}")


    logger.info("Total runtime: %0.5f seconds", time.time() - start_time)


if __name__ == "__main__":
    main()
