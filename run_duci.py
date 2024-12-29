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

    ############################ Generate signals (softmax outputs) for all models ############################
    baseline_time = time.time()
    signals = get_model_signals(models_list, auditing_dataset, configs, logger) # num_samples * num_models
    auditing_membership = auditing_membership.T
    assert signals.shape == auditing_membership.shape, f"signals or auditing_membership has incorrect shape (num_samples * num_models): {signals.shape} vs {auditing_membership.shape}"
    logger.info("Preparing signals took %0.5f seconds", time.time() - baseline_time)

    ######################################  Perform DUCI ######################################
    baseline_time = time.time()
    target_model_indices = list(range(num_experiments))

    ############################  Input your own reference model indices ############################
    #Sample: construct reference models
    reference_model_indices_all = []
    for target_model_idx in target_model_indices:
        paired_model_idx = (
            target_model_idx + 1 if target_model_idx % 2 == 0 else target_model_idx - 1
        )
        # Select reference models from non-target and non-paired model indices
        ref_indices = [
            i
            for i in range(signals.shape[1])
            if i != target_model_idx and i != paired_model_idx
        ][: 2 * num_reference_models]
        reference_model_indices_all.append(np.array(ref_indices))


    logger.info(f"Initiate DUCI for target models: {target_model_indices}")
    DUCI_instance = DUCI(logger)

    logger.info("Collecting membership prediction for each sample in the target dataset on target models and reference models.")
    logger.info("Predicting the proportion of dataset usage on target models.")
    duci_preds, true_proportions, errors = DUCI_instance.pred_proportions(
        target_model_indices, 
        reference_model_indices_all, 
        signals,
        auditing_membership,
    )

    if len(target_model_indices) > 1:
        logger.info(
            "DUCI %0.1f seconds", time.time() - baseline_time
        )
        logger.info(f"Average prediction errors: {np.mean(errors)}")
        logger.info(f"All prediction errors: {errors}")
        logger.info(f"Prediction details: DUCI predictions: {duci_preds}, True proportions: {true_proportions}")
    
    # Visualize the results
    # logger.info("Visualizing the results...")
    # DUCI_instance.visualize_results(
    #     duci_preds, 
    #     true_proportions, 
    #     target_model_indices,
    #     directories["report_dir"],
    #     logger
    # )


if __name__ == "__main__":
    main()
