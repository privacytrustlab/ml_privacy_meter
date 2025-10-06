"""This file is the main entry point for running the privacy auditing tool."""

import math

import torch

from audit import AuditingEngine, get_average_audit_results
from dataset import get_dataset
from models import get_models
from utils import setup, timer

# Enable benchmark mode in cudnn to improve performance when input sizes are consistent
torch.backends.cudnn.benchmark = True

total_timer = timer(None, "Total runtime")

@total_timer
def main():
    # Setup configurations, directories, and logger
    configs, directories, log_dir, logger = setup()
    total_timer.set_logger(logger)

    # Load the dataset
    with timer(logger, "Loading dataset"):
        dataset, population = get_dataset(configs["data"]["dataset"], directories["data_dir"], logger)

    # Define experiment parameters
    num_experiments = configs["run"]["num_experiments"]
    num_reference_models = configs["audit"]["num_ref_models"]
    num_model_pairs = max(math.ceil(num_experiments / 2.0), num_reference_models + 1)

    # Load or train models
    with timer(logger, "Model loading/training"):
        models_list, memberships = get_models(log_dir, dataset, num_model_pairs, configs, logger)

    # Audit the models
    audit = AuditingEngine(configs, directories, log_dir, logger)

    audit.sample_auditing_dataset(dataset, memberships) #type:ignore
    audit.get_model_signals(models_list, population)
    audit.audit()

    # Get average audit results across all experiments
    if num_experiments > 1:
        get_average_audit_results(
            directories["report_dir"], audit.mia_score_list, audit.membership_list, logger
        )



if __name__ == "__main__":
    main()
