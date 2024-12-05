import time
from pathlib import Path

import numpy as np
import torch.utils.data
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Subset

from attacks import tune_offline_a, run_rmia, run_loss
from ramia_scores import get_topk, get_bottomk, trim_mia_scores
from visualize import plot_roc, plot_roc_log


def compute_attack_results(mia_scores, target_memberships):
    """
    Compute attack results (TPR-FPR curve, AUC, etc.) based on MIA scores and membership of samples.

    Args:
        mia_scores (np.array): MIA score computed by the attack.
        target_memberships (np.array): Membership of samples in the training set of target model.

    Returns:
        dict: Dictionary of results, including fpr and tpr list, AUC, TPR at 1%, 0.1% and 0% FPR.
    """
    fpr_list, tpr_list, _ = roc_curve(target_memberships.ravel(), mia_scores.ravel())
    roc_auc = auc(fpr_list, tpr_list)
    one_fpr = tpr_list[np.where(fpr_list <= 0.01)[0][-1]]
    one_tenth_fpr = tpr_list[np.where(fpr_list <= 0.001)[0][-1]]
    zero_fpr = tpr_list[np.where(fpr_list <= 0.0)[0][-1]]

    return {
        "fpr": fpr_list,
        "tpr": tpr_list,
        "auc": roc_auc,
        "one_fpr": one_fpr,
        "one_tenth_fpr": one_tenth_fpr,
        "zero_fpr": zero_fpr,
    }


def get_audit_results(report_dir, model_idx, mia_scores, target_memberships, logger):
    """
    Generate and save ROC plots for attacking a single model.

    Args:
        report_dir (str): Folder for saving the ROC plots.
        model_idx (int): Index of model subjected to the attack.
        mia_scores (np.array): MIA score computed by the attack.
        target_memberships (np.array): Membership of samples in the training set of target model.
        logger (logging.Logger): Logger object for the current run.

    Returns:
        dict: Dictionary of results, including fpr and tpr list, AUC, TPR at 1%, 0.1% and 0% FPR.
    """
    attack_result = compute_attack_results(mia_scores, target_memberships)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Target Model %d: AUC %.4f, TPR@0.1%%FPR of %.4f, TPR@0.0%%FPR of %.4f",
        model_idx,
        attack_result["auc"],
        attack_result["one_tenth_fpr"],
        attack_result["zero_fpr"],
    )

    plot_roc(
        attack_result["fpr"],
        attack_result["tpr"],
        attack_result["auc"],
        f"{report_dir}/ROC_{model_idx}.png",
    )
    plot_roc_log(
        attack_result["fpr"],
        attack_result["tpr"],
        attack_result["auc"],
        f"{report_dir}/ROC_log_{model_idx}.png",
    )

    np.savez(
        f"{report_dir}/attack_result_{model_idx}",
        fpr=attack_result["fpr"],
        tpr=attack_result["tpr"],
        auc=attack_result["auc"],
        one_tenth_fpr=attack_result["one_tenth_fpr"],
        zero_fpr=attack_result["zero_fpr"],
        scores=mia_scores.ravel(),
        memberships=target_memberships.ravel(),
    )
    return attack_result


def get_average_audit_results(report_dir, mia_score_list, membership_list, logger):
    """
    Generate and save ROC plots for attacking multiple models by aggregating all scores and membership labels.

    Args:
        report_dir (str): Folder for saving the ROC plots.
        mia_score_list (list): List of MIA scores for each target model.
        membership_list (list): List of membership labels of each target model.
        logger (logging.Logger): Logger object for the current run.
    """

    mia_scores = np.concatenate(mia_score_list)
    target_memberships = np.concatenate(membership_list)

    attack_result = compute_attack_results(mia_scores, target_memberships)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Average result: AUC %.4f, TPR@0.1%%FPR of %.4f, TPR@0.0%%FPR of %.4f",
        attack_result["auc"],
        attack_result["one_tenth_fpr"],
        attack_result["zero_fpr"],
    )

    plot_roc(
        attack_result["fpr"],
        attack_result["tpr"],
        attack_result["auc"],
        f"{report_dir}/ROC_average.png",
    )
    plot_roc_log(
        attack_result["fpr"],
        attack_result["tpr"],
        attack_result["auc"],
        f"{report_dir}/ROC_log_average.png",
    )

    np.savez(
        f"{report_dir}/attack_result_average",
        fpr=attack_result["fpr"],
        tpr=attack_result["tpr"],
        auc=attack_result["auc"],
        one_tenth_fpr=attack_result["one_tenth_fpr"],
        zero_fpr=attack_result["zero_fpr"],
        scores=mia_scores.ravel(),
        memberships=target_memberships.ravel(),
    )


def audit_models(
    report_dir,
    target_model_indices,
    all_signals,
    all_memberships,
    num_reference_models,
    logger,
    configs,
):
    """
    Audit target model(s) using a Membership Inference Attack algorithm.

    Args:
        report_dir (str): Folder to save attack result.
        target_model_indices (list): List of the target model indices.
        all_signals (np.array): Signal value of all samples in all models (target and reference models).
        all_memberships (np.array): Membership matrix for all models.
        num_reference_models (int): Number of reference models used for performing the attack.
        logger (logging.Logger): Logger object for the current run.
        configs (dict): Configs provided by the user.

    Returns:
        list: List of MIA score arrays for all audited target models.
        list: List of membership labels for all target models.
    """
    all_memberships = np.transpose(all_memberships)

    mia_score_list = []
    membership_list = []

    for target_model_idx in target_model_indices:
        baseline_time = time.time()
        if configs["audit"]["algorithm"] == "RMIA":
            offline_a = tune_offline_a(
                target_model_idx, all_signals, all_memberships, logger
            )[0]
            logger.info(f"The best offline_a is %0.1f", offline_a)
            mia_scores = run_rmia(
                target_model_idx,
                all_signals,
                all_memberships,
                num_reference_models,
                offline_a,
            )
        elif configs["audit"]["algorithm"] == "LOSS":
            mia_scores = run_loss(all_signals[:, target_model_idx])
        else:
            raise NotImplementedError(
                f"{configs['audit']['algorithm']} is not implemented"
            )

        target_memberships = all_memberships[:, target_model_idx]

        mia_score_list.append(mia_scores.copy())
        membership_list.append(target_memberships.copy())

        _ = get_audit_results(
            report_dir, target_model_idx, mia_scores, target_memberships, logger
        )

        logger.info(
            "Auditing the privacy risks of target model %d costs %0.1f seconds",
            target_model_idx,
            time.time() - baseline_time,
        )

    return mia_score_list, membership_list


def audit_models_range(
    report_dir,
    target_model_indices,
    all_signals,
    all_memberships,
    num_reference_models,
    logger,
    configs,
):
    """
    Audit target model(s) using a Membership Inference Attack algorithm.

    Args:
        report_dir (str): Folder to save attack result.
        target_model_indices (list): List of the target model indices.
        all_signals (np.array): Signal value of all samples in all models (target and reference models).
        all_memberships (np.array): Membership matrix for all models.
        num_reference_models (int): Number of reference models used for performing the attack.
        logger (logging.Logger): Logger object for the current run.
        configs (dict): Configs provided by the user.

    Returns:
        list: List of MIA score arrays for all audited target models.
        list: List of membership labels for all target models.
    """
    all_memberships = np.transpose(all_memberships)

    if configs["ramia"].get("trim_ratio", None) is not None:
        if configs["ramia"].get("trim_direction", None) is None:
            raise ValueError("Need to specify trim_direction!")
        else:
            tune_trim_ratio = False
            trim_ratio = configs["ramia"]["trim_ratio"]
            trim_direction = configs["ramia"]["trim_direction"]
    else:
        tune_trim_ratio = True

    sample_size = configs["ramia"]["sample_size"]

    mia_score_list = []
    membership_list = []

    for target_model_idx in target_model_indices:
        baseline_time = time.time()

        if configs["audit"]["algorithm"] == "RMIA":
            offline_a, ref_mia_scores, ref_membership = tune_offline_a(
                target_model_idx, all_signals, all_memberships, logger
            )
            logger.info(f"The best offline_a is %0.1f", offline_a)
            mia_scores = run_rmia(
                target_model_idx,
                all_signals,
                all_memberships,
                num_reference_models,
                offline_a,
            )
        else:
            raise NotImplementedError(
                f"{configs['audit']['algorithm']} is not implemented for RaMIA"
            )

        if tune_trim_ratio:
            ref_mia_scores = ref_mia_scores.reshape(-1, sample_size)
            ref_membership = ref_membership.reshape(-1, sample_size)[:, 0]
            max_auc = 0

            logger.info(
                "Finding the optimal trim ratio and direction using the paired model"
            )

            for k in range(1, sample_size + 1):
                fpr, tpr, _ = roc_curve(
                    ref_membership, get_bottomk(ref_mia_scores, k).mean(1)
                )
                roc_auc = auc(fpr, tpr)
                if roc_auc > max_auc:
                    max_auc = roc_auc
                    trim_ratio = k / sample_size
                    trim_direction = "top"

                fpr, tpr, _ = roc_curve(
                    ref_membership, get_topk(ref_mia_scores, k).mean(1)
                )
                roc_auc = auc(fpr, tpr)
                if roc_auc > max_auc:
                    max_auc = roc_auc
                    trim_ratio = 1 - k / sample_size
                    trim_direction = "bottom"
            logger.info(
                "The optimal trim ratio is %.2f and the direction is %s",
                trim_ratio,
                trim_direction,
            )

        target_memberships = all_memberships[:, target_model_idx]

        mia_score_list.append(
            trim_mia_scores(
                mia_scores.copy().reshape(-1, sample_size), trim_ratio, trim_direction
            )
        )
        membership_list.append(target_memberships.copy().reshape(-1, sample_size)[:, 0])

        _ = get_audit_results(
            report_dir, target_model_idx, mia_scores, target_memberships, logger
        )

        logger.info(
            "Auditing the privacy risks of target model %d costs %0.1f seconds",
            target_model_idx,
            time.time() - baseline_time,
        )

    return mia_score_list, membership_list


def sample_auditing_dataset(
    configs, dataset: torch.utils.data.Dataset, logger, memberships: np.ndarray
):
    """
    Downsample the dataset in auditing if specified.

    Args:
        configs (Dict[str, Any]): Configuration dictionary
        dataset (Any): The full dataset from which the audit subset will be sampled.
        logger (Any): Logger object used to log information during downsampling.
        memberships (np.ndarray): A 2D boolean numpy array where each row corresponds to a model and
                                  each column corresponds to whether the corresponding sample is a member (True)
                                  or non-member (False).

    Returns:
        Tuple[torch.utils.data.Subset, np.ndarray]: A tuple containing:
            - The downsampled dataset or the full dataset if downsampling is not applied.
            - The corresponding membership labels for the samples in the downsampled dataset.

    Raises:
        ValueError: If the requested audit data size is larger than the full dataset or not an even number.
    """
    if configs["run"]["num_experiments"] > 1:
        logger.warning(
            "Auditing multiple models. Balanced downsampling is only based on the data membership of the FIRST target model!"
        )

    audit_data_size = configs["audit"].get("data_size", len(dataset))
    if audit_data_size < len(dataset):
        if audit_data_size % 2 != 0:
            raise ValueError("Audit data size must be an even number.")

        logger.info(
            "Downsampling the dataset for auditing to %d samples. The numbers of members and non-members are only "
            "guaranteed to be equal for the first target model, if more than one are used.",
            audit_data_size,
        )
        # Sample equal numbers of members and non-members according to the first target model randomly
        members_idx = np.random.choice(
            np.where(memberships[0, :])[0], audit_data_size // 2, replace=False
        )
        non_members_idx = np.random.choice(
            np.where(~memberships[0, :])[0], audit_data_size // 2, replace=False
        )

        # Randomly sample members and non-members
        auditing_dataset = Subset(
            dataset, np.concatenate([members_idx, non_members_idx])
        )
        auditing_membership = memberships[
            :, np.concatenate([members_idx, non_members_idx])
        ].reshape((memberships.shape[0], audit_data_size))
    elif audit_data_size == len(dataset):
        auditing_dataset = dataset
        auditing_membership = memberships
    else:
        raise ValueError("Audit data size cannot be larger than the dataset.")
    return auditing_dataset, auditing_membership
