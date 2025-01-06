import time
from pathlib import Path


import math
import scipy


import numpy as np
import torch.utils.data
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Subset

from attacks import tune_offline_a, run_rmia, run_loss
from ramia_scores import get_topk, get_bottomk, trim_mia_scores
from visualize import plot_roc, plot_roc_log, plot_eps_vs_num_guesses


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



# below are tools for DP auditing



def compute_abstain_attack_results(mia_scores, target_memberships, delta=0, p_value=0.05):
    """
    Compute attack results (TPR-FPR curve, AUC, etc.) based on MIA scores and membership of samples.

    Args:
        mia_scores (np.array): MIA score computed by the attack.
        target_memberships (np.array): Membership of samples in the training set of target model.

    Returns:
        dict: Dictionary of results, including fpr and tpr list, AUC, TPR at 1%, 0.1% and 0% FPR.
    """
    mia_scores = mia_scores.ravel()
    target_memberships = target_memberships.ravel()
    sorted_idx = np.argsort(mia_scores)
    mia_scores = mia_scores[sorted_idx]
    target_memberships = target_memberships[sorted_idx]
    step_size = int(np.sqrt(len(target_memberships.ravel())))
    assert(step_size>=1)
    k_neg_k_pos_list = sum([[(k_neg, k_pos) for k_neg in range(0, k_pos, step_size)] for k_pos in range(0, len(target_memberships.ravel()), step_size)], [])
    correct_num_list = [(1-target_memberships[:k_neg]).sum() +  target_memberships[k_pos:].sum() for (k_neg, k_pos) in k_neg_k_pos_list]
    eps_list = [get_eps_audit(len(target_memberships), k_neg + len(target_memberships) - k_pos, correct_num, delta, p_value) for ((k_neg, k_pos), correct_num) in zip(k_neg_k_pos_list, correct_num_list)]
    k_neg_k_pos_idx = np.argmax(eps_list)
    (k_neg_opt, k_pos_opt) = k_neg_k_pos_list[k_neg_k_pos_idx]
    eps_opt = eps_list[k_neg_k_pos_idx]
    correct_num_opt = correct_num_list[k_neg_k_pos_idx]
    

    return {
        "k_neg": [k_neg_k_pos_list[i][0] for i in range(len(k_neg_k_pos_list))],
        "k_pos": [k_neg_k_pos_list[i][1] for i in range(len(k_neg_k_pos_list))],
        "eps": eps_list,
        "correct_num": correct_num_list,
        "eps_opt": eps_opt,
        "k_neg_opt": k_neg_opt,
        "k_pos_opt": k_pos_opt,
        "correct_num_opt": correct_num_opt,
        "total_num": len(target_memberships),
        "delta": delta,
        "p_value": p_value,
    }


def compute_abstain_attack_results_for_k_pos_k_neg(mia_scores, target_memberships, k_pos, k_neg, delta=0, p_value=0.05):
    """
    Compute attack results (TPR-FPR curve, AUC, etc.) based on MIA scores and membership of samples.

    Args:
        mia_scores (np.array): MIA score computed by the attack.
        target_memberships (np.array): Membership of samples in the training set of target model.

    Returns:
        dict: Dictionary of results, including fpr and tpr list, AUC, TPR at 1%, 0.1% and 0% FPR.
    """
    mia_scores = mia_scores.ravel()
    target_memberships = target_memberships.ravel()
    sorted_idx = np.argsort(mia_scores)
    mia_scores = mia_scores[sorted_idx]
    target_memberships = target_memberships[sorted_idx]
    correct_num  = (1-target_memberships[:k_neg]).sum() +  target_memberships[k_pos:].sum() 
    eps = get_eps_audit(len(target_memberships), k_neg + len(target_memberships) - k_pos, correct_num, delta, p_value)

    return {
        "k_neg": k_neg,
        "k_pos": k_pos,
        "correct_num": correct_num,
        "eps": eps,
        "total_num": len(target_memberships),
        "delta": delta,
        "p_value": p_value,
    }

def get_all_dp_audit_results(report_dir, mia_score_list, membership_list, logger):
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

    attack_dp_result = compute_abstain_attack_results(mia_scores, target_memberships)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Best One Run DP Auditing Results: EPS Lower Bound %.4f under DELTA %.2e and P_VALUE %.2f (%d correct out of %d guesses, k_neg=%d and k_pos=%d",
        attack_dp_result["eps_opt"],
        attack_dp_result["delta"],
        attack_dp_result["p_value"],
        attack_dp_result["correct_num_opt"],
        len(target_memberships[:attack_dp_result["k_neg_opt"]]) + len(target_memberships[attack_dp_result["k_pos_opt"]:]),
        attack_dp_result["k_neg_opt"],
        attack_dp_result["k_pos_opt"]
    )

    plot_eps_vs_num_guesses(
        attack_dp_result["eps"],
        attack_dp_result["correct_num"],
        attack_dp_result["k_neg"],
        attack_dp_result["k_pos"],
        attack_dp_result["total_num"],
        f"{report_dir}/dp_audit_average.png",
    )

    np.savez(
        f"{report_dir}/attack_result_average_dp",
        eps=attack_dp_result["eps"],
        correct_num=attack_dp_result["correct_num"],
        k_neg=attack_dp_result["k_neg"],
        k_pos=attack_dp_result["k_pos"],
        total_num=attack_dp_result["total_num"],
        scores=mia_scores.ravel(),
        memberships=target_memberships.ravel(),
    )



def get_dp_audit_results_for_k_pos_k_neg(report_dir, mia_score_list, membership_list, logger, k_pos, k_neg):
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

    attack_dp_result = compute_abstain_attack_results_for_k_pos_k_neg(mia_scores, target_memberships, k_pos, k_neg)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "One Run DP Auditing Results: EPS Lower Bound %.4f under DELTA %.2e and P_VALUE %.2f (%d correct out of %d guesses)",
        attack_dp_result["eps"],
        attack_dp_result["delta"],
        attack_dp_result["p_value"],
        attack_dp_result["correct_num"],
        len(target_memberships[:attack_dp_result["k_neg"]]) + len(target_memberships[attack_dp_result["k_pos"]:])
    )

# Code snipplet taken from [Steinke, Thomas, Milad Nasr, and Matthew Jagielski. "Privacy auditing with one (1) training run." Advances in Neural Information Processing Systems 36 (2024).]
# m = number of examples, each included independently with probability 0.5
# r = number of guesses (i.e. excluding abstentions)
# v = number of correct guesses by auditor
# eps,delta = DP guarantee of null hypothesis
# output: p-value = probability of >=v correct guesses under null hypothesis
def p_value_DP_audit(m, r, v, eps, delta):
  assert 0 <= v <= r <= m
  assert eps >= 0
  assert 0 <= delta <= 1
  q = 1/(1+math.exp(-eps))  # accuracy of eps-DP randomized response
  beta = scipy.stats.binom.sf(v-1, r, q)  # = P[Binomial(r, q) >= v]
  if delta == 0:
    p = beta
  else:
    alpha = 0
    sum = 0  # = P[v > Binomial(r, q) >= v - i]
    for i in range(1, v + 1):
        sum = sum + scipy.stats.binom.pmf(v - i, r, q)
        if sum > i * alpha:
          alpha = sum / i
    p = beta + alpha * delta * 2 * m
  return min(p, 1)
# m = number of examples, each included independently with probability 0.5
# r = number of guesses (i.e. excluding abstentions)
# v = number of correct guesses by auditor
# p = 1-confidence e.g. p=0.05 corresponds to 95%
# output: lower bound on eps i.e. algorithm is not (eps,delta)-DP
def get_eps_audit(m, r, v, delta, p):
  m = int(m) 
  r = int(r)
  v = int(v)
  assert 0 <= v <= r <= m
  assert 0 <= delta <= 1
  assert 0 < p < 1
  eps_min = 0  # maintain p_value_DP(eps_min) < p
  eps_max = 1  # maintain p_value_DP(eps_max) >= p
  while p_value_DP_audit(m, r, v, eps_max, delta) < p: eps_max = eps_max + 1
  for _ in range(30):  # binary search
    if eps_max - eps_min <=1e-5:
      break
    eps = (eps_min + eps_max) / 2
    if p_value_DP_audit(m, r, v, eps, delta) < p:
      eps_min = eps
    else:
      eps_max = eps
  return eps_min


