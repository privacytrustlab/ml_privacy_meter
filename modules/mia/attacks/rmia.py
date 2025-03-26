from typing import Any, Optional
from sklearn.metrics import auc, roc_curve
import numpy as np


def get_out_ref_signals(
    ref_signals: np.ndarray,
    ref_memberships: np.ndarray,
    num_reference_models: Optional[int] = None,
    offline_a: float = 0.3,
) -> np.ndarray:
    """
    Get average prediction probability of samples over offline reference models (excluding the target model).

    Args:
        ref_signals (np.ndarray): Softmax value of all samples in all reference model.  Shape: (num_samples * num_models)
        ref_memberships (np.ndarray): Membership matrix for all reference models (if a sample is used for training a model).  Shape: (num_samples * num_models)
        num_reference_models (Optional[int]): Number of reference models used for the attack. Defaults to half reference models if None.
        offline_a (float): Coefficient offline_a is used to approximate p(x) using P_out in the offline setting.

    Returns:
        np.ndarray: Average softmax value for each sample over OUT reference models.
    """
    non_members = ~ref_memberships
    out_signals = ref_signals * non_members
    # Sort the signals such that only the non-zero signals (out signals) for each sample are kept
    if num_reference_models is None:
        num_reference_models = ref_signals.shape[1]
    if num_reference_models > 1:
        out_signals = -np.sort(-out_signals, axis=1)[:, :num_reference_models]
    else:
        # Derive according to ((1+a)P_out + (1-a))/2 = P(x) = (P_out + P_in)/2
        if offline_a != 0:
            out_signals += ((ref_signals + offline_a - 1) / offline_a) * ref_memberships
        else:
            out_signals += ((ref_signals - 0.7) / 0.3) * ref_memberships
    return out_signals


def tune_offline_a(
    target_model_idx: int,
    all_signals: np.ndarray,
    all_memberships: np.ndarray,
    ref_signals: np.ndarray,
    ref_memberships: np.ndarray,
    z_target_signals: np.ndarray,
    z_ref_signals: np.ndarray,
    logger: Any,
) -> float:
    """
    Fine-tune coefficient offline_a used in RMIA.

    Args:
        target_model_idx (int): Index of the target model.
        all_signals (np.ndarray): Softmax value of all samples in all models (target and reference).
        all_memberships (np.ndarray): Membership matrix for all models.
        ref_signals (np.ndarray): Softmax value of all samples in all reference models.
        ref_memberships (np.ndarray): Membership matrix for all reference models.
        z_target_signals (np.ndarray): Softmax value of population samples in the target model.
        z_ref_signals (np.ndarray): Softmax value of population samples in all reference models.
        logger (Any): Logger object for the current run.

    Returns:
        float: Optimized offline_a obtained by attacking a paired model with the help of the reference models.
    """
    paired_model_idx = (
        target_model_idx + 1 if target_model_idx % 2 == 0 else target_model_idx - 1
    )
    logger.info(f"Fine-tuning offline_a using paired model {paired_model_idx}")
    paired_memberships = all_memberships[:, paired_model_idx]
    offline_a = 0.0
    max_auc = 0
    for test_a in np.arange(0, 1.1, 0.1):
        paired_signals = all_signals[:, paired_model_idx]

        # launch RMIA using one pair of reference models on the paired model
        mia_scores = run_rmia(
            paired_signals,
            ref_signals,
            ref_memberships,
            z_target_signals,
            z_ref_signals,
            test_a,
            1,
        )

        fpr_list, tpr_list, _ = roc_curve(
            paired_memberships.ravel(), mia_scores.ravel()
        )
        roc_auc = auc(fpr_list, tpr_list)
        if roc_auc > max_auc:
            max_auc = roc_auc
            offline_a = test_a
            mia_scores_array = mia_scores.ravel().copy()
            membership_array = paired_memberships.ravel().copy()
        logger.info(f"offline_a={test_a:.2f}: AUC {roc_auc:.4f}")
    return offline_a, mia_scores_array, membership_array


def run_rmia(
    target_signals: np.ndarray,
    ref_signals: np.ndarray,
    ref_memberships: np.ndarray,
    z_target_signals: np.ndarray,
    z_ref_signals: np.ndarray,
    offline_a: float,
    num_reference_models: Optional[int] = None,
) -> np.ndarray:
    """
    Attack a target model using the RMIA attack with the help of offline reference models.

    Args:
        target_signals (np.ndarray): Softmax value of all samples in the target model.
        ref_signals (np.ndarray): Softmax value of all samples in the reference models.
        ref_memberships (np.ndarray): Membership matrix for all reference models.
        z_target_signals (np.ndarray): Softmax value of population samples in the target model.
        z_ref_signals (np.ndarray): Softmax value of population samples in all reference models.
        offline_a (float): Coefficient offline_a is used to approximate p(x) using P_out in the offline setting.
        num_reference_models (Optional[int]): Number of reference models used for the attack. Defaults to half reference models if None.

    Returns:
        np.ndarray: MIA score for all samples (a larger score indicates higher chance of being member).
    """
    assert len(ref_signals.shape) > 1, "ref_signals must be a 2D array"
    if num_reference_models is None:
        num_reference_models = max(ref_signals.shape[1] // 2, 1)
    out_signals = get_out_ref_signals(
        ref_signals, ref_memberships, num_reference_models, offline_a
    )
    mean_out_x = np.mean(out_signals, axis=1)
    mean_x = (1 + offline_a) / 2 * mean_out_x + (1 - offline_a) / 2
    prob_ratio_x = target_signals.ravel() / mean_x

    population_memberships = np.zeros_like(z_ref_signals).astype(
        bool
    )  # All population data are OUT for all models
    z_out_signals = get_out_ref_signals(
        z_ref_signals, population_memberships, num_reference_models, offline_a
    )
    mean_out_z = np.mean(z_out_signals, axis=1)
    mean_z = (1 + offline_a) / 2 * mean_out_z + (1 - offline_a) / 2
    prob_ratio_z = z_target_signals.ravel() / mean_z

    ratios = prob_ratio_x[:, np.newaxis] / prob_ratio_z
    counts = np.average(ratios > 1.0, axis=1)

    return counts
