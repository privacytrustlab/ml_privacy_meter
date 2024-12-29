from typing import Any, Tuple, Dict
import logging
import numpy as np
from .attacks import run_rmia, tune_offline_a

def run_mia(
    all_signals: np.ndarray,
    all_memberships: np.ndarray,
    target_model_idx: int,
    reference_model_indices: np.ndarray,
    logger: logging.Logger,
    args: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Launch a membership inference attack on a target model and output the MIA scores.
    
    Args:
        all_signals (np.ndarray): Softmax value of all samples in all models (target and reference models). Shape: (num_samples * num_models)
        all_memberships (np.ndarray): Membership matrix for all models. Shape: (num_samples * num_models)
        target_model_idx (int): Index of the target model.
        reference_model_indices (np.ndarray): List of indices of reference models.
        args (Dict[str, Any]): Arguments for the MIA attack.

    Returns:
        Tuple[np.ndarray, np.ndarray]: MIA scores for all samples in the target model and the membership labels for the target model.
    """
    assert all_signals.shape == all_memberships.shape, f"all_signals or all_memberships has incorrect shape (num_samples * num_models): {all_signals.shape} vs {all_memberships.shape}"
    target_signals = all_signals[:, target_model_idx]
    target_memberships = all_memberships[:, target_model_idx]

    ref_signals = all_signals[:, reference_model_indices]
    ref_memberships = all_memberships[:, reference_model_indices]
    
    logger.info(f"Args for MIA attack: {args}")
    if args["attack"] == "RMIA":
        if args.get("offline_a") is None:
            offline_a, _, _ = tune_offline_a(target_model_idx, all_signals, all_memberships, ref_signals, ref_memberships, logger)
            # offline_a = args.get("offline_a", 0.3)
        
        logger.info(f"Running RMIA attack on target model {target_model_idx} with offline_a={offline_a}")
        mia_scores = run_rmia(target_signals, ref_signals, ref_memberships, offline_a)
    else:
        raise ValueError(f"Attack type {args['attack']} is not supported. Please load your own attack scores.")  
    
    return mia_scores, target_memberships

  





