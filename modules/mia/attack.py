from typing import Any, Tuple, Dict, Optional
import logging
import numpy as np
from .attacks import run_rmia, tune_offline_a


class MIA:
    def __init__(self, logger: logging.Logger):
        """
        Initialize the MIA object with configuration storage and a logger.

        Args:
            args (Dict[str, Any]): Configuration storage for existing MIA attacks to avoid retuning.
            logger (logging.Logger): Logger instance for logging information.
        """
        self.configs: Dict[frozenset, Dict[str, Any]] = {}
        self.logger = logger

    def _serialize_args(self, args: Dict[str, Any]) -> frozenset:
        """
        Serialize the args dictionary into a frozenset to use as a hashable key.

        Args:
            args (Dict[str, Any]): The configuration arguments.

        Returns:
            frozenset: A hashable representation of the args.
        """
        keys_to_extract = [
            "attack",
            "dataset",
            "model",
        ]  # Keys to include in the frozenset
        extracted_items = [(key, args[key]) for key in keys_to_extract if key in args]
        return frozenset(extracted_items)

    def run_mia(
        self,
        all_signals: np.ndarray,
        all_memberships: np.ndarray,
        target_model_idx: int,
        reference_model_indices: np.ndarray,
        logger: logging.Logger,
        args: Dict[str, Any],
        population_signals: Optional[np.ndarray],
        reuse_offline_a: Optional[bool] = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Launch a membership inference attack on a target model and output the MIA scores.

        Args:
            all_signals (np.ndarray): Softmax value of all samples in all models (target and reference models). Shape: (num_samples * num_models)
            all_memberships (np.ndarray): Membership matrix for all models. Shape: (num_samples * num_models)
            target_model_idx (int): Index of the target model.
            reference_model_indices (np.ndarray): List of indices of reference models.
            logger (logging.Logger): Logger instance for logging information.
            args (Dict[str, Any]): Arguments for the MIA attack.
            population_signals (Optional[np.ndarray]): Softmax value of population samples in all models (target and reference models). Shape: (num_population_samples * num_models)
            reuse_offline_a (Optional[bool]): Whether to reuse the offline_a value if it has been computed before. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: MIA scores for all samples in the target model and the membership labels for the target model.
        """
        assert (
            all_signals.shape == all_memberships.shape
        ), f"all_signals or all_memberships has incorrect shape (num_samples * num_models): {all_signals.shape} vs {all_memberships.shape}"
        target_signals = all_signals[:, target_model_idx]
        target_memberships = all_memberships[:, target_model_idx]

        ref_signals = all_signals[:, reference_model_indices]
        ref_memberships = all_memberships[:, reference_model_indices]

        z_target_signals = population_signals[:, target_model_idx]
        z_ref_signals = population_signals[:, reference_model_indices]

        logger.info(f"Args for MIA attack: {args}")
        if args["attack"] == "RMIA":
            # population_signals are required
            assert (
                population_signals is not None
            ), "population_signals is required for RMIA attack"
            if args.get("offline_a") is None:
                serialized_key = self._serialize_args(args)
                # Reuse the offline_a if it has been computed before for the same dataset and architecture
                if reuse_offline_a and serialized_key in self.configs:
                    offline_a = self.configs[serialized_key]["offline_a"]
                    logger.info(f"Using cached offline_a={offline_a}")
                else:
                    offline_a, _, _ = tune_offline_a(
                        target_model_idx,
                        all_signals,
                        all_memberships,
                        ref_signals,
                        ref_memberships,
                        z_target_signals,
                        z_ref_signals,
                        logger,
                    )
                    # offline_a = args.get("offline_a", 0.3)
                    args["offline_a"] = offline_a
                    self.configs[serialized_key] = args
            else:
                offline_a = args["offline_a"]
            logger.info(
                f"Running RMIA attack on target model {target_model_idx} with offline_a={offline_a}"
            )
            mia_scores = run_rmia(
                target_signals,
                ref_signals,
                ref_memberships,
                z_target_signals,
                z_ref_signals,
                offline_a,
            )
        else:
            raise ValueError(
                f"Attack type {args['attack']} is not supported. Please load your own attack scores."
            )

        return mia_scores, target_memberships
