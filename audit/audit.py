import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_curve  # type: ignore
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm

from attacks import RMIA
from dataset import load_dataset_subsets
from utils import timer
from visualize import plot_roc, plot_roc_log  # type: ignore


class AuditingEngine:
    configs: dict[str, Any]
    logger: logging.Logger
    attack: RMIA

    auditing_dataset: Dataset[Any]
    auditing_membership: npt.NDArray[Any]
    signals: dict[str, list[npt.NDArray[np.float_]]]


    def __init__(self,
                 configs: dict[str, Any],
                 directories: dict[str, str],
                 log_dir: str,
                 logger: logging.Logger) -> None:
        self.configs = configs
        self.directories = directories
        self.log_dir = log_dir
        self.logger = logger
        self.attack = RMIA(configs, logger)

        self.signals = {}

    def sample_auditing_dataset(
        self,
        dataset: Dataset[Any],
        memberships: npt.NDArray[Any]
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
        if self.configs["run"]["num_experiments"] > 1:
            self.logger.warning(
                "Auditing multiple models. Balanced downsampling is only based on the data membership of the FIRST target model!"
            )

        audit_data_size = self.configs["audit"].get("data_size", len(dataset)) #type:ignore
        if audit_data_size < len(dataset): #type:ignore
            if audit_data_size % 2 != 0:
                raise ValueError("Audit data size must be an even number.")

            self.logger.info(
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
                dataset, np.concatenate([members_idx, non_members_idx]) #type:ignore
            )
            auditing_membership = memberships[
                :, np.concatenate([members_idx, non_members_idx])
            ].reshape((memberships.shape[0], audit_data_size))
        elif audit_data_size == len(dataset): #type:ignore
            auditing_dataset = dataset
            auditing_membership = memberships
        else:
            raise ValueError("Audit data size cannot be larger than the dataset.")

        self.auditing_dataset = auditing_dataset
        self.auditing_membership = auditing_membership

        return self

    def get_model_signals(self,
        models_list: list[nn.Module],
        population: Dataset[Any]
    ):
        population = Subset(
            population,
            np.random.choice( #type:ignore
                len(population), #type:ignore
                self.configs["audit"].get("population_size", len(population)), #type:ignore
                replace=False,
            ),
        )

        # Generate signals (softmax outputs) for all models
        with timer(self.logger, "Preparing signals"):
            self.__get_model_signals(models_list, self.auditing_dataset)
            self.__get_model_signals(models_list, population, is_population=True)

    def __get_softmax(
        self,
        model: nn.Module,
        samples: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        device: str,
        temp: float = 1.0,
    ) -> npt.NDArray[np.float_]:
        """
        Get the model's softmax probabilities for the given inputs and expected outputs.

        Args:
            model (PreTrainedModel or torch.nn.Module): Model instance.
            samples (torch.Tensor): Model input.
            labels (torch.Tensor): Model expected output.
            batch_size (int): Batch size for getting signals.
            device (str): Device used for computing signals.
            temp (float): Temperature used in softmax computation.
            pad_token_id (Optional[int]): Padding token ID to ignore in aggregation.

        Returns:
            all_softmax_list (np.array): softmax value of all samples
        """
        model.to(device)  #type:ignore
        model.eval()
        with torch.no_grad():
            softmax_list: list[torch.Tensor] = []
            batched_samples = torch.split(samples, batch_size)
            batched_labels = torch.split(labels, batch_size)

            for x, y in tqdm(
                zip(batched_samples, batched_labels),
                total=len(batched_samples),
                desc="Computing softmax",
            ):
                x = x.to(device)
                y = y.to(device)

                pred = model(x)

                logit_signals = torch.div(pred, temp)
                max_logit_signals, _ = torch.max(logit_signals, dim=1)
                # This is to avoid overflow when exp(logit_signals)
                logit_signals = torch.sub(
                    logit_signals, max_logit_signals.reshape(-1, 1)
                )
                exp_logit_signals = torch.exp(logit_signals)
                exp_logit_sum = exp_logit_signals.sum(dim=1).reshape(-1, 1)
                true_exp_logit = exp_logit_signals.gather(1, y.reshape(-1, 1))
                softmax_list.append(torch.div(true_exp_logit, exp_logit_sum).to("cpu"))

            all_softmax_list = np.concatenate(softmax_list) #type:ignore
        model.to("cpu")
        return all_softmax_list #type:ignore


    def __get_model_signals(
        self,
        models_list: list[nn.Module],
        dataset: Dataset[Any],
        is_population: bool=False
    ):
        # Check if signals are available on disk
        signal_file_name = f"{self.configs['audit']['algorithm'].lower()}_signals"
        signal_file_name += "_pop.npy" if is_population else ".npy"
        if os.path.exists(
            f"{self.configs['run']['log_dir']}/signals/{signal_file_name}",
        ):
            signals = np.load(
                f"{self.configs['run']['log_dir']}/signals/{signal_file_name}",
            )
            expected_size = len(dataset) #type:ignore
            signal_source = "training data size"

            if signals.shape[0] == expected_size: #type:ignore
                self.logger.info("Signals loaded from disk successfully.")
                self.signals["signal" if not is_population else "population_signal"] = signals
                return
            else:
                self.logger.warning(
                    f"Signals shape ({signals.shape[0]}) does not match the expected size ({expected_size}). " #type:ignore
                    f"This mismatch is likely due to a change in the {signal_source}."
                )
                self.logger.info("Ignoring the signals on disk and recomputing.")

        batch_size = self.configs["audit"]["batch_size"]  # Batch size used for inferring signals
        model_name = self.configs["train"]["model_name"]  # Algorithm used for training models
        device = self.configs["audit"]["device"]  # GPU device used for inferring signals


        dataset_samples = np.arange(len(dataset)) #type:ignore
        data, targets = load_dataset_subsets(
            dataset, dataset_samples, model_name, batch_size, device #type:ignore
        )

        signals: list[npt.NDArray[np.float_]] = []
        self.logger.info("Computing signals for all models.")

        for model in models_list:
            signals.append(
                self.__get_softmax(
                    model, data, targets, batch_size, device
                )
            )

        signals = np.concatenate(signals, axis=1) #type:ignore
        os.makedirs(f"{self.configs['run']['log_dir']}/signals", exist_ok=True)
        np.save(
            f"{self.configs['run']['log_dir']}/signals/{signal_file_name}",
            signals,
        )
        self.logger.info("Signals saved to disk.")

        self.signals["signal" if not is_population else "population_signal"] = signals

        return self

    def audit(self):
        # Perform the privacy audit
        with timer(self.logger, "Auditing privacy risk"):
            self.mia_score_list, self.membership_list = self.audit_models(
                f"{self.directories['report_dir']}/exp",
                list(range(self.configs["run"]["num_experiments"])),
                self.signals["signal"], #type:ignore
                self.signals["population_signal"], #type:ignore
                self.auditing_membership,
                self.configs["audit"]["num_ref_models"]
            )
        return self

    def audit_models(
        self,
        report_dir: str,
        target_model_indices: list[int],
        all_signals: npt.NDArray[np.float_],
        population_signals: npt.NDArray[Any],
        all_memberships: npt.NDArray[Any],
        num_reference_models: int
    ):
        """
        Audit target model(s) using a Membership Inference Attack algorithm.

        Args:
            report_dir (str): Folder to save attack result.
            target_model_indices (list): List of the target model indices.
            all_signals (np.array): Signal value of all samples in all models (target and reference models).
            population_signals (np.array): Signal value of all population data in all models (target and reference models).
            all_memberships (np.array): Membership matrix for all models.
            num_reference_models (int): Number of reference models used for performing the attack.

        Returns:
            list: List of MIA score arrays for all audited target models.
            list: List of membership labels for all target models.
        """

        all_memberships = np.transpose(all_memberships)

        mia_score_list: list[npt.NDArray[np.float_]] = []
        membership_list: list[npt.NDArray[np.float_]] = []

        for target_model_idx in target_model_indices:
            with timer(self.logger, f"Auditing the privacy risks of target model {target_model_idx}"):
                mia_scores = self.attack(target_model_idx,
                        all_signals,
                        population_signals,
                        all_memberships,
                        num_reference_models)

                target_memberships = all_memberships[:, target_model_idx]

                mia_score_list.append(mia_scores.copy())
                membership_list.append(target_memberships.copy())

                get_audit_results(
                    report_dir, target_model_idx, mia_scores, target_memberships, self.logger
                )


        return mia_score_list, membership_list



def compute_attack_results(
    mia_scores: npt.NDArray[np.float_],
    target_memberships: npt.NDArray[np.float_]) -> dict[str, Any]:
    """
    Compute attack results (TPR-FPR curve, AUC, etc.) based on MIA scores and membership of samples.

    Args:
        mia_scores (np.array): MIA score computed by the attack.
        target_memberships (np.array): Membership of samples in the training set of target model.

    Returns:
        dict: Dictionary of results, including fpr and tpr list, AUC, TPR at 1%, 0.1% and 0% FPR.
    """
    fpr_list, tpr_list, _ = roc_curve(target_memberships.ravel(), mia_scores.ravel()) #type:ignore
    roc_auc = auc(fpr_list, tpr_list) #type:ignore
    one_fpr = tpr_list[np.where(fpr_list <= 0.01)[0][-1]] #type:ignore
    one_tenth_fpr = tpr_list[np.where(fpr_list <= 0.001)[0][-1]] #type:ignore
    zero_fpr = tpr_list[np.where(fpr_list <= 0.0)[0][-1]] #type:ignore

    return {
        "fpr": fpr_list,
        "tpr": tpr_list,
        "auc": roc_auc,
        "one_fpr": one_fpr,
        "one_tenth_fpr": one_tenth_fpr,
        "zero_fpr": zero_fpr,
    }

def get_audit_results(
    report_dir: str,
    model_idx: int,
    mia_scores: npt.NDArray[np.float_],
    target_memberships: npt.NDArray[np.float_],
    logger: logging.Logger):
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

def get_average_audit_results(
    report_dir: str,
    mia_score_list: list[npt.NDArray[np.float_]],
    membership_list: list[npt.NDArray[np.float_]],
    logger: logging.Logger):
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




