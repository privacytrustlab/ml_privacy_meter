from typing import Tuple, List, Any, Dict
import numpy as np
from sklearn.metrics import roc_curve
import logging
import time

from modules.mia import MIA

class DUCI:
    def __init__(self, logger: logging.Logger, args: Dict[str, Any]):
        """
        Initialize the DUCI object.

        Args:
            num_reference_models (int): Number of reference models to use. 
                In DUCI, the real num_reference_models is the number of reference models used. 
                However, in this implementation, we follow the setting of MIA, i.e., for each
                sample, num_reference_models reference model is used. Therefore, whne the models are trained
                with half-half split, the num_reference_models is set to half of the number of reference models.
        """
        self.best_threshold = None
        self.tpr = None
        self.fpr = None
        self.logger = logger
        self.args = args

    def debias_pred(self, 
            target_model_idx: int,
            reference_model_indices: np.ndarray,
            all_signals: np.ndarray,
            all_memberships: np.ndarray,
            MIA_instance: MIA
        ) -> Tuple[float, float]:
        """
        This functions debiases the MIA signals over the target dataset received on the target model 
        through reusing the MIA signals and membership flags over the dataset of reference models. Then, 
        by aggregating the debiased signals, we can make an unbiased prediction to the proportion of 
        dataset being used in the target model.

        The proportion (\hat{p}) is computed as the aggregation of individual estimates (\hat{p}_i) 
        over all data points in the target dataset (X):
        
            \hat{p} = (1 / |X|) * sum_{i=1}^{|X|}(\hat{p}_i)

        Each individual estimate (\hat{p}_i) is calculated as:

            \hat{p}_i = (\hat{m}_i - P(\hat{m}_i = 1 | m_i = 0)) / 
                        (P(\hat{m}_i = 1 | m_i = 1) - P(\hat{m}_i = 1 | m_i = 0))
        
        where:
            - \hat{m}_i: MIA signal for the i-th data point in the target dataset.
            - P(\hat{m}_i = 1 | m_i = 0): False positive rate (FPR) derived from reference models.
            - P(\hat{m}_i = 1 | m_i = 1): True positive rate (TPR) derived from reference models.
            

        Args:
            target_model_idx (int): Index of the target model.
            mia_scores (np.ndarray): MIA scores for the target model.
            all_signals (np.ndarray): List of signal matrix with shape (samples, models).
            all_memberships (np.ndarray): List of membership matrix with shape (samples, models).

        Returns:
            dict: A dictionary containing debiased predictions, TPR, FPR, and absolute errors.
        """
        # Conduct MIA on the target model using privacy meter tools
        baseline_time = time.time()
        mia_scores, target_memberships = MIA_instance.run_mia(
            all_signals, 
            all_memberships, 
            target_model_idx, 
            reference_model_indices, 
            self.logger, 
            self.args,
            reuse_offline_a=False
        )
        self.logger.info("Collect membership prediction for target dataset on target model %d costs %0.1f seconds",
            target_model_idx, time.time() - baseline_time,
        )

        # Conduct MIA on the paired model using privacy meter tools
        ref_score_all, ref_membership_all = [], []
        paired_model_idx = (
            target_model_idx + 1 if target_model_idx % 2 == 0 else target_model_idx - 1
        )
        for ref_model_idx in [paired_model_idx]: #TODO: add population data in RMIA for reference model debiasing
        # for ref_model_idx in reference_model_indices:
            ref_mia_scores, ref_target_memberships = MIA_instance.run_mia(
                all_signals,
                all_memberships,
                ref_model_idx,
                reference_model_indices,
                self.logger,
                self.args,
                reuse_offline_a=False
            )
            ref_score_all.append(ref_mia_scores)
            ref_membership_all.append(ref_target_memberships)
        debias_memberships = np.array(ref_membership_all)
        debias_scores = np.array(ref_score_all)

        # # plot scores distribution
        # import matplotlib.pyplot as plt
        # member_scores = debias_scores.ravel()[debias_memberships.ravel() == 1]
        # non_member_scores = debias_scores.ravel()[debias_memberships.ravel() == 0]
        # plt.hist(member_scores, bins=50, alpha=0.5, label='ref Members')
        # plt.hist(non_member_scores, bins=50, alpha=0.5, label='ref Non-members')
        

        # member_scores = mia_scores.ravel()[target_memberships.ravel() == 1]
        # non_member_scores = mia_scores.ravel()[target_memberships.ravel() == 0]
        # plt.hist(member_scores, bins=50, alpha=0.5, label='Members')
        # plt.hist(non_member_scores, bins=50, alpha=0.5, label='Non-members')

        # plt.legend(loc='upper right')
        # plt.savefig('ref_scores_distribution.png')

        # Find the optimal threshold
        fpr_list, tpr_list, thresholds = roc_curve(debias_memberships.ravel(), debias_scores.ravel())
        best_idx = np.argmax(tpr_list - fpr_list)
        self.best_threshold = thresholds[best_idx]
        self.tpr = tpr_list[best_idx]
        self.fpr = fpr_list[best_idx]

        self.logger.info(f"Best threshold = {self.best_threshold} (Maximize TPR - FPR) = {self.tpr} - {self.fpr}")

        # Apply threshold to target model scores
        preds = mia_scores > self.best_threshold

        true_tpr = np.sum(preds * target_memberships) / np.sum(target_memberships)
        true_fpr = np.sum(preds * ~target_memberships) / np.sum(~target_memberships)
        true_proportion = np.mean(target_memberships)
        self.logger.info(f"True proportion {true_proportion}, True TPR: {true_tpr}, True FPR: {true_fpr}")

        # Compute debiased predictions
        debiased_pres = (preds - self.fpr)/(self.tpr - self.fpr)
        self.logger.info(f"DUCI prediction: {np.mean(debiased_pres)}; Direct Aggregation: {np.mean(preds)}")

        return debiased_pres, true_proportion

    def pred_proportions(
        self, 
        target_model_indices: List[int],
        reference_model_indices_all: List[np.ndarray],
        all_signals: np.ndarray,
        all_memberships: np.ndarray,
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Launch Membership Inference Attacks on target dataset over ALL target models using Privacy Meter tool.

        Args:
            target_model_indices (list): List of the target model indices.
            reference_model_indices (list): List of the reference model indices array.
            all_signals (np.ndarray): Softmax value of all samples in all models (target and reference models). Shape: (num_samples * num_models)
            all_memberships (np.ndarray): Membership matrix for all models. Shape: (num_samples * num_models)
        
        Returns:
            Tuple[List[float], List[float], List[float]]: A tuple containing debiased predictions, true proportions, and errors.
        """
        assert all_signals.shape == all_memberships.shape, f"all_signals or all_memberships has incorrect shape (num_samples * num_models): {all_signals.shape} vs {all_memberships.shape}"

        debiased_preds_list, true_proportion_list = [], []
        error_list = []
        # Initialize MIA instance
        MIA_instance = MIA(self.logger)
        # for target_model_idx in target_model_indices:
        for target_model_idx, reference_model_indices in zip(target_model_indices, reference_model_indices_all):
            debiased_pres, true_proportion = self.debias_pred(
                target_model_idx,
                reference_model_indices,
                all_signals,
                all_memberships,
                MIA_instance
            )

            post_debias_error = np.abs(np.mean(debiased_pres) - true_proportion)
            self.logger.info(
                r"Absolute Error $| \hat{p} - p|$: Debiased Agg MIA = %.4f",
                post_debias_error
            )

            debiased_preds_list.append(np.mean(debiased_pres))
            true_proportion_list.append(true_proportion)
            error_list.append(post_debias_error)

        return debiased_preds_list, true_proportion_list, error_list
    
    def eval():
        pass