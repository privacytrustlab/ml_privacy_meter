import numpy as np
from sklearn.metrics import roc_curve
import logging
from attacks import tune_offline_a, run_rmia, run_loss
import time

logger = logging.getLogger(__name__)

class DUCI:
    def __init__(self, num_reference_models):
        """
        Initialize the DUCI object.

        Args:
            num_reference_models (int): Number of reference models to use.
            offline_a (float): The offline adjustment parameter.
        """
        self.num_reference_models = num_reference_models
        self.offline_a = None
        self.best_threshold = None
        self.tpr = None
        self.fpr = None

    def pred_proportion(self, target_model_indices, mia_score_list, membership_list, offline_a):
        """
        This functions takes in the MIA signals of the target dataset on the target model, 
        the MIA signals and membership flags over the dataset of reference models to 
        debias the MIA signals received on the target model. Then, by aggregating the
        debiased signals, we can make an unbiased prediction to the proportion of dataset 
        being used in the target model.

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
            mia_score_list (np.ndarray): List of signal matrix with shape (samples, models).
            membership_list (np.ndarray): List of membership matrix with shape (samples, models).

        Returns:
            dict: A dictionary containing debiased predictions, TPR, FPR, and absolute errors.
        """
        self.offline_a = offline_a
        pre_debias_errors, post_debias_errors = [], []
        for idx, target_model_idx in enumerate(target_model_indices):
            all_signals = mia_score_list[idx]
            all_memberships = membership_list[idx]
            # Identify paired model and reference indices
            paired_model_idx = target_model_idx + 1 if target_model_idx % 2 == 0 else target_model_idx - 1
            columns = [
                i for i in range(all_signals.shape[1])
                if i != target_model_idx and i != paired_model_idx
            ][:2 * self.num_reference_models]

            # Select and process reference signals
            selected_signals = all_signals[:, columns]
            non_members = ~all_memberships[:, columns]
            out_signals = selected_signals * non_members
            out_signals = -np.sort(-out_signals, axis=1)[:, :self.num_reference_models]

            # Compute mean values
            mean_out_x = np.mean(out_signals, axis=1)
            mean_x = (1 + self.offline_a) / 2 * mean_out_x + (1 - self.offline_a) / 2

            # De-bias using paired model
            ref_target_signals = all_signals[:, paired_model_idx]
            mia_scores_for_debiasing = ref_target_signals / mean_x

            debias_scores = np.array(mia_scores_for_debiasing)
            debias_memberships = all_memberships[:, paired_model_idx]

            # Find the optimal threshold
            fpr_list, tpr_list, thresholds = roc_curve(debias_memberships.ravel(), debias_scores.ravel())
            self.best_threshold = thresholds[np.argmax(tpr_list - fpr_list)]
            self.tpr = np.max(tpr_list)
            self.fpr = np.min(fpr_list)

            logger.info(f"Best threshold = {self.best_threshold} (Maximize TPR - FPR) = {self.tpr} - {self.fpr}")

            # Apply threshold to target model scores
            preds = (all_signals[:, target_model_idx] > self.best_threshold).ravel()
            target_memberships = all_memberships[:, target_model_idx]

            true_tpr = np.sum(preds * target_memberships.ravel()) / np.sum(target_memberships.ravel())
            true_fpr = np.sum(preds * ~target_memberships.ravel()) / np.sum(~target_memberships.ravel())

            logger.info(f"Direct aggregation of MIA preds: {np.mean(preds)}, TPR: {true_tpr}, FPR: {true_fpr}")

            # Compute debiased predictions
            debiased_pres = (preds - self.fpr)/(self.tpr - self.fpr)
            true_proportion = np.mean(target_memberships)
            logger.info(f"Debiased aggregation of MIA preds: {np.mean(debiased_pres)}")
            pre_debias_error = np.abs(np.mean(preds) - true_proportion)
            post_debias_error = np.abs(np.mean(debiased_pres) - true_proportion)
            logger.info(
                r"Absolute Error $| \hat{p} - p|$: Agg MIA = %.4f, Debiased Agg MIA = %.4f",
                pre_debias_error,
                post_debias_error
            )
            pre_debias_errors.append(pre_debias_error)
            post_debias_errors.append(post_debias_error)

        return pre_debias_errors, post_debias_errors
    
    def get_ind_signals(
        self, 
        target_model_indices,
        all_signals,
        all_memberships,
        num_reference_models,
        logger,
        configs,
    ):
        """
        Launch Membership Inference Attacks for each samples in the target dataset on target models using Privacy Meter tool.

        Args:
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
            # TODO: call the MIA module
            if configs["audit"]["algorithm"] == "RMIA":
                offline_a = tune_offline_a(
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
            #TODO: remove the loss
            elif configs["audit"]["algorithm"] == "LOSS":
                mia_scores = run_loss(all_signals[:, target_model_idx])
            else:
                raise NotImplementedError(
                    f"{configs['audit']['algorithm']} is not implemented"
                )

            target_memberships = all_memberships[:, target_model_idx]

            mia_score_list.append(mia_scores.copy())
            membership_list.append(target_memberships.copy())

            logger.info(
                "Collect membership prediction for target dataset on target model %d costs %0.1f seconds",
                target_model_idx,
                time.time() - baseline_time,
            )
        return mia_score_list, membership_list, offline_a