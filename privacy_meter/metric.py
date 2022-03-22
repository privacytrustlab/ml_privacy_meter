from typing import Callable, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from .dataset import Dataset
from .model import Model


class Metric(ABC):
    """
    Interface to construct and perform a membership inference attack
    on a target model and dataset using auxiliary information specified
    by the user. This serves as a guideline for implementing a metric
    to be used for measuring the privacy leakage of a target model.
    """

    def __init__(self, target_model: Model, target_dataset: Dataset,
                 auxiliary_model_list: List[Model], auxiliary_dataset: Dataset,
                 signal_func_list: List[Callable],
                 threshold_func: Callable):
        """
        Constructor
        Args:
            target_model: Model that the metric will be performed on
            target_dataset: Dataset corresponding to the target model
            auxiliary_model_list: Model(s) that the metric will compute signals on
            auxiliary_dataset: Dataset corresponding to the auxiliary model(s)
            signal_func_list: Function(s) that will be used for computing signals
            threshold_func: Function that will be used for computing attack threshold(s)
        """

        self.target_model = target_model
        self.target_dataset = target_dataset
        self.auxiliary_model_list = auxiliary_model_list
        self.auxiliary_dataset = auxiliary_dataset
        self.signal_func_list = signal_func_list
        self.threshold_func = threshold_func

    @abstractmethod
    def prepare_metric(self):
        """
        Function to prepare data needed for running the metric on
        the target model and dataset, using signals computed on the
        auxiliary model(s) and dataset.
        """
        pass

    @abstractmethod
    def run_metric(self, fpr_tolerance_rate_list=None):
        """
        Function to run the metric on the target model and dataset.
        Args:
            fpr_tolerance_rate_list (optional): List of FPR tolerance values
            that may be used by the threshold function to compute the attack
            threshold for the metric.
        """
        pass


class PopulationMetric(Metric):
    """
    Inherits the Metric class to perform the population membership inference attack
    which will be used as a metric for measuring privacy leakage of a target model.
    """

    def __init__(self, target_model: Model, target_dataset: Dataset,
                 auxiliary_model_list: List[Model], auxiliary_dataset: Dataset,
                 signal_func_list: List[Callable], threshold_func: Callable):
        """
        Constructor
        Args:
            target_model: Model that the metric will be performed on
            target_dataset: Dataset corresponding to the target model
            auxiliary_model_list: Model(s) that the metric will compute signals on
            auxiliary_dataset: Dataset corresponding to the auxiliary model(s)
            signal_func_list: Function(s) that will be used for computing signals
            threshold_func: Function that will be used for computing attack threshold(s)
        """

        # Initializes the parent metric
        super().__init__(target_model, target_dataset,
                         auxiliary_model_list, auxiliary_dataset,
                         signal_func_list, threshold_func)

        self.member_signals = None
        self.non_member_signals = None
        self.auxiliary_signals = None

        self.prepare_metric()

    def prepare_metric(self):
        """
        Function to prepare data needed for running the metric on
        the target model and dataset, using signals computed on the
        auxiliary model(s) and dataset. For the population attack,
        the auxiliary model is the target model itself, and the
        auxiliary dataset is a random split from the target model's
        training data.
        """

        member_signals = []
        for signal_func in self.signal_func_list:
            signals = signal_func(
                model=self.target_model,
                dataset=self.auxiliary_dataset,
                split_name='train',
                input_feature_name='<default_input>',
                output_feature_name='<default_output>',
                indices=None
            )
            member_signals.append(signals)
        # for population metric we have a list of loss values
        self.member_signals = np.array(member_signals).flatten()

        non_member_signals = []
        for signal_func in self.signal_func_list:
            signals = signal_func(
                model=self.target_model,
                dataset=self.auxiliary_dataset,
                split_name='test',
                input_feature_name='<default_input>',
                output_feature_name='<default_output>',
                indices=None
            )
            non_member_signals.append(signals)
        # for population metric we have a list of loss values
        self.non_member_signals = np.array(non_member_signals).flatten()

        auxiliary_signals = []
        for auxiliary_model in self.auxiliary_model_list:
            for signal_func in self.signal_func_list:
                signals = signal_func(
                    model=auxiliary_model,
                    dataset=self.auxiliary_dataset,
                    split_name='population',
                    input_feature_name='<default_input>',
                    output_feature_name='<default_output>',
                    indices=None
                )
                auxiliary_signals.append(signals)
        # for population metric we have a list of loss values
        self.auxiliary_signals = np.array(population_signals).flatten()

    def run_metric(self, fpr_tolerance_rate_list=None):
        """
        Function to run the metric on the target model and dataset.
        Args:
            fpr_tolerance_rate_list (optional): List of FPR tolerance values
            that may be used by the threshold function to compute the attack
            threshold for the metric.
        """
        for fpr_tolerance_rate in fpr_tolerance_rate_list:
            threshold = self.threshold_func(self.auxiliary_signals, fpr_tolerance_rate)

            member_preds = []
            for signal in self.member_signals:
                if signal <= threshold:
                    member_preds.append(1)
                else:
                    member_preds.append(0)

            non_member_preds = []
            for signal in self.non_member_signals:
                if signal <= threshold:
                    non_member_preds.append(1)
                else:
                    non_member_preds.append(0)

            preds = np.concatenate([member_preds, non_member_preds])

            y_eval = [1] * len(self.member_signals)
            y_eval.extend([0] * len(self.non_member_signals))

            acc = accuracy_score(y_eval, preds)
            roc_auc = roc_auc_score(y_eval, preds)
            tn, fp, fn, tp = confusion_matrix(y_eval, preds).ravel()

            print(
                f"Metric performance:\n"
                f"FPR Tolerance Rate = {fpr_tolerance_rate}\n"
                f"Accuracy = {acc}\n"
                f"ROC AUC Score = {roc_auc}\n"
                f"FPR: {fp / (fp + tn)}\n"
                f"TN, FP, FN, TP = {tn, fp, fn, tp}"
            )


class ShadowMetric(Metric):
    """
    Inherits the Metric class to perform the shadow membership inference attack
    which will be used as a metric for measuring privacy leakage of a target model.
    """

    def __init__(self, target_model: Model, target_dataset: Dataset,
                 auxiliary_model_list: List[Model], auxiliary_dataset: Dataset,
                 signal_func_list: List[Callable], threshold_func: Callable,
                 shadow_split_names: List):
        """
        Constructor
        Args:
            target_model: Model that the metric will be performed on
            target_dataset: Dataset corresponding to the target model
            auxiliary_model_list: Model(s) that the metric will compute signals on
            auxiliary_dataset: Dataset corresponding to the auxiliary model(s)
            signal_func_list: Function(s) that will be used for computing signals
            threshold_func: Function that will be used for computing attack threshold(s)
        """

        # Initializes the parent metric
        super().__init__(target_model, target_dataset,
                         auxiliary_model_list, auxiliary_dataset,
                         signal_func_list, threshold_func)

        self.shadow_split_names = shadow_split_names

        self.target_member_signals = None
        self.target_non_member_signals = None
        self.auxiliary_member_signals = None
        self.auxiliary_non_member_signals = None

        self.prepare_metric()

    def prepare_metric(self):
        """
        Function to prepare data needed for running the metric on
        the target model and dataset, using signals computed on the
        auxiliary model(s) and dataset. For the shadow attack, the
        auxiliary models will be a list of shadow models and the
        auxiliary dataset will contain the train-test splits of these
        models.
        """
        member_signals = []
        for signal_func in self.signal_func_list:
            signals = signal_func(
                model=self.target_model,
                dataset=self.target_dataset,
                split_name='train',
                input_feature_name='<default_input>',
                output_feature_name='<default_output>',
                indices=None
            )
            member_signals.append(signals)
        # for shadow metric we have a list of loss values
        self.member_signals = np.array(member_signals).flatten()

        non_member_signals = []
        for signal_func in self.signal_func_list:
            signals = signal_func(
                model=self.target_model,
                dataset=self.target_dataset,
                split_name='test',
                input_feature_name='<default_input>',
                output_feature_name='<default_output>',
                indices=None
            )
            non_member_signals.append(signals)
        # for shadow metric we have a list of loss values
        self.non_member_signals = np.array(non_member_signals).flatten()

        auxiliary_member_signals, auxiliary_non_member_signals = [], []
        for idx, auxiliary_model in enumerate(self.auxiliary_model_list):
            train_split_name, test_split_name = self.shadow_split_names[idx]
            print(train_split_name)
            print(test_split_name)

            for signal_func in self.signal_func_list:
                m_signals = signal_func(
                    model=auxiliary_model,
                    dataset=self.auxiliary_dataset,
                    split_name=train_split_name,
                    input_feature_name='<default_input>',
                    output_feature_name='<default_output>'
                )
                auxiliary_member_signals.append(m_signals)

                nm_signals = signal_func(
                    model=auxiliary_model,
                    dataset=self.auxiliary_dataset,
                    split_name=test_split_name,
                    input_feature_name='<default_input>',
                    output_feature_name='<default_output>'
                )
                auxiliary_non_member_signals.append(nm_signals)
        self.auxiliary_member_signals = np.array(auxiliary_member_signals).flatten()
        self.auxiliary_non_member_signals = np.array(auxiliary_non_member_signals).flatten()

    def run_metric(self, fpr_tolerance_rate_list=None):
        """
        Function to run the metric on the target model and dataset.
        Args:
            fpr_tolerance_rate_list (optional): List of FPR tolerance values
            that may be used by the threshold function to compute the attack
            threshold for the metric.
        """
        clf = LogisticRegression()

        x = np.concatenate([self.auxiliary_member_signals, self.auxiliary_non_member_signals]).reshape(-1, 1)
        member_labels = [1] * len(self.auxiliary_member_signals)
        non_member_labels = [0] * len(self.auxiliary_non_member_signals)
        y = np.concatenate([member_labels, non_member_labels])
        clf.fit(x, y)

        member_preds = clf.predict(self.member_signals.reshape(-1, 1))
        non_member_preds = clf.predict(self.non_member_signals.reshape(-1, 1))

        preds = np.concatenate([member_preds, non_member_preds])

        y_eval = [1] * len(self.member_signals)
        y_eval.extend([0] * len(self.non_member_signals))

        acc = accuracy_score(y_eval, preds)
        roc_auc = roc_auc_score(y_eval, preds)
        tn, fp, fn, tp = confusion_matrix(y_eval, preds).ravel()

        print(
            f"Metric performance:\n"
            f"Accuracy = {acc}\n"
            f"ROC AUC Score = {roc_auc}\n"
            f"FPR: {fp / (fp + tn)}\n"
            f"TN, FP, FN, TP = {tn, fp, fn, tp}"
        )