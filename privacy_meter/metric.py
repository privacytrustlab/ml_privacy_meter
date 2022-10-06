import os
from abc import ABC, abstractmethod
from typing import Callable, Optional, List, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression

from privacy_meter.constants import *
from privacy_meter.information_source import InformationSource
from privacy_meter.information_source_signal import GroupInfo, Signal
from privacy_meter.metric_result import MetricResult
from privacy_meter.utils import flatten_array

########################################################################################################################
# METRIC CLASS
########################################################################################################################


class Metric(ABC):
    """
    Interface to construct and perform a membership inference attack on a target model and dataset using auxiliary
    information specified by the user. This serves as a guideline for implementing a metric to be used for measuring
    the privacy leakage of a target model.
    """

    def __init__(
            self,
            target_info_source: InformationSource,
            reference_info_source: InformationSource,
            signals: List[Signal],
            hypothesis_test_func: Optional[Callable],
            logs_dirname: str
    ):
        """
        Constructor

        Args:
            target_info_source: InformationSource, containing the Model that the metric will be performed on, and the
                corresponding Dataset.
            reference_info_source: List of InformationSource(s), containing the Model(s) that the metric will be
                fitted on, and their corresponding Dataset.
            signals: List of signals to be used.
            hypothesis_test_func: Function that will be used for computing attack threshold(s)
        """

        self.target_info_source = target_info_source
        self.reference_info_source = reference_info_source
        self.signals = signals
        self.hypothesis_test_func = hypothesis_test_func
        self.logs_dirname = logs_dirname

    def _load_or_compute_signal(
            self,
            signal_source: SignalSourceEnum,
    ):
        """
        Private helper function to load signals if they have been computed already, or compute and save signals
        if they haven't.

        Args:
            signal_source: Signal source to determine which information source and mapping objects need to be used.

        Returns:
            Signals computed using the specified information source and mapping object.
        """
        signal_filepath = f'{self.logs_dirname}/{type(self).__name__}_{signal_source.value}'

        if signal_source == SignalSourceEnum.TARGET_MEMBER:
            info_source_obj = self.target_info_source
            mapping_obj = self.target_model_to_train_split_mapping
        elif signal_source == SignalSourceEnum.TARGET_NON_MEMBER:
            info_source_obj = self.target_info_source
            mapping_obj = self.target_model_to_test_split_mapping
        elif signal_source == SignalSourceEnum.REFERENCE_MEMBER or signal_source == SignalSourceEnum.REFERENCE:
            info_source_obj = self.reference_info_source
            mapping_obj = self.reference_model_to_train_split_mapping
        elif signal_source == SignalSourceEnum.REFERENCE_NON_MEMBER:
            info_source_obj = self.reference_info_source
            mapping_obj = self.reference_model_to_test_split_mapping
        else:
            raise NotImplementedError

        signals = []

        if os.path.isfile(f'{signal_filepath}{NPZ_EXTENSION}'):
            with np.load(f'{signal_filepath}{NPZ_EXTENSION}', allow_pickle=True) as data:
                signals = np.array(data['arr_0'][()])
        else:
            # For each signal, compute the response of the model on the dataset according to the mapping
            for signal in self.signals:
                signals.append(
                    info_source_obj.get_signal(signal, mapping_obj)
                )
            np.savez(signal_filepath, signals)

        return signals

    def _load_or_compute_group_membership(
            self,
            signal_source: SignalSourceEnum,
    ):
        """
        Private helper function to compute group membership 
        Args:
            signal_source: Signal source to determine which information source and mapping objects need to be used.

        Returns:
            Group membership computed using the specified information source and mapping object.
        """
        if signal_source == SignalSourceEnum.TARGET_MEMBER:
            info_source_obj = self.target_info_source
            mapping_obj = self.target_model_to_train_split_mapping_group
        elif signal_source == SignalSourceEnum.TARGET_NON_MEMBER:
            info_source_obj = self.target_info_source
            mapping_obj = self.target_model_to_test_split_mapping_group
        elif signal_source == SignalSourceEnum.REFERENCE_MEMBER or signal_source == SignalSourceEnum.REFERENCE:
            info_source_obj = self.reference_info_source
            mapping_obj = self.reference_model_to_train_split_mapping_group
        elif signal_source == SignalSourceEnum.REFERENCE_NON_MEMBER:
            info_source_obj = self.reference_info_source
            mapping_obj = self.reference_model_to_test_split_mapping_group
        else:
            raise NotImplementedError
        
        group_membership = info_source_obj.get_signal(GroupInfo(), mapping_obj)

        return group_membership

    def _set_default_mappings(
            self,
            unique_dataset: bool
    ):
        """
        Private helper function, to set default values for mappings between models and dataset splits.

        Args:
            unique_dataset: Boolean indicating if target_info_source and reference_info_source use one same dataset
                object.

        """
        if unique_dataset:
            if self.target_model_to_train_split_mapping is None:
                self.target_model_to_train_split_mapping = [(0, 'train000', '<default_input>', '<default_output>')]
            if self.target_model_to_test_split_mapping is None:
                self.target_model_to_test_split_mapping = [(0, 'test000', '<default_input>', '<default_output>')]
            if self.reference_model_to_train_split_mapping is None:
                self.reference_model_to_train_split_mapping = [
                    (0, f'train{k + 1:03d}', '<default_input>', '<default_output>')
                    for k in range(len(self.reference_info_source.models))
                ]
            if self.reference_model_to_test_split_mapping is None:
                self.reference_model_to_test_split_mapping = [
                    (0, f'test{k + 1:03d}', '<default_input>', '<default_output>')
                    for k in range(len(self.reference_info_source.models))
                ]
        else:
            if self.target_model_to_train_split_mapping is None:
                self.target_model_to_train_split_mapping = [(0, 'train', '<default_input>', '<default_output>')]
            if self.target_model_to_test_split_mapping is None:
                self.target_model_to_test_split_mapping = [(0, 'test', '<default_input>', '<default_output>')]
            if self.reference_model_to_train_split_mapping is None:
                self.reference_model_to_train_split_mapping = [
                    (k, f'train', '<default_input>', '<default_output>')
                    for k in range(len(self.reference_info_source.models))
                ]
            if self.reference_model_to_test_split_mapping is None:
                self.reference_model_to_test_split_mapping = [
                    (k, f'test', '<default_input>', '<default_output>')
                    for k in range(len(self.reference_info_source.models))
                ]

    def _set_default_group_mappings(self, unique_dataset: bool):
        """
        Private helper function, to set default values for mappings between models and dataset splits for groups.
        Args:
            unique_dataset: Boolean indicating if target_info_source and reference_info_source use one same dataset
                object.

        """
        if unique_dataset:
            if self.target_model_to_train_split_mapping_group is None:
                self.target_model_to_train_split_mapping_group = [(0, 'train000', '<default_group>')]
            if self.target_model_to_test_split_mapping_group is None:
                self.target_model_to_test_split_mapping_group = [(0, 'test000', '<default_group>')]
            if self.reference_model_to_train_split_mapping_group is None:
                self.reference_model_to_train_split_mapping_group = [
                    (0, f'train{k + 1:03d}', '<default_group>')
                    for k in range(len(self.reference_info_source.models))
                ]
            if self.reference_model_to_test_split_mapping_group is None:
                self.reference_model_to_test_split_mapping_group = [
                    (0, f'test{k + 1:03d}', '<default_group>')
                    for k in range(len(self.reference_info_source.models))
                ]
        else:
            if self.target_model_to_train_split_mapping_group is None:
                self.target_model_to_train_split_mapping_group = [(0, 'train', '<default_group>')]
            if self.target_model_to_test_split_mapping_group is None:
                self.target_model_to_test_split_mapping_group = [(0, 'test', '<default_group>')]
            if self.reference_model_to_train_split_mapping_group is None:
                self.reference_model_to_train_split_mapping_group = [
                    (k, f'train','<default_group>')
                    for k in range(len(self.reference_info_source.models))
                ]
            if self.reference_model_to_test_split_mapping_group is None:
                self.reference_model_to_test_split_mapping_group = [
                    (k, f'test', '<default_group>')
                    for k in range(len(self.reference_info_source.models))
                ]

    @abstractmethod
    def prepare_metric(self):
        """
        Function to prepare data needed for running the metric on the target model and dataset, using signals computed
        on the auxiliary model(s) and dataset.
        """
        pass

    @abstractmethod
    def run_metric(self, fpr_tolerance_rate_list=None) -> Union[MetricResult, List[MetricResult]]:
        """
        Function to run the metric on the target model and dataset.

        Args:
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
                to compute the attack threshold for the metric.

        Returns:
            Result(s) of the metric.
        """
        pass

########################################################################################################################
# POPULATION_METRIC CLASS
########################################################################################################################


class PopulationMetric(Metric):
    """
    Inherits from the Metric class to perform the population membership inference attack which will be used as a metric
    for measuring privacy leakage of a target model.
    """

    def __init__(
            self,
            target_info_source: InformationSource,
            reference_info_source: InformationSource,
            signals: List[Signal],
            hypothesis_test_func: Optional[Callable],
            target_model_to_train_split_mapping: List[Tuple[int, str, str, str]] = None,
            target_model_to_test_split_mapping: List[Tuple[int, str, str, str]] = None,
            reference_model_to_train_split_mapping: List[Tuple[int, str, str, str]] = None,
            unique_dataset: bool = False,
            logs_dirname: str = None
    ):
        """
        Constructor

        Args:
            target_info_source: InformationSource, containing the Model that the metric will be performed on, and the
                corresponding Dataset.
            reference_info_source: List of InformationSource(s), containing the Model(s) that the metric will be
                fitted on, and their corresponding Dataset.
            signals: List of signals to be used.
            hypothesis_test_func: Function that will be used for computing attack threshold(s)
            target_model_to_train_split_mapping: Mapping from the target model to the train split of the target dataset.
                By default, the code will look for a split named "train".
            target_model_to_test_split_mapping: Mapping from the target model to the test split of the target dataset.
                By default, the code will look for a split named "test".
            reference_model_to_train_split_mapping: Mapping from the reference models to their train splits of the
                corresponding reference dataset. By default, the code will look for a split named "train" if only one
                reference model is provided, else for splits named "train000", "train001", "train002", etc. For the
                population metric, at least one reference dataset should be passed.
        """

        # Initializes the parent metric
        super().__init__(target_info_source=target_info_source,
                         reference_info_source=reference_info_source,
                         signals=signals,
                         hypothesis_test_func=hypothesis_test_func,
                         logs_dirname=logs_dirname)

        # Useless object, for compatibility purposes only
        self.reference_model_to_test_split_mapping = None

        # Logs directory
        self.logs_dirname = logs_dirname

        # Store the model to split mappings
        self.target_model_to_train_split_mapping = target_model_to_train_split_mapping
        self.target_model_to_test_split_mapping = target_model_to_test_split_mapping
        self.reference_model_to_train_split_mapping = reference_model_to_train_split_mapping
        self._set_default_mappings(unique_dataset)

        # Variables used in prepare_metric and run_metric
        self.member_signals, self.non_member_signals = [], []
        self.reference_signals = []

    def prepare_metric(self):
        """
        Function to prepare data needed for running the metric on the target model and dataset, using signals computed
        on the auxiliary model(s) and dataset. For the population attack, the auxiliary model is the target model
        itself, and the auxiliary dataset is a random split from the target model's training data.
        """
        # Load signals if they have been computed already; otherwise, compute and save them
        self.member_signals = flatten_array(self._load_or_compute_signal(SignalSourceEnum.TARGET_MEMBER))
        self.non_member_signals = flatten_array(self._load_or_compute_signal(SignalSourceEnum.TARGET_NON_MEMBER))
        self.reference_signals = flatten_array(self._load_or_compute_signal(SignalSourceEnum.REFERENCE))

    def run_metric(self, fpr_tolerance_rate_list=None) -> List[MetricResult]:
        """
        Function to run the metric on the target model and dataset.

        Args:
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
                to compute the attack threshold for the metric.

        Returns:
            A list of MetricResult objects, one per fpr value.
        """
        metric_result_list = []
        for fpr_tolerance_rate in fpr_tolerance_rate_list:
            threshold = self.hypothesis_test_func(self.reference_signals, fpr_tolerance_rate)

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

            predictions = np.concatenate([member_preds, non_member_preds])

            true_labels = [1] * len(self.member_signals)
            true_labels.extend([0] * len(self.non_member_signals))

            signal_values = np.concatenate([self.member_signals, self.non_member_signals])

            metric_result = MetricResult(
                metric_id=MetricEnum.POPULATION.value,
                predicted_labels=predictions,
                true_labels=true_labels,
                predictions_proba=None,
                signal_values=signal_values,
                threshold=threshold
            )

            metric_result_list.append(metric_result)

        return metric_result_list

########################################################################################################################
# SHADOW_METRIC CLASS
########################################################################################################################


class ShadowMetric(Metric):
    """
    Inherits from the Metric class to perform the shadow membership inference attack which will be used as a metric for
    measuring privacy leakage of a target model.
    """

    def __init__(
            self,
            target_info_source: InformationSource,
            reference_info_source: InformationSource,
            signals: List[Signal],
            hypothesis_test_func: Optional[Callable],
            target_model_to_train_split_mapping: List[Tuple[int, str, str, str]] = None,
            target_model_to_test_split_mapping: List[Tuple[int, str, str, str]] = None,
            reference_model_to_train_split_mapping: List[Tuple[int, str, str, str]] = None,
            reference_model_to_test_split_mapping: List[Tuple[int, str, str, str]] = None,
            reweight_samples: bool = True,
            unique_dataset: bool = False,
            logs_dirname: str = None
    ):
        """
        Constructor
        
        Args:
            target_info_source: InformationSource, containing the Model that the metric will be performed on, and the
                corresponding Dataset.
            reference_info_source: List of InformationSource(s), containing the Model(s) that the metric will be
                fitted on, and their corresponding Dataset.
            signals: List of signals to be used.
            hypothesis_test_func: Function that will be used for computing attack threshold(s)
            target_model_to_train_split_mapping: Mapping from the target model to the train split of the target dataset.
                By default, the code will look for a split named "train"
            target_model_to_test_split_mapping: Mapping from the target model to the test split of the target dataset.
                By default, the code will look for a split named "test"
            reference_model_to_train_split_mapping: Mapping from the reference models to their train splits of the
                corresponding reference dataset. By default, the code will look for a split named "train" if only one
                reference model is provided, else for splits named "train000", "train001", "train002", etc.
            reference_model_to_test_split_mapping: Mapping from the reference models to their test splits of the
                corresponding reference dataset. By default, the code will look for a split named "test" if only one
                reference model is provided, else for splits named "test000", "test001", "test002", etc.
            reweight_samples: Boolean specifying if the metric should account for an unbalance between the number of
                members vs non-members.
            unique_dataset: Boolean indicating if target_info_source and reference_info_source use one same dataset
                object.
        """

        # Initializes the parent metric
        super().__init__(target_info_source=target_info_source,
                         reference_info_source=reference_info_source,
                         signals=signals,
                         hypothesis_test_func=hypothesis_test_func,
                         logs_dirname=logs_dirname)

        # Logs directory
        self.logs_dirname = logs_dirname

        self.reweight_samples = reweight_samples

        # Store the model to split mappings
        self.target_model_to_train_split_mapping = target_model_to_train_split_mapping
        self.target_model_to_test_split_mapping = target_model_to_test_split_mapping
        self.reference_model_to_train_split_mapping = reference_model_to_train_split_mapping
        self.reference_model_to_test_split_mapping = reference_model_to_test_split_mapping
        self._set_default_mappings(unique_dataset)

        # Variables used in prepare_metric and run_metric
        self.member_signals, self.non_member_signals = [], []
        self.reference_member_signals, self.reference_non_member_signals = [], []

    def prepare_metric(self):
        """
        Function to prepare data needed for running the metric on the target model and dataset, using signals computed
        on the reference model(s) and dataset. For the shadow attack, the reference models will be a list of shadow
        models and the auxiliary dataset will contain the train-test splits of these models.
        """
        # Load signals if they have been computed already; otherwise, compute and save them
        self.member_signals = flatten_array(self._load_or_compute_signal(signal_source=SignalSourceEnum.TARGET_MEMBER))
        self.non_member_signals = flatten_array(self._load_or_compute_signal(SignalSourceEnum.TARGET_NON_MEMBER))
        self.reference_member_signals = flatten_array(self._load_or_compute_signal(SignalSourceEnum.REFERENCE_MEMBER))
        self.reference_non_member_signals = flatten_array(
            self._load_or_compute_signal(SignalSourceEnum.REFERENCE_NON_MEMBER))

    def run_metric(self, fpr_tolerance_rate_list=None) -> MetricResult:
        """
        Function to run the metric on the target model and dataset.

        Args:
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
                to compute the attack threshold for the metric.

        Returns:
            The result of the metric
        """

        # Create and fit a LogisticRegression object, from the members and non-members of the reference
        # InformationSource
        clf = LogisticRegression(class_weight={
            0: self.reference_member_signals.shape[0],
            1: self.reference_non_member_signals.shape[0],
        } if self.reweight_samples else None)
        x = np.concatenate([self.reference_member_signals, self.reference_non_member_signals]).reshape(-1, 1)
        y = np.array([1] * len(self.reference_member_signals) + [0] * len(self.reference_non_member_signals))
        clf.fit(x, y)

        signal_space = np.linspace(np.array(x).ravel().min(), np.array(x).ravel().max(), 200).reshape((-1, 1))
        i = np.max([i if v == 1 else -1 for i, v in enumerate(clf.predict(signal_space))])
        threshold = signal_space[i:i + 2].mean()

        # Predict the membership status of samples in the target InformationSource
        predictions_proba = clf.predict_proba(np.concatenate([
            self.member_signals.reshape(-1, 1),
            self.non_member_signals.reshape(-1, 1)
        ]))
        predictions_label = np.argmax(predictions_proba, axis=1)
        predictions_proba = predictions_proba[:, 1]

        true_labels = [1] * len(self.member_signals) + [0] * len(self.non_member_signals)
        signal_values = np.concatenate([self.member_signals, self.non_member_signals])

        # Evaluate the power of this inference and display the result
        metric_result = MetricResult(
            metric_id=MetricEnum.SHADOW.value,
            predictions_proba=predictions_proba,
            predicted_labels=predictions_label,
            true_labels=true_labels,
            signal_values=signal_values,
            threshold=threshold
        )

        return metric_result

########################################################################################################################
# REFERENCE_METRIC CLASS
########################################################################################################################


class ReferenceMetric(Metric):
    """
    Inherits from the Metric class to perform the reference membership inference attack which will be used as a metric
    for measuring privacy leakage of a target model.
    """

    def __init__(
            self,
            target_info_source: InformationSource,
            reference_info_source: InformationSource,
            signals: List[Signal],
            hypothesis_test_func: Optional[Callable],
            target_model_to_train_split_mapping: List[Tuple[int, str, str, str]] = None,
            target_model_to_test_split_mapping: List[Tuple[int, str, str, str]] = None,
            reference_model_to_train_split_mapping: List[Tuple[int, str, str, str]] = None,
            reference_model_to_test_split_mapping: List[Tuple[int, str, str, str]] = None,
            unique_dataset: bool = False,
            logs_dirname: str = None
    ):
        """
        Constructor

        Args:
            target_info_source: InformationSource, containing the Model that the metric will be performed on, and the
                corresponding Dataset.
            reference_info_source: List of InformationSource(s), containing the Model(s) that the metric will be
                fitted on, and their corresponding Dataset.
            signals: List of signals to be used.
            hypothesis_test_func: Function that will be used for computing attack threshold(s)
            target_model_to_train_split_mapping: Mapping from the target model to the train split of the target dataset.
                By default, the code will look for a split named "train"
            target_model_to_test_split_mapping: Mapping from the target model to the test split of the target dataset.
                By default, the code will look for a split named "test"
            reference_model_to_train_split_mapping: Mapping from the reference models to their train splits of the
                corresponding reference dataset. By default, the code will look for a split named "train"
            reference_model_to_test_split_mapping: Mapping from the reference models to their test splits of the
                corresponding reference dataset. By default, the code will look for a split named "test"
            unique_dataset: Boolean indicating if target_info_source and target_info_source use one same dataset object.
        """

        # Initializes the parent metric
        super().__init__(target_info_source=target_info_source,
                         reference_info_source=reference_info_source,
                         signals=signals,
                         hypothesis_test_func=hypothesis_test_func,
                         logs_dirname=logs_dirname)

        # Logs directory
        self.logs_dirname = logs_dirname

        # Store the model to split mappings
        self.target_model_to_train_split_mapping = target_model_to_train_split_mapping
        self.target_model_to_test_split_mapping = target_model_to_test_split_mapping

        # Custom default mapping for the reference metric
        if reference_model_to_train_split_mapping is None:
            self.reference_model_to_train_split_mapping = [
                                                              (0, 'train', '<default_input>', '<default_output>')
                                                          ] * len(self.reference_info_source.models)
        if reference_model_to_test_split_mapping is None:
            self.reference_model_to_test_split_mapping = [
                                                             (0, 'test', '<default_input>', '<default_output>')
                                                         ] * len(self.reference_info_source.models)

        self._set_default_mappings(unique_dataset)

        # Variables used in prepare_metric and run_metric
        self.member_signals, self.non_member_signals = [], []
        self.reference_member_signals, self.reference_non_member_signals = [], []
        self.pointwise_member_thresholds, self.pointwise_non_member_thresholds = [], []

    def prepare_metric(self):
        """
        Function to prepare data needed for running the metric on the target model and dataset, using signals computed
        on the reference model(s) and dataset. For the reference attack, the reference models will be a list of models
        trained on data from the same distribution, and the reference dataset will be the target model's train-test
        split.
        """
        # Load signals if they have been computed already; otherwise, compute and save them
        self.member_signals = flatten_array(self._load_or_compute_signal(SignalSourceEnum.TARGET_MEMBER))
        self.non_member_signals = flatten_array(self._load_or_compute_signal(SignalSourceEnum.TARGET_NON_MEMBER))
        self.reference_member_signals = np.array(
            self._load_or_compute_signal(SignalSourceEnum.REFERENCE_MEMBER)[0]).transpose()
        self.reference_non_member_signals = np.array(
            self._load_or_compute_signal(SignalSourceEnum.REFERENCE_NON_MEMBER)[0]).transpose()

    def run_metric(self, fpr_tolerance_rate_list=None) -> List[MetricResult]:
        """
        Function to run the metric on the target model and dataset.

        Args:
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
                to compute the attack threshold for the metric.

        Returns:
            A list of MetricResult objects, one per fpr value.
        """
        metric_result_list = []
        for fpr_tolerance_rate in fpr_tolerance_rate_list:
            member_preds = []
            for idx, signal in enumerate(self.member_signals):
                # Use pointwise threshold
                threshold = self.hypothesis_test_func(self.reference_member_signals[idx], fpr_tolerance_rate)
                if signal <= threshold:
                    member_preds.append(1)
                else:
                    member_preds.append(0)

            non_member_preds = []
            for idx, signal in enumerate(self.non_member_signals):
                # Use pointwise threshold
                threshold = self.hypothesis_test_func(self.reference_non_member_signals[idx], fpr_tolerance_rate)
                if signal <= threshold:
                    non_member_preds.append(1)
                else:
                    non_member_preds.append(0)

            predictions = np.concatenate([member_preds, non_member_preds])

            true_labels = [1] * len(self.member_signals)
            true_labels.extend([0] * len(self.non_member_signals))

            signal_values = np.concatenate([self.member_signals, self.non_member_signals])

            metric_result = MetricResult(metric_id=MetricEnum.REFERENCE.value,
                                         predicted_labels=predictions,
                                         true_labels=true_labels,
                                         predictions_proba=None,
                                         signal_values=signal_values)

            metric_result_list.append(metric_result)

        return metric_result_list



########################################################################################################################
# GroupPopulationMetric CLASS
########################################################################################################################


class GroupPopulationMetric(Metric):
    """
    Inherits from the Metric class to perform the population membership inference attack which will be used as a metric
    for measuring privacy leakage of a target model. Compared to PopulationMetric, this new metric is designed to compute the threshold per class.
    """

    def __init__(
            self,
            target_info_source: InformationSource,
            reference_info_source: InformationSource,
            signals: List[Signal],
            hypothesis_test_func: Optional[Callable],
            target_model_to_train_split_mapping: List[Tuple[int, str, str, str]] = None,
            target_model_to_test_split_mapping: List[Tuple[int, str, str, str]] = None,
            reference_model_to_train_split_mapping: List[Tuple[int, str, str, str]] = None,
            target_model_to_train_split_mapping_group: List[Tuple[int, str, str]] = None,
            target_model_to_test_split_mapping_group: List[Tuple[int, str, str]] = None,
            reference_model_to_train_split_mapping_group: List[Tuple[int, str, str]] = None,
            unique_dataset: bool = False,
            logs_dirname: str = None
    ):
        """
        Constructor

        Args:
            target_info_source: InformationSource, containing the Model that the metric will be performed on, and the
                corresponding Dataset.
            reference_info_source: List of InformationSource(s), containing the Model(s) that the metric will be
                fitted on, and their corresponding Dataset.
            signals: List of signals to be used.
            hypothesis_test_func: Function that will be used for computing attack threshold(s)
            target_model_to_train_split_mapping: Mapping from the target model to the train split of the target dataset.
                By default, the code will look for a split named "train".
            target_model_to_test_split_mapping: Mapping from the target model to the test split of the target dataset.
                By default, the code will look for a split named "test".
            reference_model_to_train_split_mapping: Mapping from the reference models to their train splits of the
                corresponding reference dataset. By default, the code will look for a split named "train" if only one
                reference model is provided, else for splits named "train000", "train001", "train002", etc. For the
                population metric, at least one reference dataset should be passed.
            target_model_to_train_split_mapping_group: Mapping from the target model to the train split of the target dataset with respect to group information.
                By default, the code will look for a split named "train".
            target_model_to_test_split_mapping_group: Mapping from the target model to the test split of the target dataset with respect to group information.
                By default, the code will look for a split named "test".
            reference_model_to_train_split_mapping_group: Mapping from the reference models to their train splits of the
                corresponding reference dataset  with respect to group information.. By default, the code will look for a split named "train" if only one
                reference model is provided, else for splits named "train000", "train001", "train002", etc. For the
                population metric, at least one reference dataset should be passed.
        """

        # Initializes the parent metric
        super().__init__(target_info_source=target_info_source,
                         reference_info_source=reference_info_source,
                         signals=signals,
                         hypothesis_test_func=hypothesis_test_func,
                         logs_dirname=logs_dirname)

        # Useless object, for compatibility purposes only
        self.reference_model_to_test_split_mapping = None
        self.reference_model_to_test_split_mapping_group = None

        # Logs directory
        self.logs_dirname = logs_dirname

        # Store the model to split mappings
        self.target_model_to_train_split_mapping = target_model_to_train_split_mapping
        self.target_model_to_test_split_mapping = target_model_to_test_split_mapping
        self.reference_model_to_train_split_mapping = reference_model_to_train_split_mapping
        self.target_model_to_train_split_mapping_group = target_model_to_train_split_mapping_group
        self.target_model_to_test_split_mapping_group = target_model_to_test_split_mapping_group
        self.reference_model_to_train_split_mapping_group = reference_model_to_train_split_mapping_group


        self._set_default_group_mappings(unique_dataset) # get the mapping for the groups
        self._set_default_mappings(unique_dataset) 

        # Variables used in prepare_metric and run_metric
        self.member_signals, self.non_member_signals = [], []
        self.reference_signals = []
        self.member_groups, self.non_member_groups = [], []
        self.reference_groups = []

    
        



    def prepare_metric(self):
        """
        Function to prepare data needed for running the metric on the target model and dataset, using signals computed
        on the auxiliary model(s) and dataset. For the population attack, the auxiliary model is the target model
        itself, and the auxiliary dataset is a random split from the target model's training data.
        """
        # Load signals if they have been computed already; otherwise, compute and save them
        self.member_signals = flatten_array(self._load_or_compute_signal(SignalSourceEnum.TARGET_MEMBER))
        self.non_member_signals = flatten_array(self._load_or_compute_signal(SignalSourceEnum.TARGET_NON_MEMBER))
        self.reference_signals = flatten_array(self._load_or_compute_signal(SignalSourceEnum.REFERENCE))
        self.reference_groups = flatten_array(self._load_or_compute_group_membership(SignalSourceEnum.REFERENCE))
        self.non_member_groups = flatten_array(self._load_or_compute_group_membership(SignalSourceEnum.TARGET_NON_MEMBER))
        self.member_groups = flatten_array(self._load_or_compute_group_membership(SignalSourceEnum.TARGET_MEMBER))


    def run_metric(self, fpr_tolerance_rate_list=None) -> List[MetricResult]:
        """
        Function to run the metric on the target model and dataset.

        Args:
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
                to compute the attack threshold for the metric.

        Returns:
            A list of MetricResult objects, one per fpr value.
        """
        metric_result_list = []
        for fpr_tolerance_rate in fpr_tolerance_rate_list:
            member_preds = []
            non_member_preds = []
            
            for g in np.unique(self.reference_groups):
                
                reference_index = np.where(self.reference_groups ==g)[0]
                non_member_index = np.where(self.non_member_groups ==g)[0]
                member_index = np.where(self.member_groups ==g)[0]
    
                threshold = self.hypothesis_test_func(self.reference_signals[reference_index], fpr_tolerance_rate)


                member_pred = (self.member_signals[member_index] < threshold).astype(int)
                non_member_pred = (self.non_member_signals[non_member_index] < threshold).astype(int)
            

                member_preds.append(member_pred)
                non_member_preds.append(non_member_pred)
            

            member_preds = np.concatenate(member_preds,axis=0)
            non_member_preds = np.concatenate(non_member_preds,axis=0)

            predictions = np.concatenate([member_preds, non_member_preds])
            

            true_labels = [1] * len(self.member_signals)
            true_labels.extend([0] * len(self.non_member_signals))

            signal_values = np.concatenate([self.member_signals, self.non_member_signals])

            print(predictions.shape)
            print(len(true_labels))
            metric_result = MetricResult(
                metric_id=MetricEnum.POPULATION.value,
                predicted_labels=predictions,
                true_labels=true_labels,
                predictions_proba=None,
                signal_values=signal_values,
                threshold=threshold
            )

            metric_result_list.append(metric_result)

        return metric_result_list
