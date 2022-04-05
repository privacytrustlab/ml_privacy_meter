import os

from abc import ABC, abstractmethod
from typing import Callable, Optional, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from privacy_meter.constants import *
from privacy_meter.information_source import InformationSource
from privacy_meter.information_source_signal import Signal
from privacy_meter.metric_result import MetricResult


class Metric(ABC):
    """
    Interface to construct and perform a membership inference attack
    on a target model and dataset using auxiliary information specified
    by the user. This serves as a guideline for implementing a metric
    to be used for measuring the privacy leakage of a target model.
    """

    def __init__(self,
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

    def __init__(
            self,
            target_info_source: InformationSource,
            reference_info_source: InformationSource,
            signals: List[Signal],
            hypothesis_test_func: Optional[Callable],
            target_model_to_train_split_mapping: List[Tuple[int, str, str, str]] = None,
            target_model_to_test_split_mapping: List[Tuple[int, str, str, str]] = None,
            reference_model_to_train_split_mapping: List[Tuple[int, str, str, str]] = None,
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
        self.reference_model_to_train_split_mapping = reference_model_to_train_split_mapping

        # Default values for all the model to split mappings
        if self.target_model_to_train_split_mapping is None:
            self.target_model_to_train_split_mapping = [(0, 'train', '<default_input>', '<default_output>')]
        if self.target_model_to_test_split_mapping is None:
            self.target_model_to_test_split_mapping = [(0, 'test', '<default_input>', '<default_output>')]
        if self.reference_model_to_train_split_mapping is None:
            if len(self.reference_info_source.models) == 1:
                self.reference_model_to_train_split_mapping = [(0, 'train', '<default_input>', '<default_output>')]
            else:
                self.reference_model_to_train_split_mapping = [
                    (0, f'train{k:03d}', '<default_input>', '<default_output>')
                    for k in range(len(self.reference_info_source.models))
                ]

        # Variables used in prepare_metric and run_metric
        self.member_signals, self.non_member_signals = [], []
        self.reference_signals = []

    def __load_or_compute_signals(self, signal_source):
        """
        Private helper function to load signals if they have been computed already, or compute and save signals
        if they haven't.

        Args:
            signal_source: Signal source to determine which information source and mapping objects need to be used

        Returns:
            Signals computed using the specified information source and mapping object.
        """
        signal_filepath, info_source_obj, mapping_obj = None, None, None
        if signal_source == SignalSourceEnum.TARGET_MEMBER:
            signal_filepath = f'{self.logs_dirname}/{MetricEnum.POPULATION.value}_{TARGET_MEMBER_SIGNALS_FILENAME}'
            info_source_obj = self.target_info_source
            mapping_obj = self.target_model_to_train_split_mapping
        elif signal_source == SignalSourceEnum.TARGET_NON_MEMBER:
            signal_filepath = f'{self.logs_dirname}/{MetricEnum.POPULATION.value}_{TARGET_NON_MEMBER_SIGNALS_FILENAME}'
            info_source_obj = self.target_info_source
            mapping_obj = self.target_model_to_test_split_mapping
        elif signal_source == SignalSourceEnum.REFERENCE:
            signal_filepath = f'{self.logs_dirname}/{MetricEnum.POPULATION.value}_{REFERENCE_SIGNALS_FILENAME}'
            info_source_obj = self.reference_info_source
            mapping_obj = self.reference_model_to_train_split_mapping

        signals = []
        if os.path.isfile(f'{signal_filepath}{NPZ_EXTENSION}'):
            with np.load(f'{signal_filepath}{NPZ_EXTENSION}', allow_pickle=True) as data:
                signals = data['arr_0'][()]
        else:
            # For each signal compute the response of both the model on the dataset according to the mapping
            for signal in self.signals:
                signals.append(
                    info_source_obj.get_signal(signal, mapping_obj)
                )
            # For the shadow metric we have a list of loss values
            signals = np.array(signals).flatten()
            np.savez(signal_filepath, signals)

        return signals

    def prepare_metric(self):
        """
        Function to prepare data needed for running the metric on
        the target model and dataset, using signals computed on the
        auxiliary model(s) and dataset. For the population attack,
        the auxiliary model is the target model itself, and the
        auxiliary dataset is a random split from the target model's
        training data.
        """
        # Load signals if they have been computed already
        # Otherwise, compute and save them
        self.member_signals = self.__load_or_compute_signals(signal_source=SignalSourceEnum.TARGET_MEMBER)
        self.non_member_signals = self.__load_or_compute_signals(signal_source=SignalSourceEnum.TARGET_NON_MEMBER)
        self.reference_signals = self.__load_or_compute_signals(signal_source=SignalSourceEnum.REFERENCE)

    def run_metric(self, fpr_tolerance_rate_list=None):
        """
        Function to run the metric on the target model and dataset.
        Args:
            fpr_tolerance_rate_list (optional): List of FPR tolerance values
            that may be used by the threshold function to compute the attack
            threshold for the metric.
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

            metric_result = MetricResult(metric_name="Population metric",
                                         predicted_labels=predictions,
                                         true_labels=true_labels,
                                         predictions_proba=None,
                                         signal_values=signal_values)

            metric_result_list.append(metric_result)

        return metric_result_list


class ShadowMetric(Metric):
    """
    Inherits the Metric class to perform the shadow membership inference attack
    which will be used as a metric for measuring privacy leakage of a target model.
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
        Constructor.
        
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

        self.reweight_samples = reweight_samples

        # Store the model to split mappings
        self.target_model_to_train_split_mapping = target_model_to_train_split_mapping
        self.target_model_to_test_split_mapping = target_model_to_test_split_mapping
        self.reference_model_to_train_split_mapping = reference_model_to_train_split_mapping
        self.reference_model_to_test_split_mapping = reference_model_to_test_split_mapping

        # Default values for all the model to split mappings

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
                if len(self.reference_info_source.models) == 1 and unique_dataset:
                    self.reference_model_to_train_split_mapping = [(0, 'train', '<default_input>', '<default_output>')]
                else:
                    self.reference_model_to_train_split_mapping = [
                        (0, f'train{k:03d}', '<default_input>', '<default_output>')
                        for k in range(len(self.reference_info_source.models))
                    ]
            if self.reference_model_to_test_split_mapping is None:
                if len(self.reference_info_source.models) == 1:
                    self.reference_model_to_test_split_mapping = [(0, 'test', '<default_input>', '<default_output>')]
                else:
                    self.reference_model_to_test_split_mapping = [
                        (0, f'test{k:03d}', '<default_input>', '<default_output>')
                        for k in range(len(self.reference_info_source.models))
                    ]

        # Variables used in prepare_metric and run_metric
        self.member_signals, self.non_member_signals = [], []
        self.reference_member_signals, self.reference_non_member_signals = [], []

    def __load_or_compute_signals(self, signal_source):
        """
        Private helper function to load signals if they have been computed already, or compute and save signals
        if they haven't.

        Args:
            signal_source: Signal source to determine which information source and mapping objects need to be used

        Returns:
            Signals computed using the specified information source and mapping object.
        """
        signal_filepath, info_source_obj, mapping_obj = None, None, None
        if signal_source == SignalSourceEnum.TARGET_MEMBER:
            signal_filepath = f'{self.logs_dirname}/{MetricEnum.SHADOW.value}_{TARGET_MEMBER_SIGNALS_FILENAME}'
            info_source_obj = self.target_info_source
            mapping_obj = self.target_model_to_train_split_mapping
        elif signal_source == SignalSourceEnum.TARGET_NON_MEMBER:
            signal_filepath = f'{self.logs_dirname}/{MetricEnum.SHADOW.value}_{TARGET_NON_MEMBER_SIGNALS_FILENAME}'
            info_source_obj = self.target_info_source
            mapping_obj = self.target_model_to_test_split_mapping
        elif signal_source == SignalSourceEnum.REFERENCE_MEMBER:
            signal_filepath = f'{self.logs_dirname}/{MetricEnum.SHADOW.value}_{REFERENCE_MEMBER_SIGNALS_FILENAME}'
            info_source_obj = self.reference_info_source
            mapping_obj = self.reference_model_to_train_split_mapping
        elif signal_source == SignalSourceEnum.REFERENCE_NON_MEMBER:
            signal_filepath = f'{self.logs_dirname}/{MetricEnum.SHADOW.value}_{REFERENCE_NON_MEMBER_SIGNALS_FILENAME}'
            info_source_obj = self.reference_info_source
            mapping_obj = self.reference_model_to_test_split_mapping

        signals = []
        if os.path.isfile(f'{signal_filepath}{NPZ_EXTENSION}'):
            with np.load(f'{signal_filepath}{NPZ_EXTENSION}', allow_pickle=True) as data:
                signals = data['arr_0'][()]
        else:
            # For each signal compute the response of both the model on the dataset according to the mapping
            for signal in self.signals:
                signals.append(
                    info_source_obj.get_signal(signal, mapping_obj)
                )
            # For the shadow metric we have a list of loss values
            signals = np.array(signals).flatten()
            np.savez(signal_filepath, signals)

        return signals

    def prepare_metric(self):
        """
        Function to prepare data needed for running the metric on the target model and dataset, using signals computed
        on the reference model(s) and dataset. For the shadow attack, the reference models will be a list of shadow
        models and the auxiliary dataset will contain the train-test splits of these models.
        """
        # Load signals if they have been computed already
        # Otherwise, compute and save them
        self.member_signals = self.__load_or_compute_signals(signal_source=SignalSourceEnum.TARGET_MEMBER)
        self.non_member_signals = self.__load_or_compute_signals(signal_source=SignalSourceEnum.TARGET_NON_MEMBER)
        self.reference_member_signals = self.__load_or_compute_signals(signal_source=SignalSourceEnum.REFERENCE_MEMBER)
        self.reference_non_member_signals = self.__load_or_compute_signals(
            signal_source=SignalSourceEnum.REFERENCE_NON_MEMBER
        )

    def run_metric(self, fpr_tolerance_rate_list=None):
        """
        Function to run the metric on the target model and dataset.
        Args:
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
            to compute the attack threshold for the metric.
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

        # Predict the membership status of samples in the target InformationSource
        predictions_proba = clf.predict_proba(np.concatenate([
            self.member_signals.reshape(-1, 1),
            self.non_member_signals.reshape(-1, 1)
        ]))
        predictions_label = np.argmax(predictions_proba, axis=1)
        predictions_proba = predictions_proba[:, 1]

        true_labels = [1] * len(self.member_signals) + [0] * len(self.non_member_signals)

        # Evaluate the power of this inference and display the result
        metric_result = MetricResult(
            metric_name="Shadow metric",
            predictions_proba=predictions_proba,
            predicted_labels=predictions_label,
            true_labels=true_labels,
            signal_values=x
        )

        return metric_result
