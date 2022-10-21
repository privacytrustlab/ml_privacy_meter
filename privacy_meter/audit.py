import os
from datetime import datetime
from typing import List, Union

from privacy_meter.constants import MetricEnum, InferenceGame
from privacy_meter.hypothesis_test import threshold_func
from privacy_meter.information_source import InformationSource
from privacy_meter.information_source_signal import ModelLoss
from privacy_meter.metric import Metric, PopulationMetric, ShadowMetric, ReferenceMetric,GroupPopulationMetric
from privacy_meter.metric_result import MetricResult


class Audit:
    """
    This class orchestrates how the Metric objects and the InformationSource objects interact with one
    another. The three steps of using this class are 1) initialization 2) audit.prepare() 3) audit.run().
    """

    def __init__(
            self,
            metrics: Union[Union[MetricEnum, Metric], List[Union[MetricEnum, Metric]]],
            inference_game_type: InferenceGame,
            target_info_sources: Union[InformationSource, List[InformationSource]] = None,
            reference_info_sources: Union[InformationSource, List[InformationSource]] = None,
            fpr_tolerances: Union[float, List[float]] = None,
            logs_directory_names: Union[str, List[str]] = None
    ):
        """
        Constructor

        Args:
            metrics: Metric object or list of Metric objects to be used for the audit.
            inference_game_type: The type of inference game being played: average privacy loss of a training algorithm,
                privacy loss of a model, privacy loss of a data record, or worst-case privacy loss of a training
                algorithm.
            target_info_sources: InformationSource object(s), containing the Model(s) that the metric will be performed
                on, and the corresponding Dataset(s).
            reference_info_sources: InformationSource object(s), containing the Model(s) that the metric will be fitted
                on, and the corresponding Dataset(s).
            fpr_tolerances: FPR tolerance value(s) to be used by the audit.
            logs_directory_names: Path(s) to logging directory(ies).
        """

        self.metrics = metrics
        self.inference_game_type = inference_game_type
        self.target_info_sources = target_info_sources
        self.reference_info_sources = reference_info_sources
        self.fpr_tolerances = fpr_tolerances
        self.logs_directory_names = logs_directory_names

        self.__init_lists()
        self.__init_logs_directories()
        self.__init_metric_objects()

    def __init_logs_directories(self):
        """
        Private function part of the initialization process, to specify default logging directory(ies), and create them
        if necessary.
        """
        if self.logs_directory_names is None:
            self.logs_directory_names = []
            for i in range(len(self.metrics)):
                logs_dirname = os.path.join(
                    os.getcwd(),
                    datetime.now().strftime(f'log_%Y-%m-%d_%H-%M-%S-{i:03d}')
                )
                os.mkdir(logs_dirname)
                self.logs_directory_names.append(logs_dirname)
        else:
            for path in self.logs_directory_names:
                if not os.path.isdir(path):
                    os.mkdir(path)

    def __init_metric_objects(self):
        """
        Private function part of the initialization process, to create Metric objects from MetricEnum ones.
        """
        self.metric_objects = []
        for k, metric in enumerate(self.metrics):
            if type(metric) == MetricEnum:
                # If the user wants to use default versions of metrics
                if metric == MetricEnum.POPULATION:
                    self.metric_objects.append(PopulationMetric(
                        target_info_source=self.target_info_sources[k],
                        reference_info_source=self.reference_info_sources[k],
                        signals=[ModelLoss()],
                        hypothesis_test_func=threshold_func,
                        logs_dirname=self.logs_directory_names[k]
                    ))
                elif metric == MetricEnum.SHADOW:
                    self.metric_objects.append(ShadowMetric(
                        target_info_source=self.target_info_sources[k],
                        reference_info_source=self.reference_info_sources[k],
                        signals=[ModelLoss()],
                        hypothesis_test_func=None,
                        logs_dirname=self.logs_directory_names[k]
                    ))
                elif metric == MetricEnum.REFERENCE:
                    self.metric_objects.append(ReferenceMetric(
                        target_info_source=self.target_info_sources[k],
                        reference_info_source=self.reference_info_sources[k],
                        signals=[ModelLoss()],
                        hypothesis_test_func=threshold_func,
                        logs_dirname=self.logs_directory_names[k]
                    ))
                elif metric == MetricEnum.GROUPPOPULATION:
                    self.metric_objects.append(GroupPopulationMetric(
                        target_info_source=self.target_info_sources[k],
                        reference_info_source=self.reference_info_sources[k],
                        signals=[ModelLoss()],
                        hypothesis_test_func=threshold_func,
                        logs_dirname=self.logs_directory_names[k]
                    ))

            else:
                # If the user wants to pass in their custom metric object
                metric.logs_dirname = self.logs_directory_names[k]
                self.metric_objects.append(metric)

    def __init_lists(self):
        """
        Private function part of the initialization process, to have consistent argument types.
        """
        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics]
        if not isinstance(self.target_info_sources, list):
            self.target_info_sources = [self.target_info_sources]
        if not isinstance(self.reference_info_sources, list):
            self.reference_info_sources = [self.reference_info_sources]
        if not isinstance(self.fpr_tolerances, list):
            self.fpr_tolerances = [self.fpr_tolerances]
        if self.logs_directory_names is not None and not isinstance(self.logs_directory_names, list):
            self.logs_directory_names = [self.logs_directory_names]

    def prepare(self,max_val=100,min_val=0):
        """
        Core function that should be called after the initialization and before the audit.run() function. Runs the
        prepare_metric function of all metric objects, which computes (or loads from memory) the signals required for
        the membership inference algorithms.
        """
        for i in range(len(self.metric_objects)):
            self.metric_objects[i].prepare_metric(max_val=max_val,min_val=min_val)

    def run(self) -> List[MetricResult]:
        """
        Core function that should be called after the audit.prepare() function. This actually runs the metrics'
        membership inference algorithms.

        Returns:
            A list of MetricResult objects (one per metric)
        """
        print(f"Results are stored in: {self.logs_directory_names}")
        return [self.metric_objects[i].run_metric(self.fpr_tolerances) for i in range(len(self.metric_objects))]
