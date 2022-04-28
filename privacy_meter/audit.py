import os
from datetime import datetime

from typing import List, Union

from privacy_meter.constants import MetricEnum, InferenceGame
from privacy_meter.information_source import InformationSource
from privacy_meter.information_source_signal import ModelLoss
from privacy_meter.metric import Metric, PopulationMetric, ShadowMetric, ReferenceMetric
from privacy_meter.hypothesis_test import threshold_func


METRIC_TYPE = Union[MetricEnum, Metric]


class Audit:

    def __init__(
            self,
            metrics: Union[METRIC_TYPE, List[METRIC_TYPE]],
            inference_game_type: InferenceGame,
            target_info_sources: Union[InformationSource, List[InformationSource]] = None,
            reference_info_sources: Union[InformationSource, List[InformationSource]] = None,
            fpr_tolerances: Union[float, List[float]] = None,
            logs_directory_names: Union[str, List[str]] = None
    ):

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
        self.metric_objects = []
        for k, metric in enumerate(self.metrics):
            if type(metric) == MetricEnum:
                # if the user wants to use default versions of metrics
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
            else:
                # if the user wants to pass in their custom metric object
                metric.logs_dirname = self.logs_directory_names[k]
                self.metric_objects.append(metric)

    def __init_lists(self):
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

    def prepare(self):
        for i in range(len(self.metric_objects)):
            self.metric_objects[i].prepare_metric()

    def run(self):
        print(f"Results are stored in: {self.logs_directory_names}")
        return [self.metric_objects[i].run_metric(self.fpr_tolerances) for i in range(len(self.metric_objects))]

    def report(self):
        pass
