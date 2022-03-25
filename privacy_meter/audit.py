from typing import List, Union
from enum import Enum

from privacy_meter.information_source import InformationSource
from privacy_meter.information_source_signal import ModelLoss
from privacy_meter.metric import Metric, PopulationMetric, ShadowMetric
from privacy_meter.hypothesis_test import threshold_func


class MetricEnum(Enum):
    POPULATION = "population"
    SHADOW = "shadow"


class Audit:
    def __init__(
            self,
            metric: Union[MetricEnum, Metric],
            target_info_source: InformationSource = None,
            reference_info_source: InformationSource = None,
            fpr_tolerance_list: List = None
    ):
        self.metric_object = None
        if type(metric) == MetricEnum:
            # if the user wants to use default versions of metrics
            if metric == MetricEnum.POPULATION:
                self.metric_object = PopulationMetric(
                    target_info_source=target_info_source,
                    reference_info_source=reference_info_source,
                    signals=[ModelLoss()],
                    hypothesis_test_func=threshold_func
                )
            elif metric == MetricEnum.SHADOW:
                self.metric_object = ShadowMetric(
                    target_info_source=target_info_source,
                    reference_info_source=reference_info_source,
                    signals=[ModelLoss()],
                    hypothesis_test_func=None
                )
        else:
            # if the user wants to pass in their custom metric object
            self.metric_object = metric

        self.fpr_tolerance_list = fpr_tolerance_list

    def run(self):
        self.metric_object.run_metric(self.fpr_tolerance_list)

    def report(self):
        pass
