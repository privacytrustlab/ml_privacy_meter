from typing import List, Tuple

from privacy_meter.dataset import Dataset
from privacy_meter.model import Model
from privacy_meter.signal import Signal


class InformationSource:
    """
    Interface to dispatch Model objects, Dataset objects, and any additional objects required, to Signal objects.
    """

    def __init__(self,
                 models: List[Model],
                 datasets: List[Dataset],
                 default_model_to_split_mapping: List[Tuple[int, str, str, str]] = None
                 ):
        """Constructor

        Args:
            models: List of models to be queried.
            datasets: List of datasets to be queried.
            default_model_to_split_mapping: List of tuples, indicating how each model should query the dataset.
                More specifically, for model #i:
                default_model_to_split_mapping[i][0] contains the index of the dataset in the list,
                default_model_to_split_mapping[i][1] contains the name of the split,
                default_model_to_split_mapping[i][2] contains the name of the input feature,
                default_model_to_split_mapping[i][3] contains the name of the output feature.
                This can also be provided independently for each call, through the model_to_split_mapping argument of
                the get_signal function.
        """
        self.models = models
        self.datasets = datasets
        self.default_model_to_split_mapping = default_model_to_split_mapping

    def get_signal(self,
                   signal: Signal,
                   model_to_split_mapping: List[Tuple[int, str, str, str]] = None,
                   extra: dict = None
                   ):
        """Calls the signal object with the appropriate arguments: Model objects and Dataset objects specified at
        object instantiation, plus and any additional object required.

        Args:
            signal: The signal object to call
            model_to_split_mapping: List of tuples, indicating how each model should query the dataset.
                More specifically, for model #i:
                model_to_split_mapping[i][0] contains the index of the dataset in the list,
                model_to_split_mapping[i][1] contains the name of the split,
                model_to_split_mapping[i][2] contains the name of the input feature,
                model_to_split_mapping[i][3] contains the name of the output feature.
                This can also be provided once and for all at the instantiation of InformationSource, through the
                default_model_to_split_mapping argument.
            extra: Dictionary containing any additional parameter that should be passed to the signal object.

        Returns:
            The signal value.
        """

        # If no value of model_to_split_mapping is provided, use the default value
        if model_to_split_mapping is None:
            model_to_split_mapping = self.default_model_to_split_mapping
        # If no value of model_to_split_mapping is provided and no default value is set, raise an exception
        if model_to_split_mapping is None:
            raise TypeError(
                'At least one of self.default_model_to_split_mapping and model_to_split_mapping should be specified'
            )

        # Calls the signal object, and returns the value of the call
        return signal(models=self.models,
                      datasets=self.datasets,
                      model_to_split_mapping=model_to_split_mapping,
                      extra=extra)
