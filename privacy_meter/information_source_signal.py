from abc import ABC, abstractmethod

from typing import List, Tuple

import numpy as np

from privacy_meter.dataset import Dataset
from privacy_meter.model import Model


class Signal(ABC):
    """
    Abstract class, representing any type of signal that can be obtained from a Model and/or a Dataset.
    """

    @abstractmethod
    def __call__(self,
                 models: List[Model],
                 datasets: List[Dataset],
                 model_to_split_mapping: List[Tuple[int, str, str, str]],
                 extra: dict
                 ):
        """Built-in call method.

        Args:
            models: List of models that can be queried.
            datasets: List of datasets that can be queried.
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
        pass


class ModelOutput(Signal):
    """
    Inherits of the Signal class, used to represent any type of signal that can be obtained from a Model and/or a Dataset.
    This particular class is used to get the output of a model.
    """

    def __call__(self,
                 models: List[Model],
                 datasets: List[Dataset],
                 model_to_split_mapping: List[Tuple[int, str, str, str]],
                 extra: dict
                 ):
        """Built-in call method.

        Args:
            models: List of models that can be queried.
            datasets: List of datasets that can be queried.
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

        results = []
        # Compute the signal for each model
        for k, model in enumerate(models):
            # Extract the features to be used
            dataset_index, split_name, input_feature, output_feature = model_to_split_mapping[k]
            x = datasets[dataset_index].get_feature(split_name, input_feature)
            # Compute the signal
            results.append(model.get_outputs(x))
        return results


class ModelIntermediateOutput(Signal):
    """
    Inherits of the Signal class, used to represent any type of signal that can be obtained from a Model and/or a Dataset.
    This particular class is used to get the value of an intermediate layer of model.
    """

    def __call__(self,
                 models: List[Model],
                 datasets: List[Dataset],
                 model_to_split_mapping: List[Tuple[int, str, str, str]],
                 extra: dict
                 ):
        """Built-in call method.

        Args:
            models: List of models that can be queried.
            datasets: List of datasets that can be queried.
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

        if 'layers' not in list(extra):
            raise TypeError('extra parameter "layers" is required')

        results = []
        # Compute the signal for each model
        for k, model in enumerate(models):
            # Extract the features to be used
            dataset_index, split_name, input_feature, output_feature = model_to_split_mapping[k]
            x = datasets[dataset_index].get_feature(split_name, input_feature)
            # Compute the signal
            results.append(model.get_intermediate_outputs(extra["layers"], x))
        return results


class ModelLoss(Signal):
    """
    Inherits of the Signal class, used to represent any type of signal that can be obtained from a Model and/or a Dataset.
    This particular class is used to get the loss of a model.
    """

    def __call__(self,
                 models: List[Model],
                 datasets: List[Dataset],
                 model_to_split_mapping: List[Tuple[int, str, str, str]],
                 extra: dict
                 ):
        """Built-in call method.

        Args:
            models: List of models that can be queried.
            datasets: List of datasets that can be queried.
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

        results = []
        # Compute the signal for each model
        for k, model in enumerate(models):
            # Extract the features to be used
            dataset_index, split_name, input_feature, output_feature = model_to_split_mapping[k]
            x = datasets[dataset_index].get_feature(split_name, input_feature)
            y = datasets[dataset_index].get_feature(split_name, output_feature)
            # Compute the signal for each sample
            for (sample_x, sample_y) in zip(x, y):
                xx, yy = np.expand_dims(sample_x, axis=0), np.expand_dims(sample_y, axis=0)
                results.append(model.get_loss(xx, yy))
        return results


class ModelGradient(Signal):
    """
    Inherits of the Signal class, used to represent any type of signal that can be obtained from a Model and/or a Dataset.
    This particular class is used to get the gradient of a model.
    """

    def __call__(self,
                 models: List[Model],
                 datasets: List[Dataset],
                 model_to_split_mapping: List[Tuple[int, str, str, str]],
                 extra: dict
                 ):
        """Built-in call method.

        Args:
            models: List of models that can be queried.
            datasets: List of datasets that can be queried.
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

        results = []
        # Compute the signal for each model
        for k, model in enumerate(models):
            # Extract the features to be used
            dataset_index, split_name, input_feature, output_feature = model_to_split_mapping[k]
            x = datasets[dataset_index].get_feature(split_name, input_feature)
            y = datasets[dataset_index].get_feature(split_name, output_feature)
            # Compute the signal
            results.append(model.get_grad(x, y))
        return results
