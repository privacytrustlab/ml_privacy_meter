from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from privacy_meter.dataset import Dataset
from privacy_meter.model import Model

########################################################################################################################
# SIGNAL CLASS
########################################################################################################################


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


########################################################################################################################
# DATASET_SAMPLE CLASS
########################################################################################################################


class DatasetSample(Signal):
    """
    Inherits from the Signal class, used to represent any type of signal that can be obtained from a Model and/or a
    Dataset.
    This particular class is used to get a given point from the Dataset.
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
            The sample point from the dataset.
        """

        dataset_index, split_name, input_feature, output_feature = model_to_split_mapping[
            extra["model_num"]]
        x = datasets[dataset_index].get_feature(split_name, input_feature)[
            extra["point_num"]]
        return x

########################################################################################################################
# MODEL_LOGIT CLASS
########################################################################################################################


class ModelLogits(Signal):
    """
    Inherits from the Signal class, used to represent any type of signal that can be obtained from a Model and/or a Dataset.
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
            dataset_index, split_name, input_feature, output_feature = model_to_split_mapping[
                k]
            x = datasets[dataset_index].get_feature(split_name, input_feature)
            # Compute the signal
            results.append(model.get_logits(x))
        return results

########################################################################################################################
# MODEL_INTERMEDIATE_OUTPUT CLASS
########################################################################################################################


class ModelIntermediateOutput(Signal):
    """
    Inherits from the Signal class, used to represent any type of signal that can be obtained from a Model and/or a
    Dataset.
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
            dataset_index, split_name, input_feature, output_feature = model_to_split_mapping[
                k]
            x = datasets[dataset_index].get_feature(split_name, input_feature)
            # Compute the signal
            results.append(model.get_intermediate_outputs(extra["layers"], x))
        return results

########################################################################################################################
# MODEL_LOSS CLASS
########################################################################################################################


class ModelLoss(Signal):
    """
    Inherits from the Signal class, used to represent any type of signal that can be obtained from a Model and/or a
    Dataset.
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
            dataset_index, split_name, input_feature, output_feature = model_to_split_mapping[
                k]
            x = datasets[dataset_index].get_feature(split_name, input_feature)

            # Check if output feature has been provided, else pass None
            if output_feature is not None:
                y = datasets[dataset_index].get_feature(
                    split_name, output_feature)
            else:
                y = None

            # Compute the signal for each sample
            results.append(model.get_loss(x, y))
        return results

########################################################################################################################
# MODEL_GRADIENT CLASS
########################################################################################################################


class ModelGradient(Signal):
    """
    Inherits from the Signal class, used to represent any type of signal that can be obtained from a Model and/or a
    Dataset.
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
            dataset_index, split_name, input_feature, output_feature = model_to_split_mapping[
                k]
            x = datasets[dataset_index].get_feature(split_name, input_feature)
            y = datasets[dataset_index].get_feature(split_name, output_feature)
            # Compute the signal for each sample
            for (sample_x, sample_y) in zip(x, y):
                xx, yy = np.expand_dims(
                    sample_x, axis=0), np.expand_dims(sample_y, axis=0)
                results.append(model.get_grad(xx, yy))
        return results


########################################################################################################################
# Group Information
########################################################################################################################


class GroupInfo(Signal):
    """
    Inherits from the Signal class, used to represent any type of signal that can be obtained from a Model and/or a
    Dataset.
    This particular class is used to get the group membership of data records.
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
                model_to_split_mapping[i][2] contains the name of the group feature
                This can also be provided once and for all at the instantiation of InformationSource, through the
                default_model_to_split_mapping argument.
            extra: Dictionary containing any additional parameter that should be passed to the signal object.

        Returns:
            The signal value.
        """

        results = []
        # Given the group membership for each dataset used by each model
        for k in range(len(models)):
            dataset_index, split_name, group_feature = model_to_split_mapping[k]
            g = datasets[dataset_index].get_feature(split_name, group_feature)
            results.append(g)
        return results


class ModelGradientNorm(Signal):
    """
    Inherits from the Signal class, used to represent any type of signal that can be obtained from a Model and/or a
    Dataset.
    This particular class is used to get the gradient norm of a model.
    """

    def __init__(self, is_features=True, layer_number=10) -> None:
        super().__init__()
        self.is_features = is_features
        self.layer_number = layer_number

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
            dataset_index, split_name, input_feature, output_feature = model_to_split_mapping[
                k]
            x = datasets[dataset_index].get_feature(split_name, input_feature)
            y = datasets[dataset_index].get_feature(split_name, output_feature)
            results.append(model.get_gradnorm(
                x, y, is_features=self.is_features, layer_number=self.layer_number))
        return results
