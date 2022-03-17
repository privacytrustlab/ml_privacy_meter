from itertools import product
import numpy as np
import tensorflow_datasets as tfds


class Dataset:

    def __init__(self,
                 data_dict: dict,
                 default_input: str,
                 default_output: str,
                 preproc_fn_dict: dict = None,
                 splits_indices_dict=None
                 ):
        """Constructor

        Args:
            data_dict: Contains the dataset as a dict
            default_input: The key of the data_dict that should be used by default to get the input of a model
            default_output: The key of the data_dict that should be used by default to get the expected output
                of a model
            preproc_fn_dict: Contains optional preprocessing functions for each feature
            splits_indices_dict: Contains optional lists of indices, if several intersecting subsets of the dataset
                were used
        """

        # Store parameters
        self.data_dict = data_dict
        self.default_input = default_input
        self.default_output = default_output
        self.preproc_fn_dict = preproc_fn_dict
        self.splits_indices_dict = splits_indices_dict

        # Store splits names and features names
        self.splits = list(self.data_dict)
        self.features = list(self.data_dict[self.splits[0]])

        # If preprocessing functions were passed as parameters, execute them
        if preproc_fn_dict is not None:
            self.preprocess()

    def preprocess(self):
        """
        Preprocessing function, executed by the constructor, based on the preproc_fn_dict attribute.
        """
        for (split, feature) in product(self.splits, self.features):
            if feature in list(self.preproc_fn_dict):
                fn = self.preproc_fn_dict[feature]
                self.data_dict[split][feature] = fn(self.data_dict[split][feature])

    def get_feature(self,
                    split_name: str,
                    feature_name: str,
                    indices: list = None
                    ):

        """Returns a specific feature from samples of a specific split.

        Args:
            split_name: Name of the split
            feature_name: Name of the feature
            indices: Optional list of indices. If not specified, the entire subset is returned.

        Returns:
            The requested feature, from samples of the requested split.
        """

        # Two placeholders can be used to trigger either the default input or the default output, as specified during
        # object creation
        if feature_name == '<default_input>':
            feature_name = self.default_input
        elif feature_name == '<default_output>':
            feature_name = self.default_output

        # If 'indices' is not specified, returns the entire array. Else just return those indices
        if indices is None:
            return self.data_dict[split_name][feature_name]
        else:
            return self.data_dict[split_name][feature_name][indices]

    def __str__(self):
        """
        Return a string describing the dataset
        """
        txt = [
            f'{" DATASET OBJECT ":=^48}',
            f'Splits            = {self.splits}',
            f'Features          = {self.features}',
            f'Default features  = {self.default_input} --> {self.default_output}',
            '=' * 48
        ]
        return '\n'.join(txt)
