from itertools import product
from typing import Union, Dict

import numpy as np


class Dataset:
    """
    Wrapper around a dictionary-like formatted dataset, with functions to run preprocessing, to define default
    input/output features, and to split a dataset easily.
    """

    def __init__(self,
                 data_dict: dict,
                 default_input: str,
                 default_output: str,
                 preproc_fn_dict: dict = None,
                 preprocessed: bool = False
                 ):
        """Constructor

        Args:
            data_dict: Contains the dataset as a dict
            default_input: The key of the data_dict that should be used by default to get the input of a model
            default_output: The key of the data_dict that should be used by default to get the expected output
                of a model
            preproc_fn_dict: Contains optional preprocessing functions for each feature
            preprocessed: Indicates if the preprocessing of preproc_fn_dict has already been applied
        """

        # Store parameters
        self.data_dict = data_dict
        self.default_input = default_input
        self.default_output = default_output
        self.preproc_fn_dict = preproc_fn_dict

        # Store splits names and features names
        self.splits = list(self.data_dict)
        self.features = list(self.data_dict[self.splits[0]])

        # If preprocessing functions were passed as parameters, execute them
        if not preprocessed and preproc_fn_dict is not None:
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

    def subdivide(self,
                  num_splits: int,
                  split_names: list = None,
                  method: str = 'independent',
                  split_size: Union[int, Dict[str, int]] = None,
                  delete_original: bool = False,
                  in_place: bool = True,
                  return_results: bool = False
                  ):
        """Subdivides the splits contained in split_names into sub-splits, e.g. for shadow model training.

        Args:
            num_splits: Number of sub-splits per original split.
            split_names: The splits to subdivide (e.g. train and test). By default, includes all splits.
            method: Either independent or random. If method is independent, then the sub-splits are a partition of the
                original split (i.e. they contain the entire split without repetition). If method is random, then each
                sub-split is a random subset of the original split (i.e. some samples might be missing or repeated). If
                method is hybrid, then each sub-split is a random subset of the original split, with the guarantee that
                the 1st one is not overlapping with the others.
            split_size: If method is random, this is the size of one split (ignored if method is independent). Can
                either be an integer, or a dictionary of integer (one per split).
            delete_original: Indicates if the original split should be deleted.
            in_place: Indicates if the new splits should be included in the parent object or not
            return_results: Indicates if the new splits should be returned or not

        Returns:
            If in_place, a list of new Dataset objects, with the sub-splits. Otherwise, nothing, as the results are
            stored in self.data_dict.
        """

        # By default, includes all splits.
        if split_names is None:
            split_names = self.splits

        # List of results if in_place is False
        new_datasets_dict = [{} for _ in range(num_splits)]

        for split in split_names:

            if split_size is not None:
                parsed_split_size = split_size if isinstance(split_size, int) else split_size[split]

            # If method is random, then each sub-split is a random subset of the original split.
            if method == 'random':
                assert split_size is not None, 'Argument split_size is required when method is "random" or "hybrid"'
                indices = np.random.randint(self.data_dict[split][self.features[0]].shape[0], size=(num_splits, parsed_split_size))

            # If method is independent, then the sub-splits are a partition of the original split.
            elif method == 'independent':
                indices = np.arange(self.data_dict[split][self.features[0]].shape[0])
                np.random.shuffle(indices)
                indices = np.array_split(indices, num_splits)

            # If method is hybrid, then each sub-split is a random subset of the original split, with the guarantee that
            # the 1st one is not overlapping with the others
            elif method == 'hybrid':
                assert split_size is not None, 'Argument split_size is required when method is "random" or "hybrid"'
                available_indices = np.arange(self.data_dict[split][self.features[0]].shape[0])
                indices_a = np.random.choice(available_indices, size=(1, parsed_split_size), replace=False)
                available_indices = np.setdiff1d(available_indices, indices_a.flatten())
                indices_b = np.random.choice(available_indices, size=(num_splits-1, parsed_split_size), replace=True)
                indices = np.concatenate((indices_a, indices_b))

            else:
                raise ValueError(f'Split method "{method}" does not exist.')

            for split_n in range(num_splits):
                # Fill the dictionary if in_place is True
                if in_place:
                    self.data_dict[f'{split}{split_n:03d}'] = {}
                    for feature in self.features:
                        self.data_dict[f'{split}{split_n:03d}'][feature] = self.data_dict[split][feature][indices[split_n]]
                # Create new dictionaries if return_results is True
                if return_results:
                    new_datasets_dict[split_n][f'{split}'] = {}
                    for feature in self.features:
                        new_datasets_dict[split_n][f'{split}'][feature] = self.data_dict[split][feature][indices[split_n]]

            # delete_original indicates if the original split should be deleted.
            if delete_original:
                del self.data_dict[split]

        # Update the list of splits
        self.splits = list(self.data_dict)

        # Return new datasets if return_results is True
        if return_results:
            return [
                Dataset(
                    data_dict=new_datasets_dict[i],
                    default_input=self.default_input,
                    default_output=self.default_output,
                    preproc_fn_dict=self.preproc_fn_dict,
                    preprocessed=True
                )
                for i in range(num_splits)
            ]

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
