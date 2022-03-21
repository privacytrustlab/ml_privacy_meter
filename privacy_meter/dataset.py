from itertools import product
import numpy as np
import tensorflow_datasets as tfds


class Dataset:

    def __init__(self,
                 data_dict: dict,
                 default_input: str,
                 default_output: str,
                 preproc_fn_dict: dict = None
                 ):
        """Constructor

        Args:
            data_dict: Contains the dataset as a dict
            default_input: The key of the data_dict that should be used by default to get the input of a model
            default_output: The key of the data_dict that should be used by default to get the expected output
                of a model
            preproc_fn_dict: Contains optional preprocessing functions for each feature
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

    def subdivide(self,
                  num_splits: int,
                  split_names: list = None,
                  method: str = 'independent',
                  split_size: int = None,
                  delete_original: bool = False
                  ):
        """Subdivides the splits contained in split_names into sub-splits, e.g. for shadow model training.

        Args:
            num_splits: Number of sub-splits per original split.
            split_names: The splits to subdivide (e.g. train and test). By default, includes all splits.
            method: Either independent or random. If method is independent, then the sub-splits are a partition of the
                original split (i.e. they contain the entire split without repetition). If method is random, then each
                sub-split is a random subset of the original split (i.e. some samples might be missing or repeated).
            split_size: If method is random, this is the size of one split (ignored if method is independent).
            delete_original: Indicates if the original split should be deleted.

        Returns:
            Nothing; results are stored in self.data_dict.
        """

        # By default, includes all splits.
        if split_names is None:
            split_names = self.splits

        for split in split_names:
            for feature in self.features:

                # If method is random, then each sub-split is a random subset of the original split.
                if method == 'random':
                    assert split_size is not None
                    indices = np.random.randint(self.data_dict[split][feature].shape[0], size=(num_splits, split_size))
                    for split_n in range(num_splits):
                        # Initialize the dictionary if necessary.
                        if f'{split}{split_n:03d}' not in list(self.data_dict):
                            self.data_dict[f'{split}{split_n:03d}'] = {}
                        # Fill the dictionary.
                        self.data_dict[f'{split}{split_n:03d}'][feature] = self.data_dict[split][feature][indices[split_n]]

                # If method is independent, then the sub-splits are a partition of the original split.
                elif method == 'independent':
                    arr = np.array_split(self.data_dict[split][feature], num_splits)
                    for split_n in range(num_splits):
                        # Initialize the dictionary if necessary.
                        if f'{split}{split_n:03d}' not in list(self.data_dict):
                            self.data_dict[f'{split}{split_n:03d}'] = {}
                        # Fill the dictionary.
                        self.data_dict[f'{split}{split_n:03d}'][feature] = arr[split_n]

                else:
                    raise ValueError(f'Split method "{method}" does not exist.')

            # delete_original indicates if the original split should be deleted.
            if delete_original:
                del self.data_dict[split]

        # Update the list of splits
        self.splits = list(self.data_dict)

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
