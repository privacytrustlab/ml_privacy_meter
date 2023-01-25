import numpy as np
import pytest
import sys
sys.path.append('../')
from util import (get_split, load_models_by_conditions,
                  load_models_by_model_idx, load_models_with_data_idx_list,
                  load_models_without_data_idx_list)


# This test is for testing functions in datasets.py.


def test_get_split():
    all_index = list(range(10000))
    # Check for proper error raising
    with pytest.raises(NotImplementedError):
        get_split(all_index, [], 1, 'test')
    with pytest.raises(ValueError):
        get_split(all_index, [], 10001, 'uniform')
    with pytest.raises(ValueError):
        get_split(all_index, [1], 10000, 'no_overlapping')

    # Check for the correct type
    assert type(get_split(all_index, [], 10000, 'uniform')) == np.ndarray

    # Check for unique points in the selection
    r_list = get_split(all_index, [], 10000, 'uniform')
    assert len(r_list) == 10000
    assert len(np.unique(r_list)) == 10000

    # Check for unique points in the selection
    r_list = get_split(all_index, [1], 9999, 'no_overlapping')
    assert 1 not in r_list  # Exclude points already in selection
    assert len(r_list) == 9999  # Check for correct number of points
    assert len(np.unique(r_list)) == 9999  # Check for uniqueness

    # Test that randomness works
    a_list = get_split(all_index, [1], 9999, 'no_overlapping')
    b_list = get_split(all_index, [1], 9999, 'no_overlapping')
    assert np.mean(a_list == b_list) != 1


test_data = [({'condition1': 'value1', 'condition2': 'value2'},
              {'model_metadata': {1: {'condition1': 'value1', 'condition2': 'value2'},
                                  2: {'condition3': 'value3', 'condition4': 'value4'},
                                  3: {'condition1': 'value1', 'condition3': 'value3'}}}, 2, [1], []),
             ({'condition3': 'value3', 'condition4': 'value4'},
              {'model_metadata': {1: {'condition1': 'value1', 'condition2': 'value2'},
                                  2: {'condition3': 'value3', 'condition4': 'value4'},
                                  3: {'condition1': 'value1', 'condition3': 'value3'}}}, 1, [], [2]),
             ({}, {'model_metadata': {1: {'condition1': 'value1', 'condition2': 'value2'},
                                      2: {'condition3': 'value3', 'condition4': 'value4'},
                                      3: {'condition1': 'value1', 'condition3': 'value3'}}}, 3, [2], [])]


@pytest.mark.parametrize('conditions, model_metadata_list, \
                          num_models, target_idx, expected', test_data)
def test_load_models_by_conditions(conditions, model_metadata_list, num_models, target_idx, expected):
    # Act
    output = load_models_by_conditions(
        model_metadata_list, conditions, num_models, target_idx)
    # Assert
    assert output == expected


model_metadata_list = {
    "model_metadata": {
        1: {
            "model_idx": 6,
            "train_split": [1, 2, 3]
        },
        2: {
            "model_idx": 7,
            "train_split": [1, 4, 7]
        },
        3: {
            "model_idx": 8,
            "train_split": [1, 5, 8]
        },
    }
}


@pytest.mark.parametrize("data_idx_list, expected_matched_idx",
                         [([1, 5], [3])])
def test_load_models_with_data_idx_list(data_idx_list, expected_matched_idx):
    assert load_models_with_data_idx_list(
        model_metadata_list, data_idx_list) == expected_matched_idx


model_metadata_list = {
    "model_metadata": {
        1: {
            "model_idx": 6,
            "train_split": [1, 2, 3]
        },
        2: {
            "model_idx": 7,
            "train_split": [1, 4, 7]
        },
        3: {
            "model_idx": 8,
            "train_split": [1, 5, 8]
        },
    }
}


@pytest.mark.parametrize("model_idx_list, expected_matched_idx", [
    ([6], [1]),
    ([7], [2]),
    ([8], [3]),
    ([6, 7], [1, 2]),
    ([6, 8], [1, 3]),
    ([7, 8], [2, 3])
])
def test_load_models_by_model_idx(model_idx_list, expected_matched_idx):
    assert load_models_by_model_idx(
        model_metadata_list, model_idx_list) == expected_matched_idx


model_metadata_list = {
    "model_metadata": {
        1: {
            "model_idx": 6,
            "train_split": [1, 2, 3]
        },
        2: {
            "model_idx": 7,
            "train_split": [1, 4, 7]
        },
        3: {
            "model_idx": 8,
            "train_split": [1, 5, 8]
        },
    }
}


@pytest.mark.parametrize("data_idx_list, expected_matched_idx", [
    ([2], [2, 3]),
    ([4], [1, 3]),
    ([5], [1, 2]),
    ([2, 4], [3]),
    ([2, 5], [2]),
    ([4, 5], [1])
])
def test_load_models_without_data_idx_list(data_idx_list, expected_matched_idx):
    assert load_models_without_data_idx_list(
        model_metadata_list, data_idx_list) == expected_matched_idx
