
import sys
sys.path.append('../')
from train import train, inference
from dataset import get_dataset, get_dataset_subset
from util import (get_split, load_models_by_conditions,
                  load_models_with_data_idx_list, load_models_by_model_idx,
                  load_models_without_data_idx_list)
from models import get_model
import pytest
import torch
import numpy as np
import os
from models import Net, AlexNet


# This test is for testing all the underlying functions.
# All the data and models used for testing will be saved in log_dir.

log_dir = 'test_cases'


# Test datasets
def test_get_dataset_on_cifar():
    dataset = get_dataset('cifar10', 'data')
    assert len(dataset) == 60000
    assert os.path.exists(f"{log_dir}/cifar10.pkl")
    for i in range(10):
        assert (np.array(dataset.targets) == i).sum() == 6000


def test_get_dataset_on_other_datasets():
    with pytest.raises(NotImplementedError):
        get_dataset('mnist', log_dir)


def test_get_dataset_subset_tensor():
    dataset = get_dataset('cifar10', 'data')
    selected_index = list(np.random.choice(range(60000), 80))
    data, targets = get_dataset_subset(dataset, selected_index)
    assert type(data) == torch.Tensor
    assert len(data) == 80
    assert type(targets) == torch.Tensor


def test_get_dataset_subset_index():
    dataset = get_dataset('cifar10', 'data')
    with pytest.raises(AssertionError):
        get_dataset_subset(dataset, [-1])

    with pytest.raises(AssertionError):
        get_dataset_subset(dataset, [67000])


# Test models
def test_get_model():
    model = get_model('CNN')
    assert type(model) == Net
    model = get_model('alexnet')
    assert type(model) == AlexNet

    with pytest.raises(NotImplementedError):
        get_model('LR')


def test_train_on_cnn():
    dataset = get_dataset('cifar10', log_dir)

    # Set the configuration for training
    configs = {
        'epochs': 1,
        'optimizer': 'Adam',
        'learning_rate': 0.01,
        'weight_decay': 0,
        'momentum': 0,
    }

    model = get_model('CNN')
    original_param_dict = model.state_dict()

    # Get the dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1000, shuffle=False, num_workers=2)

    # Set the device for training
    configs['device'] = 'cuda:0'

    # Train the model
    updated_model = train(model, train_loader, configs)

    # Make sure that the returned model is a model class object
    assert type(updated_model) == Net

    # Get the model's updated parameters
    updated_param_dict = updated_model.state_dict()

    # Make sure that the original model is updated
    for key in updated_param_dict:
        assert torch.equal(
            updated_param_dict[key], original_param_dict[key]) == False


def test_train_on_alexnet():
    dataset = get_dataset('cifar10', 'data')
    configs = {
        'epochs': 1,
        'optimizer': 'Adam',
        'learning_rate': 0.01,
        'weight_decay': 0,
        'momentum': 0,
    }
    model = get_model('alexnet')
    o_w = model.state_dict()
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1000, shuffle=False, num_workers=2)
    configs['device'] = 'cuda:0'
    updated_model = train(model, train_loader, configs)
    assert type(updated_model) == AlexNet
    updated_w = updated_model.state_dict()
    # Make sure that the original model is updated. Train function returns a different model.
    for key in updated_w:
        assert torch.equal(updated_w[key], o_w[key]) == False


# Test inference function and the effect of the training
def test_inference():
    dataset = get_dataset('cifar10', 'data')
    model = get_model('CNN')
    o_w = model.state_dict()
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1000, shuffle=False, num_workers=2)
    loss_b, acc_b = inference(model, data_loader, 'cuda:1')

    updated_w = model.state_dict()
    # Make sure that the original model is not updated. Inference function does not change the model parameters.
    for key in updated_w:
        assert torch.equal(updated_w[key], o_w[key])

    configs = {
        'epochs': 1,
        'optimizer': 'Adam',
        'learning_rate': 0.01,
        'weight_decay': 0,
        'momentum': 0,
        'device': 'cuda:1',
    }

    updated_model = train(model, data_loader, configs)
    loss_a, acc_a = inference(
        updated_model, data_loader, 'cuda:1')
    assert loss_a < loss_b, acc_a > acc_b


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