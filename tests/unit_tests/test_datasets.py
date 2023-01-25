
import sys
sys.path.append('../../')
from dataset import get_dataset, get_dataset_subset
import pytest
import torch
import numpy as np
import os
from models import Net, AlexNet


# This test is for testing functions in datasets.py.

# Test datasets
def test_get_dataset_on_cifar():
    dataset = get_dataset('cifar10', 'data')
    assert len(dataset) == 60000
    assert os.path.exists("data/cifar10.pkl")
    for i in range(10):
        assert (np.array(dataset.targets) == i).sum() == 6000


def test_get_dataset_on_other_datasets():
    with pytest.raises(NotImplementedError):
        get_dataset('mnist', "data")


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
