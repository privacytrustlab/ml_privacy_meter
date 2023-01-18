
from util import get_split
from train import *
from models import *
from dataset import *
import pytest
import sys
sys.path.append('../')


# This test is for testing all the underlying functions.
# All the data and models used for testing will be saved in log_dir.

log_dir = 'test_cases'


# Test datasets
def test_get_dataset_on_cifar():
    dataset = get_dataset('cifar10', log_dir)
    assert len(dataset) == 60000
    assert os.path.exists(f"{log_dir}/cifar10.pkl")
    for i in range(10):
        assert (np.array(dataset.targets) == i).sum() == 6000


def test_get_dataset_on_other_datasets():
    with pytest.raises(NotImplementedError):
        get_dataset('mnist', log_dir)


def test_get_cifar10_subset():
    dataset = get_dataset('cifar10', 'data')
    selected_index = list(np.random.choice(range(60000), 100))
    data = get_cifar10_subset(dataset, selected_index, is_tensor=False)
    assert type(data) == torchvision.datasets.cifar.CIFAR10
    assert len(data) == 100
    assert type(data.data) == np.ndarray
    assert type(data.targets) == list


def test_get_cifar10_subset_tensor():
    dataset = get_dataset('cifar10', log_dir)
    selected_index = list(np.random.choice(range(60000), 80))
    data = get_cifar10_subset(dataset, selected_index, is_tensor=True)
    assert type(data) == torchvision.datasets.cifar.CIFAR10
    assert len(data) == 80
    assert type(data.data) == torch.Tensor
    assert type(data.targets) == torch.Tensor


def test_get_cifar10_subset_index():
    dataset = get_dataset('cifar10', log_dir)
    with pytest.raises(AssertionError):
        get_cifar10_subset(dataset, [-1], is_tensor=False)

    with pytest.raises(AssertionError):
        get_cifar10_subset(dataset, [67000], is_tensor=False)

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
        'lr': 0.01,
        'wd': 0,
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
    dataset = get_dataset('cifar10', log_dir)
    configs = {
        'epochs': 1,
        'optimizer': 'Adam',
        'lr': 0.01,
        'wd': 0,
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
    dataset = get_dataset('cifar10', log_dir)
    model = get_model('CNN')
    o_w = model.state_dict()
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1000, shuffle=False, num_workers=2)
    loss_b, acc_b = inference(model, data_loader, 'cuda:1', is_train=False)

    updated_w = model.state_dict()
    # Make sure that the original model is not updated. Inference function does not change the model parameters.
    for key in updated_w:
        assert torch.equal(updated_w[key], o_w[key])

    configs = {
        'epochs': 1,
        'optimizer': 'Adam',
        'lr': 0.01,
        'wd': 0,
        'momentum': 0,
        'device': 'cuda:1',
    }

    updated_model = train(model, data_loader, configs)
    loss_a, acc_a = inference(
        updated_model, data_loader, 'cuda:1', is_train=False)
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
