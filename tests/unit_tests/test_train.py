
import sys
sys.path.append('../')

import torch

from dataset import get_dataset
from models import AlexNet, Net, get_model
from train import inference, train


# This test is for testing functions in train.py.
def test_train_on_cnn():
    dataset = get_dataset('cifar10', "data")

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
