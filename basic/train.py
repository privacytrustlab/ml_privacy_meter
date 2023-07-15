"""This file contains functions for training and testing the model."""
import time
from ast import Tuple

import torch
from torch import nn
from util import get_optimizer


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    configs: dict,
    test_loader: torch.utils.data.DataLoader = None,
):
    """Train the model based on on the train loader
    Args:
        model(nn.Module): Model for evaluation.
        train_loader(torch.utils.data.DataLoader): Data loader for training.
        configs (dict): Configurations for training.
    Return:
        nn.Module: Trained model.
    """
    # Get the device for training
    device = configs.get("device", "cpu")

    # Set the model to the device
    model.to(device)
    model.train()
    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, configs)
    # Get the number of epochs for training
    epochs = configs.get("epochs", 1)

    # Loop over each epoch
    for epoch_idx in range(epochs):
        start_time = time.time()
        train_loss, train_acc = 0, 0
        # Loop over the training set
        model.train()
        for data, target in train_loader:
            # Move data to the device
            data, target = data.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )
            # Cast target to long tensor
            target = target.long()

            # Set the gradients to zero
            optimizer.zero_grad(set_to_none=True)

            # Get the model output
            output = model(data)
            # Calculate the loss
            loss = criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            train_acc += pred.eq(target.data.view_as(pred)).sum()
            # Perform the backward pass
            loss.backward()
            # Take a step using optimizer
            optimizer.step()
            # Add the loss to the total loss
            train_loss += loss.item()

        print(f"Epoch: {epoch_idx+1}/{epochs} |", end=" ")
        print(f"Train Loss: {train_loss/len(train_loader):.8f} ", end=" ")
        print(f"Train Acc: {float(train_acc)/len(train_loader.dataset):.8f} ", end=" ")

        if test_loader is not None:
            model.eval()
            with torch.no_grad():
                test_loss, test_acc = 0, 0
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    # Cast target to long tensor
                    target = target.long()
                    # Computing output and loss
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    # Computing accuracy
                    pred = output.data.max(1, keepdim=True)[1]
                    test_acc += pred.eq(target.data.view_as(pred)).sum()

            print(f"Test Loss: {test_loss/len(test_loader):.8f} ", end=" ")
            print(f"Test Acc: {float(test_acc)/len(test_loader.dataset):.8f} ", end=" ")
        print(f"One step uses {time.time() - start_time:.2f} seconds")

    # Move the model back to the CPU
    model.to("cpu")

    # Return the model
    return model


# Test Function
def inference(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str
) -> Tuple(float, float):
    """Evaluate the model performance on the test loader
    Args:
        model (torch.nn.Module): Model for evaluation
        loader (torch.utils.data.DataLoader): Data Loader for testing
        device (str): GPU or CPU
    Return:
        loss (float): Loss for the given model on the test dataset.
        acc (float): Accuracy for the given model on the test dataset.
    """

    # Setting model to eval mode and moving to specified device
    model.eval()
    model.to(device)

    # Assigning variables for computing loss and accuracy
    loss, acc, criterion = 0, 0, nn.CrossEntropyLoss()

    # Disable gradient calculation to save memory
    with torch.no_grad():
        for data, target in loader:
            # Moving data and target to the device
            data, target = data.to(device), target.to(device)
            # Cast target to long tensor
            target = target.long()

            # Computing output and loss
            output = model(data)
            loss += criterion(output, target).item()

            # Computing accuracy
            pred = output.data.max(1, keepdim=True)[1]
            acc += pred.eq(target.data.view_as(pred)).sum()

        # Averaging the losses
        loss /= len(loader)

        # Calculating accuracy
        acc = float(acc) / len(loader.dataset)

        # Move model back to CPU
        model.to("cpu")

        # Return loss and accuracy
        return loss, acc
