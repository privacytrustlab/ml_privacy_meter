from ast import Tuple
import torch
from torch import nn
import torch

import torch
from util import get_optimizer


# Train Function
def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          configs: dict) -> torch.nn.Module:
    """Train the model based on on the train loader
    Args:
        model(nn.Module): Model for evaluation.
        train_loader(torch.utils.data.DataLoader): Data loader for training.
        configs (dict): Configurations for training.
    Return:
        nn.Module: Trained model.
    """
    # Get the device for training
    device = configs.get('device', torch.device('cpu'))

    # Set the model to the device
    model.to(device)
    model.train()

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, configs)

    # Get the number of epochs for training
    epochs = configs.get('epochs', 1)

    # Loop over each epoch
    for epoch_idx in range(epochs):
        train_loss = 0
        # Loop over the training set
        for data, target in train_loader:

            # Move data to the device
            data, target = data.to(device), target.to(device)

            # Set the gradients to zero
            optimizer.zero_grad()

            # Get the model output
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)

            # Perform the backward pass
            loss.backward()

            # Take a step using optimizer
            optimizer.step()

            # Add the loss to the total loss
            train_loss += loss.item()

        # Print the epoch and loss summary
        print(
            f'Epoch: {epoch_idx+1}/{epochs} | Loss: {train_loss/len(train_loader):.8f}')

    # Move the model back to the CPU
    model.to("cpu")

    # Return the model
    return model

# Test Function


def inference(model: torch.nn.Module,
              loader: torch.utils.data.DataLoader,
              device: str,
              is_train: bool = False) -> Tuple(float, float):
    """Evaluate the model performance on the test loader
    Args:
        model (torch.nn.Module): Model for evaluation
        loader (torch.utils.data.DataLoader): Data Loader for testing
        device (str): GPU or CPU
        is_train (bool, optional): Whether test_loader is from the train dataset or test dataset. Defaults to False.
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
        for batch_idx, (data, target) in enumerate(loader):
            # Moving data and target to the device
            data, target = data.to(device), target.to(device)

            # Computing output and loss
            output = model(data)
            loss += criterion(output, target).item()

            # Computing accuracy
            pred = output.data.max(1, keepdim=True)[1]
            acc += pred.eq(target.data.view_as(pred)).sum()

        # Averaging the losses
        loss /= len(loader)

        # Calculating accuracy
        acc = float(acc)/len(loader.dataset)

        # Printing the results
        print(f"{'Train' if is_train else 'Test'} accuracy {acc}, loss {loss}")

        # Move model back to CPU
        model.to("cpu")

        # Return loss and accuracy
        return loss, acc
