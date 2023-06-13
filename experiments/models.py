"""This file contains the definition of models"""
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from wide_resnet import WideResNet

INPUT_OUTPUT_SHAPE = {
    "cifar10": [3, 10],
    "cifar100": [3, 100],
    "purchase100": [600, 100],
    "texas100": [6169, 100],
}


class NN(nn.Module):
    """Simple CNN for CIFAR10 dataset."""

    def __init__(self, in_shape, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_shape, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, inputs):
        """Forward pass of the model."""
        inputs = inputs.flatten(1)
        inputs = torch.tanh(self.fc1(inputs))
        outputs = self.fc2(inputs)
        return outputs


class CNN(nn.Module):
    """Simple CNN for CIFAR10 dataset."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        """Forward pass of the model."""
        inputs = self.pool(F.relu(self.conv1(inputs)))
        inputs = self.pool(F.relu(self.conv2(inputs)))
        # flatten all dimensions except batch
        inputs = inputs.reshape(-1, 16 * 5 * 5)
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        outputs = self.fc3(inputs)
        return outputs


class AlexNet(nn.Module):
    """AlexNet model for CIFAR10 dataset."""

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, inputs):
        """Forward pass of the model."""
        inputs = self.features(inputs)
        inputs = inputs.reshape(inputs.size(0), 256 * 2 * 2)
        outputs = self.classifier(inputs)
        return outputs


def get_model(model_type: str, dataset_name: str):
    """Instantiate the model based on the model_type

    Args:
        model_type (str): Name of the model
        dataset_name (str): Name of the dataset
    Returns:
        torch.nn.Module: A model
    """
    num_classes = INPUT_OUTPUT_SHAPE[dataset_name][1]
    in_shape = INPUT_OUTPUT_SHAPE[dataset_name][0]
    if model_type == "CNN":
        return CNN(num_classes=num_classes)
    elif model_type == "alexnet":
        return AlexNet(num_classes=num_classes)
    elif model_type == "wrn28-1":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=28, width=1)
    elif model_type == "wrn28-2":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=28, width=2)
    elif model_type == "wrn28-10":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=28, width=10)
    elif model_type == "nn":
        # for purchase dataset
        return NN(in_shape=in_shape, num_classes=num_classes)
    elif model_type == "vgg16":
        return torchvision.models.vgg16(pretrained=False)
    else:
        raise NotImplementedError(f"{model_type} is not implemented")
