"""This file contains the definition of models"""
import torch.nn.functional as F
from torch import nn
from wide_resnet import WideResNet


class Net(nn.Module):
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


def get_model(model_type: str, num_classes: int = 10) -> nn.Module:
    """Instantiate the model based on the model_type

    Args:
        model_type (str): Name of the model
        num_classes (int): Number of classes
    Returns:
        torch.nn.Module: A model
    """
    if model_type == "CNN":
        return Net(num_classes=num_classes)
    if model_type == "alexnet":
        return AlexNet(num_classes=num_classes)
    if model_type == "wrn28-2":
        model = WideResNet(nin=3, nclass=10, depth=28, width=2)
        return model
    if model_type == "CNN_100":
        return Net(num_classes=100)
    raise NotImplementedError(f"{model_type} is not implemented")
