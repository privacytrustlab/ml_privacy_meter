import torch
from torch import nn


class MLP(nn.Module):
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
