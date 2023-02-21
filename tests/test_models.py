"""This test is for testing functions in models.py."""
import sys

sys.path.append("../experiments")

import pytest

from models import AlexNet, Net, get_model


def test_get_model():
    model = get_model("CNN")
    assert type(model) == Net
    model = get_model("alexnet")
    assert type(model) == AlexNet

    with pytest.raises(NotImplementedError):
        get_model("LR")
