
import sys
sys.path.append('../../')

from models import Net, AlexNet
import pytest
from models import get_model


# This test is for testing functions in models.py.

# Test models
def test_get_model():
    model = get_model('CNN')
    assert type(model) == Net
    model = get_model('alexnet')
    assert type(model) == AlexNet

    with pytest.raises(NotImplementedError):
        get_model('LR')
