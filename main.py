from math import ceil
import numpy as np
import torch
from torch import nn, optim, Tensor
import tensorflow as tf
from torchvision import models

from privacy_meter.audit import Audit, MetricEnum
from privacy_meter.audit_report import ROCCurveReport, SignalHistogramReport
from privacy_meter.constants import InferenceGame
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource
from privacy_meter.model import PytorchModel

import yaml

#todo: implement the privacy loss auditing for the model and algorithms

def get_dataset(dataset):
    return dataset
    


if __name__ == '__main__':

    config_file = open("config.yaml", 'r')
    configs = yaml.load(config_file,Loader=yaml.Loader)

    np.random.seed(configs['run']['random_seed'])

    print(configs['train'])