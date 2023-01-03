




import numpy as np
import torch
from torch import nn, optim
import argparse
from privacy_meter.audit import Audit, MetricEnum
from privacy_meter.audit_report import ROCCurveReport, SignalHistogramReport
from privacy_meter.constants import InferenceGame
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource
import torch
import yaml
import seaborn as sns
import torch
import torchvision
import pandas as pd
import torchvision.transforms as transforms
import logging
from sklearn.model_selection import train_test_split
import time
import copy
from models import Net
import collections
#todo: In this code, we provide the tutorials about auditing privacy risk for different types of games

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

from pathlib import Path

import pickle
import os
from privacy_meter.model import PytorchModelTensor
from privacy_meter import audit_report
from util import get_cifar10_subset,get_model,get_optimizer,get_dataset
from train import train
import matplotlib.pyplot as plt
import os




if __name__ == "__main__":
    log_dir = "demo"
    
    if os.path.exists(f"{log_dir}/models_metadata.pkl"):
        with open(f"{log_dir}/models_metadata.pkl","rb") as f:
            model_metadata_list = pickle.load(f)
        updated_model_metadata_list = copy.deepcopy(model_metadata_list)
        
        for idx in model_metadata_list['model_metadata']:
            metadata = model_metadata_list['model_metadata'][idx]
            # here, indicate the conditions to delete models
            if metadata['epochs'] == 0:
                if os.path.exists(metadata['model_path']):
                    os.remove(metadata['model_path']) # delete the model 
                del updated_model_metadata_list['model_metadata'][idx]
                print(f"delete model saved in {metadata['model_path']}")
       
        with open(f"{log_dir}/models_metadata.pkl","wb") as f:
            pickle.dump(updated_model_metadata_list,f)
            
    else:
        raise ValueError(f"Model metadata ({log_dir}/models_metadata.pkl) file does not exist")