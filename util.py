import numpy as np
import torch
from torch import optim
import torch

import torch
import torchvision
import torchvision.transforms as transforms
import logging
import copy
from models import Net
#todo: In this code, we provide the tutorials about auditing privacy risk for different types of games

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


import pickle
import os




def get_dataset(dataset_name,log_dir):
    # load the dataset 
    if os.path.exists((f'{log_dir}/data.pkl')):
        with open(f'{log_dir}/data.pkl','rb') as f:
            all_data = pickle.load(f)
    else:
        if dataset_name == 'cifar10':
            
            transform = transforms.Compose(
                [transforms.ToTensor()]
            )

            train_data = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
            test_data = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
            X = np.concatenate([train_data.data,test_data.data],axis=0)
            Y = np.concatenate([train_data.targets,test_data.targets],axis=0)
            
            all_data = copy.deepcopy(train_data)
            all_data.data = X
            all_data.targets = Y
            with open(f'{log_dir}/data.pkl','wb') as f:
                pickle.dump(all_data,f)
        else:
            raise ValueError(f"{dataset_name} is not implemented")
        
    
    N = len(all_data)
    logging.info(f"the whole dataset size: {N}")   
    return all_data


def get_model(model_type):
    # init a model 
    if model_type == 'CNN':
        return Net()

def get_optimizer(model,configs):
    if configs['optimizer'] == 'SGD':
        return optim.SGD(model.parameters(),lr = configs['lr'],momentum=configs['momentum'], weight_decay=configs['wd'])
    elif configs['optimizer'] == 'Adam':
        return optim.Adam(model.parameters(),lr = configs['lr'],weight_decay=configs['wd'])
    
    else:
        raise AttributeError(f"Optimizer {configs['optimizer']}  has not been implemented. Please choose from SGD or Adam")



def get_cifar10_subset(dataset, index,is_tensor=False):
    selected_data = copy.deepcopy(dataset)
   
    selected_data.data = selected_data.data[index]
    selected_data.targets = list(np.array(selected_data.targets)[index])
    
    if is_tensor:
        selected_data.data = torch.from_numpy(selected_data.data).float().permute(0, 3, 1,2)/255 # channel first 
        selected_data.targets = torch.tensor(selected_data.targets)
        
    return selected_data