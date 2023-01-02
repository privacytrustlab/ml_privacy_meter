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
import torch
import yaml

import torch
import torchvision
import torchvision.transforms as transforms
import logging
from sklearn.model_selection import train_test_split
import time
import copy
from models import Net

#todo: In this code, we provide the tutorials about auditing privacy risk for different types of games

import multiprocessing as mp

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

    

 



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

def train(model, train_loader, test_loader,configs):
    # update the model based on the train_dataset
    logging.info('training models')
   
   
    device = configs['device']
    
    model.to(device)
    model.train()
    optimizer = get_optimizer(model,configs)
    for epoch_idx in range(configs['epochs']):
        train_loss = 0
        for batch_idx, (data,target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            ceiterion = nn.CrossEntropyLoss()
            loss = ceiterion(output,target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        logging.info(f'epoch {epoch_idx}: training loss is {train_loss/len(train_loader)}')    
        
        inference(model,test_loader,device)
    model.to('cpu')
    
    return model


def inference(model,test_loader,device):
    model.eval()
    loss = 0
    acc = 0
    with torch.no_grad():
        for batch_idx, (data,target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            ceiterion = nn.CrossEntropyLoss()
            loss += ceiterion(output,target).item()
            pred = output.data.max(1,keepdim=True)[1]
            acc += pred.eq(target.data.view_as(pred)).sum()

        loss /= len(test_loader)
        acc = float(acc)/len(test_loader.dataset)
        
    print(f'Test accuracy {acc}, Test loss {loss}')
    


def get_dataset(dataset_name,group=None):
    # load the dataset 
    if dataset_name == 'cifar10':
        
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        train_data = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
        test_data = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
        X = np.concatenate([train_data.data,test_data.data],axis=0)
        Y = np.concatenate([train_data.targets,test_data.targets],axis=0)
        
        all_data = copy.deepcopy(train_data)
        all_data.data = X
        all_data.targets = Y
        N = len(all_data)
        
        logging.info(f"the whole dataset size: {N}")
        
    return all_data


def prepare_datasets(dataset_size,configs):
    # split the dataset based on the setting    
    data_use_matrix = [] # keep track on which data samples are used
    index_list = [] # list of data split
    all_index = np.arange(dataset_size)
    
    start_time = time.time()
    for split in range(configs['num_split']):
        data_use = np.zeros(dataset_size)
        selected_index = np.random.choice(all_index,int((configs['f_train']+configs['f_test'])*dataset_size),replace=False)
        audit_index = np.array([i for i in all_index if i not in selected_index])
        train_index, test_index = train_test_split(selected_index, test_size=configs['f_test']/(configs['f_train']+configs['f_test']))
        index_list.append({'train':train_index,'test':test_index,'audit':audit_index})
        
        # track teh data usage
        data_use[train_index] = 1 #train
        data_use[test_index] = 0 #test
        data_use[audit_index] = -1 #audit
        data_use_matrix.append(data_use)
    
    logging.info(f'Spliting the dataset cost {time.time()-start_time} seconds')
    dataset_splits = {'split':index_list,'data_use_matrix':data_use_matrix}
    return dataset_splits


def prepare_models(dataset,data_split,configs, split_idx):
    # train the mdoels based on the datasets we have 
    start_time = time.time()
    model_list = []
    
    
    logging.info(f'training models for {split_idx}-th split of the dataset')
    train_index = data_split['split'][split_idx]['train']
    test_index = data_split['split'][split_idx]['test']
    
    train_data = copy.deepcopy(dataset)
    train_data.data = train_data.data[train_index]
    train_data.targets = list(np.array(train_data.targets)[train_index])
    
    test_data = copy.deepcopy(dataset)
    test_data.data = test_data.data[test_index]
    test_data.targets = list(np.array(test_data.targets)[test_index])
    
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=configs['train_batch_size'],
                                            shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=configs['test_batch_size'],
                                            shuffle=False, num_workers=2)
    
    print(20*"#")
    print(f'Training the {split_idx}-th model: the training dataset of size {len(train_data)} and test dataset of size {len(test_data)}')
    model = Net()
    train(model,train_loader,test_loader,configs) # all the operation is done on gpu at this stage
    
    model_list.append(copy.deepcopy(model))
    print(20*"#")
        
    logging.info(f'training {split_idx} uses {time.time()-start_time} seconds') 
    return model_list

def prepare_information_source(models, datasets, configs):
    # prepare the information source based on the settings, including training reference mdoels
    return None

def preparation(configs,models, datasets):
    #todo: call data_prepare
    return None


def generate_priavcy_risk_report(pm_results, configs):
    return None


if __name__ == '__main__':

    config_file = open("config.yaml", 'r')
    configs = yaml.load(config_file,Loader=yaml.Loader)

    np.random.seed(configs['run']['random_seed'])
    torch.manual_seed(configs['run']['random_seed'])
    
    
    dataset = get_dataset(configs['data']['dataset']) # can load from the disk
    data_split_info = prepare_datasets(len(dataset),configs['data'])
    
       # Step 1: Use multiprocessing.Pool() and specify number of cores to use (here I use 4).
    pool = mp.Pool(10)
    results = pool.starmap(prepare_models, [( dataset,data_split_info,configs['train'], idx) for idx in range(10)])
    # Step 3: Don't forget to close
    pool.close()




# with futures.ProcessPoolExecutor() as pool:
#     model_list = []
#     for mdoel in pool.map(prepare_models,):
#         model_list.append(model)
#     # prepare_models(dataset,data_split_info,configs['train'])
    