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

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

from concurrent import futures
from pathlib import Path

import pickle
import os
from privacy_meter.model import PytorchModelTensor
from privacy_meter import audit_report

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

        print(f'epoch {epoch_idx}: training loss is {train_loss/len(train_loader)}')    
        
        inference(model,test_loader,device)
        inference(model,train_loader,device)
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


def prepare_datasets(dataset_size,configs):
    # split the dataset based on the setting   
    if os.path.exists((f'{log_dir}/data_split.pkl')):
        with open(f'{log_dir}/data_split.pkl','rb') as f:
            dataset_splits = pickle.load(f)
    else:
         
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
        
        logging.info(f"Prepare {configs['num_split']} datasets cost {time.time()-start_time} seconds")
        dataset_splits = {'split':index_list,'data_use_matrix':data_use_matrix}
        with open(f'{log_dir}/data_split.pkl','wb') as f:
                pickle.dump(dataset_splits,f)
    
    return dataset_splits

def get_cifar10_subset(dataset, index,is_tensor=False):
    selected_data = copy.deepcopy(dataset)
   
    selected_data.data = selected_data.data[index]
    selected_data.targets = list(np.array(selected_data.targets)[index])
    
    if is_tensor:
        selected_data.data = torch.from_numpy(selected_data.data).float().permute(0, 3, 1,2) # channel first
        selected_data.targets = torch.tensor(selected_data.targets)
        
    return selected_data

def prepare_models(dataset,data_split,configs):
    # train the mdoels based on the datasets we have 
    if os.path.exists((f'{log_dir}/model_list.pkl')):
        with open(f'{log_dir}/model_list.pkl','rb') as f:
            model_list = pickle.load(f)
    else:
        start_time = time.time()
        model_list = []
        
        for split in range(len(data_split['split'])): # iterative over the dataset splits
            logging.info(f'training models for {split}-th split of the dataset') 
            train_data = get_cifar10_subset(dataset,data_split['split'][split]['train'])
            
            test_data = get_cifar10_subset(dataset,data_split['split'][split]['test'])
            
            
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=configs['train_batch_size'],
                                                    shuffle=True, num_workers=2)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=configs['test_batch_size'],
                                                    shuffle=False, num_workers=2)
            
            print(20*"#")
            print(f'Training the {split}-th model: the training dataset of size {len(train_data)} and test dataset of size {len(test_data)}')
            model = Net()
            train(model,train_loader,test_loader,configs) # all the operation is done on gpu at this stage
            
            model_list.append(copy.deepcopy(model))
            print(20*"#")
        
        with open(f'{log_dir}/model_list.pkl','wb') as f:
            pickle.dump(model_list,f)
        logging.info(f'training {split} uses {time.time()-start_time} seconds') 
    return model_list



def prepare_information_source(dataset,data_split,model_list,configs):
    # construct the auditing datataset for each setting
    if configs['algorithm'] == 'population':
        target_dataset_list = []
        audit_dataset_list = []
        target_model_list = []
        for split in range(len(data_split['split'])): # iterative over the dataset splits
            logging.info(f'training models for {split}-th split of the dataset')
            
            train_data = get_cifar10_subset(dataset,data_split['split'][split]['train'],is_tensor=True)
            
            test_data = get_cifar10_subset(dataset, data_split['split'][split]['test'],is_tensor=True)
            
            audit_data = get_cifar10_subset(dataset, data_split['split'][split]['audit'],is_tensor=True)
                
            # create the target model's dataset
            target_dataset = Dataset(
                data_dict={
                    'train': {'x': train_data.data, 'y': train_data.targets},
                    'test': {'x': test_data.data, 'y': test_data.targets},
                },
                default_input='x',
                default_output='y'
            )
            
            audit_dataset  = Dataset(
                data_dict={
                    'train': {'x': audit_data.data, 'y': audit_data.targets}
                },
                default_input='x',
                default_output='y'
            )
            
            target_dataset_list.append(target_dataset)
            audit_dataset_list.append(audit_dataset)
    
            target_model = PytorchModelTensor(model_obj=model_list[split], loss_fn=nn.CrossEntropyLoss(),device=configs['device'], batch_size=configs['audit_batch_size'])
            target_model_list.append(target_model)
    
        target_info_source = InformationSource(
            models=target_model_list, 
            datasets=target_dataset_list
        )

        reference_info_source = InformationSource(
            models=target_model_list,
            datasets=audit_dataset_list
        )
        
    

    # prepare the information source based on the settings, including training reference mdoels
    return target_info_source,reference_info_source



def audit(target_info, reference_info, configs):
    #todo: call data_prepare
    return None


def generate_priavcy_risk_report(audit_results,configs, data_split_info=None):
    audit_report.REPORT_FILES_DIR = 'privacy_meter/report_files'
    
    if len(audit_results) == 1 and configs['privacy_game']=='privacy_loss_model':
        ROCCurveReport.generate_report(
            metric_result=audit_results[0],
            inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
            save=True, 
            filename = log_dir+'/ROC.png'
        )
        SignalHistogramReport.generate_report(
            metric_result=audit_results[0][0],
            inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
            save=True, 
            filename = log_dir+'/Signals.png'
        )
            
    return None


if __name__ == '__main__':

    config_file = open("config.yaml", 'r')
    configs = yaml.load(config_file,Loader=yaml.Loader)

    np.random.seed(configs['run']['random_seed'])
    torch.manual_seed(configs['run']['random_seed'])
    
    # checks about the setting
    if configs['audit']['privacy_game'] == 'privacy_loss_model':
        assert configs['data']['num_split'] == 1, "only need one model for auditing the privacy risk for a trained model"
    elif configs['audit']['privacy_game'] == 'avg_privacy_loss_training_algo':
        assert configs['data']['num_split'] > 1, "need more models for computing the average privacy loss for an algorithm"
    else:
        raise ValueError(f"{configs['audit']['privacy_game']} has not been implemented")
    
    global log_dir 
    log_dir = configs['run']['log_dir']
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(f'{log_dir}/report').mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(configs['data']['dataset']) # can load from the disk
    data_split_info = prepare_datasets(len(dataset),configs['data'])    
    model_list = prepare_models(dataset,data_split_info,configs['train'])
    target_info_source, reference_info_source = prepare_information_source(dataset,data_split_info,model_list,configs['audit'])
        
    audit_obj = Audit(
        metrics=MetricEnum.POPULATION,
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
        fpr_tolerances=None,
        logs_directory_names=f'{log_dir}/report'
    )
    audit_obj.prepare()
    audit_results = audit_obj.run()
    
    generate_priavcy_risk_report(audit_results,configs['audit'])