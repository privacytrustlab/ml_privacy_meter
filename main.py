from math import ceil
import numpy as np
import torch
from torch import nn, optim, Tensor
import tensorflow as tf
from torchvision import models
import argparse
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

def train(model, train_loader,configs,test_loader=None):
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
        
        if test_loader is not None:
            inference(model,test_loader,device)
        inference(model,train_loader,device)
    model.to('cpu')
    
    return model


def inference(model,test_loader,device):
    model.eval()
    model.to(device)
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
    


def get_dataset(dataset_name):
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
            
            train_index, test_index = train_test_split(selected_index, test_size=configs['f_test']/(configs['f_train']+configs['f_test']))
            
            if configs['audit_split_method'] == 'no_overlapping':
                audit_index = np.array([i for i in all_index if i not in selected_index])
            elif configs['audit_split_method'] == 'uniform':
                audit_index =  np.random.choice(all_index,int(configs['f_audit']*dataset_size),replace=False)
            else:
                raise ValueError(f"{configs['audit_split_method']}' is not implemented")
            
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
        selected_data.data = torch.from_numpy(selected_data.data).float().permute(0, 3, 1,2)/255 # channel first 
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
            
            print(50*"#")
            print(f'Training the {split}-th model: the training dataset of size {len(train_data)} and test dataset of size {len(test_data)}')
            
            model = get_model(configs['model_name'])
            model = train(model,train_loader,configs,test_loader) # all the operation is done on gpu at this stage
            
            model_list.append(copy.deepcopy(model))
            print(50*"#")
        
        with open(f'{log_dir}/model_list.pkl','wb') as f:
            pickle.dump(model_list,f)
        logging.info(f'training {split} uses {time.time()-start_time} seconds') 
    return model_list




def get_info_source_population_attack(dataset,data_split,model, configs):
    train_data = get_cifar10_subset(dataset,data_split['train'],is_tensor=True)
    test_data = get_cifar10_subset(dataset, data_split['test'],is_tensor=True)
    audit_data = get_cifar10_subset(dataset, data_split['audit'],is_tensor=True)
    audit_data_usage = [data_split['audit']] #indicate which data point is used for auditing
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
    target_model = PytorchModelTensor(model_obj=model, loss_fn=nn.CrossEntropyLoss(),device=configs['device'], batch_size=configs['audit_batch_size'])
    return [target_dataset], [audit_dataset], [target_model], [target_model],audit_data_usage


def get_reference_datasets(audit_index,num_reference_datasets,size):
    data_use_matrix = [] # indicate which data points is used for auditing
    all_auditing_index = audit_index
    for reference_idx in range(num_reference_datasets):
        data_index = np.random.choice(all_auditing_index,size,replace=False)
        data_use_matrix.append(data_index)
    return data_use_matrix    



def get_info_source_reference_attack(dataset,data_split,model,configs):
    start_time = time.time()
    train_data = get_cifar10_subset(dataset,data_split['train'],is_tensor=True)
    test_data = get_cifar10_subset(dataset, data_split['test'],is_tensor=True)
    target_dataset = Dataset(
        data_dict={
            'train': {'x': train_data.data, 'y': train_data.targets},
            'test': {'x': test_data.data, 'y': test_data.targets},
        },
        default_input='x',
        default_output='y'
    )
    
    if os.path.exists((f'{log_dir}/reference_model_list.pkl')):
        with open(f'{log_dir}/reference_model_list.pkl','rb') as f:
            reference_models = pickle.load(f)
        with open(f'{log_dir}/reference_datasets_list.pkl','rb') as f:
            reference_dataset_list = pickle.load(f)
  
    else:
        reference_models = []
        reference_dataset_list = get_reference_datasets(data_split['audit'],configs['num_reference_models'],int(configs['f_reference_dataset']*len(train_data)))
        
        for reference_idx in range(len(reference_dataset_list)):
            print(f'training  {reference_idx}-th reference model')
            reference_loader = torch.utils.data.DataLoader(get_cifar10_subset(dataset,reference_dataset_list[reference_idx]), batch_size=configs['reference_batch_size'],
                                                        shuffle=False, num_workers=2)
        
            reference_model = get_model(configs['model_name'])
            reference_model = train(reference_model,reference_loader,configs,None)
            reference_models.append(PytorchModelTensor(model_obj=reference_model, loss_fn=nn.CrossEntropyLoss(),device=configs['device'], batch_size=configs['audit_batch_size']))

        with open(f'{log_dir}/reference_model_list.pkl','wb') as f:
            pickle.dump(reference_models,f)
        with open(f'{log_dir}/reference_datasets_list.pkl','wb') as f:
            pickle.dump(reference_dataset_list,f)

    
    target_model = PytorchModelTensor(model_obj=model, loss_fn=nn.CrossEntropyLoss(),device=configs['device'], batch_size=configs['audit_batch_size'])
    return [target_dataset], [target_dataset], [target_model], reference_models, reference_dataset_list




def prepare_information_source(dataset,data_split,model_list,configs):
    audit_data_usage_matrix = []
    
    reference_info_source_list = []
    target_info_source_list = []
    metric_list = []
    log_dir_list = []
    for split in range(len(data_split['split'])): # iterative over the dataset splits
        logging.info(f'preparing information sources for {split}-th split of the dataset')
        # create the target model's dataset
        if configs['algorithm'] == 'population':
            target_dataset, audit_dataset, target_model, audit_models, audit_data_usage = get_info_source_population_attack(dataset,data_split['split'][split],model_list[split],configs)
            metrics = MetricEnum.POPULATION
        elif configs['algorithm'] == 'reference':
            target_dataset, audit_dataset, target_model, audit_models, audit_data_usage = get_info_source_reference_attack(dataset,data_split['split'][split],model_list[split],configs)
            metrics = MetricEnum.REFERENCE      
        metric_list.append(metrics)  
        audit_data_usage_matrix.append(audit_data_usage)
        
        target_info_source = InformationSource(
            models=target_model, 
            datasets=target_dataset
        )

        reference_info_source = InformationSource(
            models=audit_models,
            datasets=audit_dataset
        )
        reference_info_source_list.append(reference_info_source)
        target_info_source_list.append(target_info_source)
        
        
        log_dir_path = f"{log_dir}/{configs['report_log']}_{split}"
        Path(log_dir_path).mkdir(parents=True, exist_ok=True)
        log_dir_list.append(log_dir_path)
        
    return target_info_source_list, reference_info_source_list,metric_list,audit_data_usage_matrix,log_dir_list



def prepare_priavcy_risk_report(audit_results,configs, data_split_info=None,save_path=None):
    audit_report.REPORT_FILES_DIR = 'privacy_meter/report_files'
    if save_path is None:
        save_path = log_dir
    
    if len(audit_results) == 1 and configs['privacy_game']=='privacy_loss_model':
        ROCCurveReport.generate_report(
            metric_result=audit_results[0],
            inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
            save=True, 
            filename = f"{save_path}/ROC.png"
        )
        SignalHistogramReport.generate_report(
            metric_result=audit_results[0][0],
            inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
            save=True, 
            filename = f"{save_path}/Histogram.png"
        )
    
    elif len(audit_results) > 1 and configs['privacy_game'] == 'avg_privacy_loss_training_algo':
        ROCCurveReport.generate_report(
            metric_result=audit_results,
            inference_game_type=InferenceGame.AVG_PRIVACY_LOSS_TRAINING_ALGO,
            save=True, 
            filename = f"{save_path}/ROC.png"
        )
        
        SignalHistogramReport.generate_report(
            metric_result=audit_results,
            inference_game_type=InferenceGame.AVG_PRIVACY_LOSS_TRAINING_ALGO,
            save=True, 
            filename = f"{save_path}/Histogram.png"
        )

    else:
        
        raise ValueError(f"{configs['privacy_game']} is not implemented yet")
            
    return None

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--cf', type=str, default="config.yaml",help='Yaml file which contains the configurations')
    args = parser.parse_args()
    
    
    start_time = time.time()
    config_file = open(args.cf, 'r')
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
    inference_game_type = configs['audit']['privacy_game'].upper()
    
    
    
    # indicate the folder path for the logs
    global log_dir
    log_dir = configs['run']['log_dir']

    
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{log_dir}/{configs['audit']['report_log']}").mkdir(parents=True, exist_ok=True)


    # construct the dataset
    
    baseline_time = time.time()
    dataset = get_dataset(configs['data']['dataset']) # can load from the disk
    data_split_info = prepare_datasets(len(dataset),configs['data'])    
    
    logging.info(f'prepare the dataset costs {time.time()-baseline_time} seconds')
    logging.info(50*"#")
    
    
    baseline_time = time.time()
    model_list = prepare_models(dataset,data_split_info,configs['train'])
    
    logging.info(f'prepare the target model costs {time.time()-baseline_time} seconds')
    logging.info(50*"#")
    
    baseline_time = time.time()
    target_info_source, reference_info_source,metrics,reference_data_usage,log_dir_list = prepare_information_source(dataset,data_split_info,model_list,configs['audit'])
    logging.info(f'prepare the information source costs {time.time()-baseline_time} seconds')
    logging.info(50*"#")
    
    
    baseline_time = time.time()
    audit_obj = Audit(
        metrics=metrics,
        inference_game_type=inference_game_type,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
        fpr_tolerances=None,
        logs_directory_names=log_dir_list 
    )
    audit_obj.prepare()
    audit_results = audit_obj.run()  
    logging.info(f'running privacy meter costs {time.time()-baseline_time} seconds')
    logging.info(50*"#")
    
     
    baseline_time = time.time()
    prepare_priavcy_risk_report(audit_results,configs['audit'],reference_data_usage,save_path=f"{log_dir}/{configs['audit']['report_log']}")
    logging.info(f'plotting the report {time.time()-baseline_time} seconds')
    logging.info(50*"#")
    
    logging.info(f'overall process costs {time.time()-start_time} seconds')