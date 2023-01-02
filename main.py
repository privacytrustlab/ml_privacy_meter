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

import torch
import torchvision
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



def prepare_datasets(dataset_size,num_target_model,configs,model_metadata_list,matched_idx=None):         
    index_list = [] # list of data split
    all_index = np.arange(dataset_size)
    
    if matched_idx is not None: 
        for metadata_idx in matched_idx:
            metadata = model_metadata_list['model_metadata'][metadata_idx]
            if 'test_split' in metadata:
                index_list.append({'train':metadata['train_split'],'test':metadata['test_split'],'audit':metadata['audit_split']})
            else: # for resuing the reference models, we randomly select the 
                rest_index = np.array([i for i in all_index if i not in metadata['train_split']])
                test_index = np.random.choice(rest_index,int(configs['f_test']*dataset_size),replace=False)
                
                if configs['split_method'] == 'no_overlapping':
                    audit_index = np.array([i for i in all_index if i not in metadata['train_split'] and i not in test_index])
                elif configs['split_method'] == 'uniform':
                    audit_index =  np.random.choice(all_index,int(configs['f_audit']*dataset_size),replace=False)
                else:
                    raise ValueError(f"{configs['split_method']}' is not implemented")
                
                index_list.append({'train':metadata['train_split'],'test':test_index,'audit':audit_index})
 
    num_addition_dataset = num_target_model - len(index_list)
    for split in range(num_addition_dataset):
        selected_index = np.random.choice(all_index,int((configs['f_train']+configs['f_test'])*dataset_size),replace=False)
        
        train_index, test_index = train_test_split(selected_index, test_size=configs['f_test']/(configs['f_train']+configs['f_test']))
        
        if configs['split_method'] == 'no_overlapping':
            audit_index = np.array([i for i in all_index if i not in selected_index])
        elif configs['split_method'] == 'uniform':
            audit_index =  np.random.choice(all_index,int(configs['f_audit']*dataset_size),replace=False)
        else:
            raise ValueError(f"{configs['split_method']}' is not implemented")
        
        index_list.append({'train':train_index,'test':test_index,'audit':audit_index})
    
    dataset_splits = {'split':index_list,'split_method':configs['split_method']}
    return dataset_splits



def prepare_models(dataset,data_split,configs,model_metadata_list,matched_idx=None): 
    model_list = []
    if matched_idx is not None: 
        for metadata_idx in matched_idx:
            metadata = model_metadata_list['model_metadata'][metadata_idx]
            model = get_model(configs['model_name'])
            with open(f"{metadata['model_path']}",'rb') as f:
                model_weight = pickle.load(f)
            model.load_state_dict(model_weight)
            model_list.append(model)
        logging.info(f"Load existing {len(model_list)} target models")
    

    for split in range(len(model_list),len(data_split['split'])): # iterative over the dataset splits
        meta_data = {}        
        baseline_time = time.time()
        train_data = get_cifar10_subset(dataset,data_split['split'][split]['train'])
        
        test_data = get_cifar10_subset(dataset,data_split['split'][split]['test'])
        
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=configs['batch_size'],
                                                shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=configs['test_batch_size'],
                                                shuffle=False, num_workers=2)
        
        print(50*"-")
        print(f'Training the {split}-th model: the training dataset of size {len(train_data)} and test dataset of size {len(test_data)}')
        
        model = get_model(configs['model_name'])
        model = train(model,train_loader,configs,test_loader) # all the operation is done on gpu at this stage
        model_list.append(copy.deepcopy(model))
        
        logging.info(f'Training target model uses {time.time()-baseline_time} seconds')
        print(50*"-")
        

        # save the information about the metadata
        model_idx = model_metadata_list['current_idx']
        model_metadata_list['current_idx'] +=1
        
        with open(f'{log_dir}/model_{model_idx}.pkl','wb') as f:
            pickle.dump(model.state_dict(),f)
       
        
        meta_data['train_split'] = data_split['split'][split]['train']
        meta_data['test_split'] = data_split['split'][split]['test']
        meta_data['audit_split'] = data_split['split'][split]['audit']
        meta_data['optimizer'] = configs['optimizer']
        meta_data['batch_size'] = configs['batch_size']
        meta_data['epochs'] = configs['epochs']
        meta_data['model_name'] = configs['model_name']
        meta_data['split_method'] = data_split['split_method']
        meta_data['idx'] = model_idx
        meta_data['lr'] = configs['lr']
        meta_data['wd'] = configs['wd']
        meta_data['model_path'] = f'{log_dir}/model_{model_idx}.pkl'
        model_metadata_list['model_metadata'].append(meta_data)
        with open(f'{log_dir}/models_metadata.pkl','wb') as f:
            pickle.dump(model_metadata_list,f)
            
    return model_list,model_metadata_list




def get_info_source_population_attack(dataset,data_split,model, configs):
    train_data = get_cifar10_subset(dataset,data_split['train'],is_tensor=True)
    test_data = get_cifar10_subset(dataset, data_split['test'],is_tensor=True)
    audit_data = get_cifar10_subset(dataset, data_split['audit'],is_tensor=True)
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
    return [target_dataset], [audit_dataset], [target_model], [target_model]


def get_reference_datasets(audit_index,num_reference_datasets,size,split_method):
    index_list = [] # indicate which data points is used for auditing
    all_auditing_index = audit_index
    
    for reference_idx in range(num_reference_datasets):
        if split_method == 'uniform':
            data_index = np.random.choice(all_auditing_index,size,replace=False)
        index_list.append(data_index)
    return index_list    



def get_info_source_reference_attack(dataset,data_split,model,configs,model_metadata_list,matched_reference_idx=None):
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
        
    
    reference_models = []

    if matched_reference_idx is not None: 
        for metadata_idx in matched_reference_idx:
            metadata = model_metadata_list['model_metadata'][metadata_idx]
            
            model = get_model(configs['model_name'])
            with open(f"{metadata['model_path']}",'rb') as f:
                model_weight = pickle.load(f)
            model.load_state_dict(model_weight)
            reference_models.append(PytorchModelTensor(model_obj=model, loss_fn=nn.CrossEntropyLoss(),device=configs['device'], batch_size=configs['audit_batch_size']))
    
    logging.info(f"Load existing {len(reference_models)} reference models")
    num_reference_models = configs['num_reference_models'] - len(reference_models)
    reference_dataset_list = get_reference_datasets(data_split['audit'],num_reference_models,int(configs['f_reference_dataset']*len(train_data)),configs['split_method'])
    
    
    for reference_idx in range(len(reference_dataset_list)):
        meta_data = {}
        logging.info(f'Training  {reference_idx}-th reference model')
        start_time = time.time()
        reference_loader = torch.utils.data.DataLoader(get_cifar10_subset(dataset,reference_dataset_list[reference_idx]), batch_size=configs['batch_size'],
                                                    shuffle=False, num_workers=2)
    
        reference_model = get_model(configs['model_name'])
        reference_model = train(reference_model,reference_loader,configs,None)
        
        
        logging.info(f'Training reference model uses {time.time()-start_time} seconds')
        
        
        model_idx = model_metadata_list['current_idx']
        model_metadata_list['current_idx'] +=1
        with open(f'{log_dir}/model_{model_idx}.pkl','wb') as f:
            pickle.dump(model.state_dict(),f)
            
        meta_data['train_split'] = reference_dataset_list[reference_idx]
        meta_data['optimizer'] = configs['optimizer']
        meta_data['batch_size'] = configs['batch_size']
        meta_data['epochs'] = configs['epochs']
        meta_data['split_method']  = configs['split_method']
        meta_data['idx'] = model_idx
        meta_data['lr'] = configs['lr']
        meta_data['wd'] = configs['wd']
        meta_data['model_name'] = configs['model_name']
        meta_data['model_path'] = f'{log_dir}/model_{model_idx}.pkl'
        model_metadata_list['model_metadata'].append(meta_data)
        with open(f'{log_dir}/models_metadata.pkl','wb') as f:
            pickle.dump(model_metadata_list,f)
        
        
        reference_models.append(PytorchModelTensor(model_obj=reference_model, loss_fn=nn.CrossEntropyLoss(),device=configs['device'], batch_size=configs['audit_batch_size']))

        

    
    target_model = PytorchModelTensor(model_obj=model, loss_fn=nn.CrossEntropyLoss(),device=configs['device'], batch_size=configs['audit_batch_size'])
        
    
    return [target_dataset], [target_dataset], [target_model], reference_models,model_metadata_list




def prepare_information_source(dataset,data_split,model_list,configs,model_metadata_list,matched_reference_idx=None):
    # audit_data_usage_matrix = []
    reference_info_source_list = []
    target_info_source_list = []
    metric_list = []
    log_dir_list = []
    for split in range(len(data_split['split'])): # iterative over the dataset splits
        logging.info(f'preparing information sources for {split}-th split of the dataset')
        # create the target model's dataset
        if configs['algorithm'] == 'population':
            target_dataset, audit_dataset, target_model, audit_models = get_info_source_population_attack(dataset,data_split['split'][split],model_list[split],configs)
            metrics = MetricEnum.POPULATION
        elif configs['algorithm'] == 'reference':
            if matched_reference_idx != None and len(matched_reference_idx) > split:
                target_dataset, audit_dataset, target_model, audit_models, model_metadata_list = get_info_source_reference_attack(dataset,data_split['split'][split],model_list[split],configs,model_metadata_list,matched_reference_idx[split])
            else:
                target_dataset, audit_dataset, target_model, audit_models, model_metadata_list = get_info_source_reference_attack(dataset,data_split['split'][split],model_list[split],configs,model_metadata_list)

            metrics = MetricEnum.REFERENCE      
        metric_list.append(metrics)  
        # audit_data_usage_matrix.append(audit_data_usage)
        
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
        
    return target_info_source_list, reference_info_source_list,metric_list,log_dir_list, model_metadata_list



def prepare_priavcy_risk_report(audit_results,configs,save_path=None):
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


def load_existing_target_model(N, model_metadata_list,configs):
    matched_idx = []
    matching_key = ['optimizer','batch_size','epochs','lr','wd']
    for meta_idx,meta_data in enumerate(model_metadata_list['model_metadata']):
        if len(meta_data['train_split']) == int(N*configs['data']['f_train']) and np.mean([meta_data[key] == configs['train'][key] for key in matching_key]) ==1:
            if configs['train']['key'] == 'none':
                matched_idx.append(meta_idx)
            elif configs['train']['key'] == 'data_idx':
                if (configs['train']['type'] == 'include' and configs['train']['idx'] in meta_data['train_split']) or (configs['train']['type'] == 'exclude' and configs['train']['idx'] not in meta_data['train_spit']): #check if the idx is in the training dataset size.
                    matched_idx.append(meta_idx)
            elif configs['train']['key'] == 'model_idx':
                if meta_data['idx'] == configs['train']['idx']:
                    matched_idx.append(meta_idx)
                    return matched_idx
                
            else:
                raise ValueError(f'key can only be model idx or data idx')
                
        if len(matched_idx) == configs['train']['num_target_model']:
                return matched_idx
    
    return None
   


def load_existing_reference_models(N, model_metadata_list,configs,matched_target_idx):
    reference_matched_idx_list = []
    matching_key = ['optimizer','batch_size','epochs','lr','wd']
    
    for target_idx in matched_target_idx:
        reference_matched_idx = []
        for meta_idx,meta_data in enumerate(model_metadata_list['model_metadata']):
            if meta_idx != target_idx and len(meta_data['train_split']) == int(N*configs['data']['f_audit']) and np.mean([meta_data[key] == configs['audit'][key] for key in matching_key]) == 1:
                if configs['audit']['key'] == 'none':
                    # load the models based on the training configuration: trained in the same way but different datasets
                    if configs['data']['split_method'] == 'no_overlapping':
                        if len(set(model_metadata_list['model_metadata'][meta_idx]['train_split']) - set(model_metadata_list['model_metadata'][target_idx]['train_split'])) == len(set(model_metadata_list['model_metadata'][meta_idx]['train_split'])):
                            reference_matched_idx.append(meta_idx)    
                    elif configs['data']['split_method'] == 'uniform':
                        if collections.Counter(model_metadata_list['model_metadata'][meta_idx]['train_split']) != collections.Counter(model_metadata_list['model_metadata'][target_idx]['train_split']):
                            reference_matched_idx.append(meta_idx)    
                            
                elif configs['audit']['key'] == 'data_idx':
                    # load the models based on the training configuration include or exclude of the data_idx
                    if (configs['audit']['type'] == 'include' and configs['audit']['idx'] in meta_data['train_split']) or (configs['audit']['type'] == 'exclude' and configs['audit']['idx'] not in meta_data['train_split']): #check if the idx is in the training dataset size.
                        reference_matched_idx.append(meta_idx)
                else:
                    raise ValueError(f'key can only be data idx for loading reference models')
            # check if the existing referencce models already satisfy the constraints            
            if len(reference_matched_idx) == configs['audit']['num_reference_models']:
                break
            
        reference_matched_idx_list.append(reference_matched_idx)
                        
    return reference_matched_idx_list



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cf', type=str, default="config_sample.yaml",help='Yaml file which contains the configurations')
    args = parser.parse_args()
    
    
    start_time = time.time()
    config_file = open(args.cf, 'r')
    configs = yaml.load(config_file,Loader=yaml.Loader)

    # np.random.seed(configs['run']['random_seed']) # this is to ensure that the data split will be different when calling the train and test split
    torch.manual_seed(configs['run']['random_seed'])
    
    # checks about the setting
    
    if configs['audit']['privacy_game'] == 'privacy_loss_model':
        assert configs['train']['num_target_model'] == 1, "only need one model for auditing the privacy risk for a trained model"
    elif configs['audit']['privacy_game'] == 'avg_privacy_loss_training_algo' or configs['audit']['privacy_game'] == 'privacy_loss_sample':
        assert configs['train']['num_target_model'] > 1, "need more models for computing the average privacy loss for an algorithm"
    else:
        raise ValueError(f"{configs['audit']['privacy_game']} has not been implemented")
    inference_game_type = configs['audit']['privacy_game'].upper()
    

    
    # indicate the folder path for the logs
    global log_dir
        
    log_dir = configs['run']['log_dir']
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{log_dir}/{configs['audit']['report_log']}").mkdir(parents=True, exist_ok=True)

    # load models based on metadata
    if os.path.exists((f'{log_dir}/models_metadata.pkl')):
        with open(f'{log_dir}/models_metadata.pkl','rb') as f:
            model_metadata_list = pickle.load(f)
    else: 
        model_metadata_list = {'model_metadata':[],'current_idx':0} # model_meta_list saved all the trained models information

    
    # construct the dataset
    baseline_time = time.time()
    dataset = get_dataset(configs['data']['dataset'],log_dir) # can load from the disk
    
    
    if configs['audit']['privacy_game'] in ['avg_privacy_loss_training_algo','privacy_loss_model']:
        #load the saved models which are useful for the auditing (target model information)
        if model_metadata_list['current_idx'] > 0:
            matched_idx = load_existing_target_model(len(dataset),model_metadata_list,configs)
            if configs['audit']['algorithm'] == 'reference' and matched_idx is not None:
                matched_reference_idx = load_existing_reference_models(len(dataset),model_metadata_list,configs,matched_idx)
            else:
                matched_reference_idx = None
        else:
            matched_idx,matched_reference_idx = None,None
            
            
        data_split_info = prepare_datasets(len(dataset),configs['train']['num_target_model'],configs['data'],model_metadata_list,matched_idx)    
        
        logging.info(f'prepare the dataset costs {time.time()-baseline_time} seconds')
        logging.info(25*"#"+"Prepare the the target models"+25*"#")
        
        
        baseline_time = time.time()
        model_list, model_metadata_list = prepare_models(dataset,data_split_info,configs['train'],model_metadata_list,matched_idx)
        
        logging.info(f'prepare the target model costs {time.time()-baseline_time} seconds')
        logging.info(25*"#"+"Prepare the information source, including attack models"+25*"#")
        
        
        
        
        with open(f'{log_dir}/models_metadata.pkl','wb') as f:
            pickle.dump(model_metadata_list,f)
            
        # for auditing the average priavcy risk over the dataset 
        baseline_time = time.time()
        target_info_source, reference_info_source,metrics,log_dir_list,model_metadata_list = prepare_information_source(dataset,data_split_info,model_list,configs['audit'],model_metadata_list,matched_reference_idx)
        logging.info(f'prepare the information source costs {time.time()-baseline_time} seconds')
        logging.info(25*"#"+"Auditing the privacy risk"+25*"#")
        
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
        logging.info(25*"#"+"Generating privacy risk report"+25*"#")
        
        
        baseline_time = time.time()
        prepare_priavcy_risk_report(audit_results,configs['audit'],save_path=f"{log_dir}/{configs['audit']['report_log']}")
        logging.info(f'plotting the report {time.time()-baseline_time} seconds')    
        logging.info(f'overall process costs {time.time()-start_time} seconds')
        logging.info(100*"#")

    
    # elif configs['audit']['privacy_game'] == 'privacy_loss_sample':
        # TODO: load the references information source and target information source 
        # TODO: given a set of target models (trained with data_idx) and reference models (trained without data_idx), infer the membership information from 
        
    
        # target_info_source['models'].get_signal()
        