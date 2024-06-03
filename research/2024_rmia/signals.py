"""
Script dedicated to loading and computing new signals from logits
"""
import os
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from scipy import stats
from scipy.stats import entropy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def factorial(n):
    fact = 1
    for i in range(2, n + 1):
        fact = fact * i
    return fact


def get_taylor(logit_signals, n):
    power = logit_signals
    taylor = power + 1.0
    for i in range(2, n):
        power = power * logit_signals
        taylor = taylor + (power / factorial(i))
    return taylor


def load_input_labels(base_dataset_path):
    try:
        original_labels = torch.from_numpy(np.load(base_dataset_path + '/y_train.npy'))
    except Exception as e:
        print(e)
        print("Fail")
        print(base_dataset_path)

    return original_labels

def load_inputs(base_dataset_path):
    try:
        original_inputs = torch.from_numpy(np.load(base_dataset_path + '/x_train.npy'))
        original_labels = torch.from_numpy(np.load(base_dataset_path + '/y_train.npy'))
    except Exception as e:
        print(e)
        print("Fail")
        print(base_dataset_path)
    return original_inputs, original_labels

def get_experiment_model_index(subfolder_name):
    try:
        return int(subfolder_name.replace('experiment-', '').split('_')[0])
    except ValueError:
        return 0

def load_input_gradient(base_dataset_path:str, epoch_number:int, model:int=None, num_augmentations:int=0):
    """
    Load Gradients wrt input for the corresponding model
    if model = None, then return all logits and memberships at once sorted by model index
    """
    try:
        all_gradients = []
        keeps = [] # membership matrix for each model
        
        if model is not None: # select that particular model's output signal
            for each_folder in os.listdir(base_dataset_path):
                if each_folder == "x_train.npy" or each_folder == "y_train.npy" or each_folder == "x_test.npy" or each_folder == "y_test.npy" :
                    continue
                dataset_path = base_dataset_path + '/' + each_folder
                dataset_index = int(each_folder.replace('experiment-', '').split('_')[0])
                if model is not None: # if we select a model, only takes the model's logits
                    if dataset_index != model:
                        continue

                logit_path = dataset_path + '/gradients'
                new_row = torch.from_numpy(np.load(dataset_path + '/keep.npy'))

                logits_name = f"{epoch_number:010d}_{num_augmentations:04d}.npy" # depends on how the file is named after inference.py
                
                original_ref_signals = torch.from_numpy(np.array(np.load(logit_path + '/' + logits_name), dtype=np.float64))
                all_gradients.append(original_ref_signals[:, :, :, :])
            
                keeps.append(new_row)

            all_gradients = torch.stack(all_gradients)
            keeps = torch.stack(keeps)    
        else : # go through all trained models
            all_subfolders = [f for f in os.listdir(base_dataset_path) if os.path.isdir(os.path.join(base_dataset_path, f))]
            sorted_subfolders = sorted(all_subfolders, key=get_experiment_model_index)

            for each_folder in sorted_subfolders: # MAJOR FIX
                
                if each_folder == "x_train.npy" or each_folder == "y_train.npy" or each_folder == "x_test.npy" or each_folder == "y_test.npy":
                    continue
                dataset_path = base_dataset_path + '/' + each_folder
                dataset_index = get_experiment_model_index(each_folder)

                logit_path = dataset_path + '/gradients'
                new_row = torch.from_numpy(np.load(dataset_path + '/keep.npy'))
                
                original_ref_signals = torch.from_numpy(np.array(np.load(logit_path + '/' + f"{epoch_number:010d}_{num_augmentations:04d}.npy"), dtype=np.float64))
            
                all_gradients.append(original_ref_signals[:, :, :, :])
                keeps.append(new_row)

            all_gradients = torch.stack(all_gradients)
            keeps = torch.stack(keeps)    

    except Exception as e:
        print(e)
        print("Signals Not Found: You may have to compute them.")
        print(base_dataset_path)
    return all_gradients, keeps

def load_input_logits(base_dataset_path:str, epoch_number:int, model:int=None, num_augmentations:int=2):
    """
    Load Logits for the corresponding set of augmentations and the model
    if model = None, then return all logits and memberships at once sorted by model index
    """
    try:
        all_logits = []
        keeps = [] # membership matrix for each model
        
        if model is not None: # select that particular model's output signal
            for each_folder in os.listdir(base_dataset_path):
                if each_folder == "x_train.npy" or each_folder == "y_train.npy" or each_folder == "x_test.npy" or each_folder == "y_test.npy" :
                    continue
                dataset_path = base_dataset_path + '/' + each_folder
                dataset_index = int(each_folder.replace('experiment-', '').split('_')[0])
                if model is not None: # if we select a model, only takes the model's logits
                    if dataset_index != model:
                        continue

                logit_path = dataset_path + '/logits'
                new_row = torch.from_numpy(np.load(dataset_path + '/keep.npy'))

                logits_name = f"{epoch_number:010d}_{num_augmentations:04d}.npy" # depends on how the file is named after inference.py
                
                original_ref_signals = torch.from_numpy(np.array(np.load(logit_path + '/' + logits_name), dtype=np.float64))
                all_logits.append(original_ref_signals[:, 0, :, :])
            
                keeps.append(new_row)

            all_logits = torch.stack(all_logits)
            keeps = torch.stack(keeps)    
        else : # go through all trained models
            all_subfolders = [f for f in os.listdir(base_dataset_path) if os.path.isdir(os.path.join(base_dataset_path, f))]
            sorted_subfolders = sorted(all_subfolders, key=get_experiment_model_index)

            for each_folder in sorted_subfolders: # MAJOR FIX
                
                if each_folder == "x_train.npy" or each_folder == "y_train.npy" or each_folder == "x_test.npy" or each_folder == "y_test.npy":
                    continue
                dataset_path = base_dataset_path + '/' + each_folder
                dataset_index = get_experiment_model_index(each_folder)

                logit_path = dataset_path + '/logits'
                new_row = torch.from_numpy(np.load(dataset_path + '/keep.npy'))
                
                original_ref_signals = torch.from_numpy(np.array(np.load(logit_path + '/' + f"{epoch_number:010d}_{num_augmentations:04d}.npy"), dtype=np.float64))
            
                all_logits.append(original_ref_signals[:, 0, :, :])
                keeps.append(new_row)

            all_logits = torch.stack(all_logits)
            keeps = torch.stack(keeps)    

    except Exception as e:
        print(e)
        print("Signals Not Found: You may have to compute them.")
        print(base_dataset_path)
    return all_logits, keeps


def convert_signals(all_logits, all_true_labels, metric, temp, extra=None):
    if metric == 'softmax':
        logit_signals = torch.div(all_logits, temp)
        max_logit_signals, max_indices = torch.max(logit_signals, dim=1)
        logit_signals = torch.sub(logit_signals, max_logit_signals.reshape(-1, 1))
        exp_logit_signals = torch.exp(logit_signals)
        exp_logit_sum = exp_logit_signals.sum(axis=1).reshape(-1, 1)
        true_exp_logit = exp_logit_signals.gather(1, all_true_labels.reshape(-1, 1))
        output_signals = torch.div(true_exp_logit, exp_logit_sum)
    elif metric == 'taylor':
        n = extra["taylor_n"]
        taylor_signals = get_taylor(all_logits, n)
        taylor_logit_sum = taylor_signals.sum(axis=1).reshape(-1, 1)
        true_taylor_logit = taylor_signals.gather(1, all_true_labels.reshape(-1, 1))
        output_signals = torch.div(true_taylor_logit, taylor_logit_sum)
    elif metric == 'soft-margin':
        m = float(extra["taylor_m"])
        logit_signals = torch.div(all_logits, temp)
        exp_logit_signals = torch.exp(logit_signals)
        exp_logit_sum = exp_logit_signals.sum(axis=1).reshape(-1, 1)
        true_logits = logit_signals.gather(1, all_true_labels.reshape(-1, 1))
        exp_true_logit = exp_logit_signals.gather(1, all_true_labels.reshape(-1, 1))
        exp_logit_sum = exp_logit_sum - exp_true_logit
        soft_true_logit = torch.exp(true_logits - m)
        exp_logit_sum = exp_logit_sum + soft_true_logit
        output_signals = torch.div(soft_true_logit, exp_logit_sum)
    elif metric == 'taylor-soft-margin':
        m, n = float(extra["taylor_m"]), int(extra["taylor_n"])
        logit_signals = torch.div(all_logits, temp)
        taylor_logits = get_taylor(logit_signals, n)
        taylor_logit_sum = taylor_logits.sum(axis=1).reshape(-1, 1)
        true_logit = logit_signals.gather(1, all_true_labels.reshape(-1, 1))
        taylor_true_logit = taylor_logits.gather(1, all_true_labels.reshape(-1, 1))
        taylor_logit_sum = taylor_logit_sum - taylor_true_logit
        soft_taylor_true_logit = get_taylor(true_logit - m, n)
        taylor_logit_sum = taylor_logit_sum + soft_taylor_true_logit
        output_signals = torch.div(soft_taylor_true_logit, taylor_logit_sum)
    elif metric == 'logits':
        output_signals = all_logits
    elif metric == 'log-logit-scaling': 
        # Correct Logit signal used by LiRA from Membership Inference Attacks From First Principles https://arxiv.org/abs/2112.03570 
        # Taken and readapted from https://github.com/carlini/privacy/blob/better-mi/research/mi_lira_2021/score.py
        # Can be used to compute the loss as in https://github.com/yuan74/ml_privacy_meter/blob/2022_enhanced_mia/research/2022_enhanced_mia/plot_attack_via_reference_or_distill.py
        predictions = all_logits - torch.max(all_logits, dim=1, keepdim=True).values
        predictions = torch.exp(predictions)
        predictions = predictions/torch.sum(predictions,dim=1,keepdim=True)
        COUNT = predictions.shape[0]
        y_true = predictions[np.arange(COUNT),all_true_labels[:COUNT]]
        predictions[np.arange(COUNT),all_true_labels[:COUNT]] = 0
        y_wrong = torch.sum(predictions, dim=1)
        output_signals = (torch.log(y_true+1e-45) - torch.log(y_wrong+1e-45))  

    output_signals = torch.flatten(output_signals)
    return output_signals
