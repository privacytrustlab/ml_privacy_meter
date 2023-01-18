from dataset import get_dataset, get_cifar10_subset
from core import *
import matplotlib.pyplot as plt
from privacy_meter.model import PytorchModelTensor
import os
import pickle
from pathlib import Path
import copy
import time
import pandas as pd
import seaborn as sns
import yaml
from privacy_meter.audit import Audit
import argparse
from torch import nn
import torch
import numpy as np
import logging
logging.basicConfig(filename='log_time.log',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    filemode='w',
                    datefmt='%Y-%m-%d %H:%M:%S')


def set_default(configs):
    """Set the default value for the config files

    Args:
        configs (dict): All the configuration information.
    """
    # Set the run configuration.
    with open('default.yaml', 'r') as f:
        default_configs = yaml.load(f, Loader=yaml.Loader)

    for key in ['run', 'data', 'audit', 'train']:
        if key not in configs:
            print(
                f'Warning: configurations in {key} is not specified. Set those to default value.')
            configs[key] = {}
        for var in default_configs[key]:
            if var not in configs[key]:
                configs[key][var] = default_configs[key][var]
                print(
                    f'Warning: {key}.{var} is not specified. Set it to default value {default_configs[key][var]}.')

    return configs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cf', type=str, default="config_model.yaml",
                        help='Yaml file which contains the configurations')

    # Load the parameters
    args = parser.parse_args()
    with open(args.cf, 'r') as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    configs = set_default(configs)

    # Set the random seed, log_dir and inference_game
    torch.manual_seed(configs['run']['random_seed'])
    global log_dir
    log_dir = configs['run']['log_dir']
    inference_game_type = configs['audit']['privacy_game'].upper()

    # Create folders for saving the logs if they do not exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(
        f"{log_dir}/{configs['audit']['report_log']}").mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Load or initialize models based on metadata
    if os.path.exists((f'{log_dir}/models_metadata.pkl')):
        with open(f'{log_dir}/models_metadata.pkl', 'rb') as f:
            model_metadata_list = pickle.load(f)
    else:
        model_metadata_list = {'model_metadata': {}, 'current_idx': 0}

    # Load the dataset
    baseline_time = time.time()
    dataset = get_dataset(configs['data']['dataset'], log_dir)

    # Check the auditing game. If we are interested in auditing the privacy risk for a model or a training algorithm (set of models trained using the same algorithm).
    if configs['audit']['privacy_game'] in ['avg_privacy_loss_training_algo', 'privacy_loss_model']:
        if configs['audit']['privacy_game'] == 'privacy_loss_model':
            assert configs['train']['num_target_model'] == 1, "only need one model for auditing the privacy risk for a trained model"
        elif configs['audit']['privacy_game'] == 'avg_privacy_loss_training_algo':
            assert configs['train']['num_target_model'] > 1, "need more models for computing the average privacy loss for an algorithm"
        else:
            raise ValueError(
                f"{configs['audit']['privacy_game']} has not been implemented")

        # Load the saved models which matches the requirments for reference models and target models
        if model_metadata_list['current_idx'] > 0:
            matched_idx = load_existing_target_model(
                len(dataset), model_metadata_list, configs)
            if configs['audit']['algorithm'] == 'reference' and matched_idx is not None:
                matched_reference_idx = load_existing_reference_models(
                    len(dataset), model_metadata_list, configs, matched_idx)
            else:
                matched_reference_idx = None
        else:
            matched_idx, matched_reference_idx = None, None

        # Prepare the datasets
        print(25*">"+"Prepare the the datasets")
        data_split_info = prepare_datasets(len(
            dataset), configs['train']['num_target_model'], configs['data'], model_metadata_list, matched_idx)
        logging.info(
            f'Prepare the datasets costs {time.time()-baseline_time} seconds')

        # Prepare the target models
        print(25*">"+"Prepare the the target models")
        baseline_time = time.time()
        model_list, model_metadata_list, matched_idx = prepare_models(
            log_dir, dataset, data_split_info, configs['train'], model_metadata_list, matched_idx)
        logging.info(
            f'Prepare the target model costs {time.time()-baseline_time} seconds')

        # Prepare the information sources
        print(25*">"+"Prepare the information source, including attack models")
        baseline_time = time.time()
        target_info_source, reference_info_source, metrics, log_dir_list, model_metadata_list = prepare_information_source(
            log_dir, dataset, data_split_info, model_list, configs['audit'], model_metadata_list, matched_reference_idx)
        logging.info(
            f'Prepare the information source costs {time.time()-baseline_time} seconds')

        # Call core of privacy meter
        print(25*">"+"Auditing the privacy risk")
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
        logging.info(
            f'Prepare privacy meter results costs {time.time()-baseline_time} seconds')

        # Generate the privacy risk report
        print(25*">"+"Generating privacy risk report")
        baseline_time = time.time()
        prepare_priavcy_risk_report(
            log_dir, audit_results, configs['audit'], save_path=f"{log_dir}/{configs['audit']['report_log']}")
        print(100*"#")

        logging.info(
            f'Prepare the plot for the privacy risk report costs {time.time()-baseline_time} seconds')
        logging.info(
            f'Run the priavcy meter for the all steps costs {time.time()-start_time} seconds')

    # Auditing the priavcy risk for an individual data point
    elif configs['audit']['privacy_game'] == 'privacy_loss_sample':
        # Construct the models trained on data indicated by train.idx and not trained on it
        in_configs = copy.deepcopy(configs)
        in_configs['train']['type'] = 'include'
        in_configs['train']['num_target_model'] = configs['train']['num_in_models']

        out_configs = copy.deepcopy(configs)
        out_configs['train']['type'] = 'exclude'
        out_configs['train']['num_target_model'] = configs['train']['num_out_models']

        # Load existing models that match the requirement
        matched_in_idx = load_existing_target_model(
            len(dataset), model_metadata_list, in_configs)
        matched_out_idx = load_existing_target_model(
            len(dataset), model_metadata_list, out_configs)

        # Train additional models if the existing models are not enough
        if len(matched_in_idx) < configs['train']['num_in_models']:
            data_split_info_in = prepare_datasets_for_sample_privacy_risk(len(dataset), configs['train']['num_in_models'], configs['train']['num_in_models'] - len(
                matched_in_idx), configs['train']['idx'], configs['data'], 'include', model_metadata_list)
            in_model_list, model_metadata_list, matched_in_idx = prepare_models(
                log_dir, dataset, data_split_info_in, configs['train'], model_metadata_list, matched_in_idx)
        else:
            in_model_list, model_metadata_list, matched_in_idx = prepare_models(log_dir, dataset, {'split': [
            ]}, configs['train'], model_metadata_list, matched_in_idx[:configs['train']['num_in_models']])

        if len(matched_out_idx) < configs['train']['num_out_models']:
            data_split_info_out = prepare_datasets_for_sample_privacy_risk(len(dataset), configs['train']['num_out_models'], configs['train']['num_out_models'] - len(
                matched_out_idx), configs['train']['idx'], configs['data'], 'exclude', model_metadata_list)
            out_model_list, model_metadata_list, matched_out_idx = prepare_models(
                log_dir, dataset, data_split_info_out, configs['train'], model_metadata_list, matched_out_idx)
        else:
            out_model_list, model_metadata_list, matched_out_idx = prepare_models(log_dir, dataset, {'split': [
            ]}, configs['train'], model_metadata_list, matched_out_idx[:configs['train']['num_out_models']])

        # Obtain models trained on train.idx and without it
        in_model_list_pm = [PytorchModelTensor(model_obj=model, loss_fn=nn.CrossEntropyLoss(
        ), batch_size=1000) for model in in_model_list]
        out_model_list_pm = [PytorchModelTensor(model_obj=model, loss_fn=nn.CrossEntropyLoss(
        ), batch_size=1000) for model in out_model_list]

        # Test the models' performance on the data indicated by the audit.idx
        target_data = get_cifar10_subset(
            dataset, [configs['audit']['idx']], is_tensor=True)
        in_signal = np.array([model.get_loss(
            target_data.data, target_data.targets).item() for model in in_model_list_pm])
        out_signal = np.array([model.get_loss(
            target_data.data, target_data.targets).item() for model in out_model_list_pm])

        # Rescale the loss
        in_signal = in_signal+0.000001  # avoid nan
        in_signal = np.log(
            np.divide(np.exp(- in_signal), (1 - np.exp(- in_signal))))
        out_signal = out_signal+0.000001  # avoid nan
        out_signal = np.log(np.divide(np.exp(- out_signal),
                            (1 - np.exp(- out_signal))))

        # Generate the privacy risk report
        labels = np.concatenate(
            [np.ones(in_signal.shape[0]), np.zeros(out_signal.shape[0])])
        histogram = sns.histplot(
            data=pd.DataFrame({
                'Signal': np.concatenate([in_signal, out_signal]),
                'Membership': [f"In ({configs['train']['idx']})" if y == 1 else f"Out ({configs['train']['idx']})" for y in labels]
            }),
            x='Signal',
            hue='Membership',
            element='step',
            kde=True
        )
        plt.grid()
        plt.xlabel(f"Signal value")
        plt.ylabel('Number of Models')
        plt.title(f"Signal histogram for data point {configs['audit']['idx']}")
        plt.savefig(
            f"{log_dir}/{configs['audit']['report_log']}/individual_pr_{configs['train']['idx']}_{configs['audit']['idx']}.png")
