import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import logging
import copy
import pickle
import os



def get_dataset(dataset_name,log_dir):
    """Load the dataset 

    Args:
        dataset_name (str): Dataset name
        log_dir (str): Indicate the log directory for loading the dataset

    Raises:
        NotImplementedError: Check if the dataset has been implemented.

    Returns:
        Pytorch Dataset: Whole dataset.
    """
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
            
            all_data = train_data
            all_data.data = X
            all_data.targets = Y
            with open(f'{log_dir}/data.pkl','wb') as f:
                pickle.dump(all_data,f)
        else:
            raise NotImplementedError(f"{dataset_name} is not implemented")
        
    
    N = len(all_data)
    logging.info(f"the whole dataset size: {N}")   
    return all_data


def get_cifar10_subset(dataset, index,is_tensor=False):
    """Get a subset of the cifar10 dataset

    Args:
        dataset: Whole dataset
        index (List): List of index
        is_tensor (bool, optional): Whether to return tensors of the data. Defaults to False.

    Returns:
        selected_data: Dataset which only contains the data indicated by the index
    """
    selected_data = copy.deepcopy(dataset)
   
    selected_data.data = selected_data.data[index]
    selected_data.targets = list(np.array(selected_data.targets)[index])
    
    if is_tensor:
        selected_data.data = torch.from_numpy(selected_data.data).float().permute(0, 3, 1,2)/255 # channel first 
        selected_data.targets = torch.tensor(selected_data.targets)
        
    return selected_data