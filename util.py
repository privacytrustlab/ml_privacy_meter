import numpy as np
import torch.optim as optim


def get_optimizer(model,configs):
    """Get the optimizer for the given model

    Args:
        model: The model we want to optimize
        configs (dict): Configurations for the optimizer

    Raises:
        NotImplementedError: Check if the optimizer is implemented.

    Returns:
        optim: Optimizer for the given model
    """
    if configs['optimizer'] == 'SGD':
        return optim.SGD(model.parameters(),lr = configs['lr'],momentum=configs['momentum'], weight_decay=configs['wd'])
    elif configs['optimizer'] == 'Adam':
        return optim.Adam(model.parameters(),lr = configs['lr'],weight_decay=configs['wd'])
    
    else:
        raise NotImplementedError(f"Optimizer {configs['optimizer']}  has not been implemented. Please choose from SGD or Adam")


def get_split(all_index,used_index, size,split_method):
    """Select points based on the splitting methods

    Args:
        all_index (list): All the possible dataset index list
        used_index (list): Index list of used points
        size (int): Size of the points needs to be selected
        split_method (str): Splitting method

    Raises:
        NotImplementedError: Check if the splitting the methods are implemented
        ValueError: Check if there are enough points to select
    Returns:
        List: List of index
    """
    if split_method == 'no_overlapping':
        selected_index = np.array([i for i in all_index if i not in used_index])
        if size <= len(selected_index): 
            selected_index = np.random.choice(selected_index,size,replace=False)
        else:
            raise ValueError("The remaining data points are not enough.")
    elif split_method == 'uniform':
        if size <= len(all_index):
            selected_index =  np.random.choice(all_index,size,replace=False)
        else:
            raise ValueError("The remaining data points are not enough.")
    else:
        raise NotImplementedError(f"{split_method} is not implemented. We only support uniform and no_overlapping.")

    return selected_index 