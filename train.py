import torch
from torch import nn
import torch

import torch
from util import get_optimizer




def train(model, train_loader,configs):
    """Train the model based on on the train loader
    Args:
        model: Model for evaluation.
        train_loader: Data loader for training.
        configs (dict): Configurations for training.
    Return:
        model: Trained model.
    """
    assert all(name in configs for name in ['device','epochs','lr','optimizer','wd']), "Specify 'device','epochs','lr','optimizer','wd' for training models"
    assert type(train_loader)== torch.utils.data.DataLoader, "Input the correct data loader for training"
    
    device = configs['device']
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model,configs)
    for epoch_idx in range(configs['epochs']):
        train_loss = 0
        for batch_idx, (data,target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print(f'epoch:{epoch_idx}')
    model.to('cpu')
    
    return model


def inference(model,test_loader,device,is_train=False):
    """Evaluate the model performance on the test loader

    Args:
        model (_type_): Model for evaluation
        test_loader (_type_): Data Loader for testing
        device (str): GPU or CPU
        is_train (bool, optional): Whether test_loader is from the train dataset or test dataset. Defaults to False.
        is_back_cpu (bool, optional): Whether to put the model back to cpu.
    Return:
        loss (float): Loss for the given model on the test dataset.
        acc (float): Accuracy for the given model on the test dataset.
    """
    assert type(test_loader)== torch.utils.data.DataLoader, "Input the correct data loader for evaluating"
    
    model.eval()
    model.to(device)
    loss = 0
    acc = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (data,target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss += criterion(output,target).item()
            pred = output.data.max(1,keepdim=True)[1]
            acc += pred.eq(target.data.view_as(pred)).sum()

        loss /= len(test_loader)
        acc = float(acc)/len(test_loader.dataset)
        
    print(f"{'Train' if is_train else 'Test'} accuracy {acc}, loss {loss}")
    model.to("cpu")
    
    return loss,acc


