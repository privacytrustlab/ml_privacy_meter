import torch
from torch import nn
import torch

import torch
import logging
#todo: In this code, we provide the tutorials about auditing privacy risk for different types of games

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


from util import get_optimizer




def train(model, train_loader,configs,test_loader=None):
    """Train the model based on on the train loader
    Args:
        model: Model for evaluation.
        train_loader: Data loader for training.
        configs (dict): Configurations for training.
        test_loader (optional): Data Loader for evaluating.
    
    Return:
        model: Trained model.
    """
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
        
        print(f'epoch:{epoch_idx}')
        if test_loader is not None:
            inference(model,test_loader,device,is_train=False)
        inference(model,train_loader,device,is_train=True)
    model.to('cpu')
    
    return model


def inference(model,test_loader,device,is_train=False):
    """Evaluate the model performance on the test loader

    Args:
        model (_type_): Model for evaluation
        test_loader (_type_): Data Loader for testing
        device (str): GPU or CPU
        is_train (bool, optional): Whether test_loader is from the train dataset or test dataset. Defaults to False.
    """
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
        
    print(f"{'Train' if is_train else 'Test'} accuracy {acc}, loss {loss}")
    
