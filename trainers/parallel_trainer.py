"""This module enables parallel training of models across multiple GPUs."""

import torch
import torch.cuda.amp as amp
import torch.multiprocessing as mp
from typing import List, Dict, Any
import copy
import time
import logging
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import sys
from threading import Thread
from models.utils import get_model

def get_optimal_batch_size(gpu_type: str, base_batch_size: int) -> int:
    """Adjust batch size based on GPU type."""
    gpu_memory_map = {
        'T4': 16,    # g4dn - 16GB
        'A10G': 24,  # g5 - 24GB
        'V100': 16   # p3 - 16GB
    }
    
    # Detect GPU type
    gpu_name = torch.cuda.get_device_name(0)
    if 'A10G' in gpu_name:
        return int(base_batch_size * 1.5)  # 50% larger batches for A10G
    elif 'V100' in gpu_name:
        return int(base_batch_size * 1.2)  # 20% larger batches for V100
    return base_batch_size

def monitor_progress(progress_dict, stop_event):
    """Monitor and display training progress for all GPUs."""
    while not stop_event.is_set():
        output = "\r"
        active_gpus = sorted(progress_dict.keys())  # Sort GPUs for consistent display
        for gpu_id in active_gpus:
            info = progress_dict[gpu_id]
            output += f"GPU {gpu_id}: {info['epoch']}/100 ({info['loss']:.4f}|{info['acc']:.4f}) | "
        print(output, end='', flush=True)
        time.sleep(0.5)
    print()  # New line after monitoring ends

def train_model_on_gpu(gpu_id: int, model_name: str, dataset_name: str, 
                      dataset, train_indices: np.ndarray, test_indices: np.ndarray,
                      train_config: Dict, shared_dict: Dict, progress_dict: Dict,
                      model_idx: int, log_dir: str):
    """Train a single model on specified GPU"""
    from models.utils import get_model
    from dataset.utils import get_dataloader
    from trainers.default_trainer import train, inference
    
    try:
        device = f'cuda:{gpu_id}'
        train_config['device'] = device
        
        # Optimize batch size for GPU type
        original_batch_size = train_config['batch_size']
        train_config['batch_size'] = get_optimal_batch_size(
            gpu_type='auto', 
            base_batch_size=original_batch_size
        )
        
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        test_subset = torch.utils.data.Subset(dataset, test_indices)
        
        train_loader = get_dataloader(
            train_subset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=4,
        )
        test_loader = get_dataloader(
            test_subset,
            batch_size=train_config['batch_size'],
            num_workers=4,
        )
        
        # Initialize model
        model = get_model(model_name, dataset_name, {'train': train_config})
        model = model.to(device)

        # Enable mixed precision training
        scaler = torch.amp.GradScaler('cuda')

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )

        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_config['learning_rate'],
            epochs=train_config['epochs'],
            steps_per_epoch=len(train_loader)
        )

        # Training loop
        epochs = train_config['epochs']
        progress_dict[gpu_id] = {'epoch': 0, 'loss': 0.0, 'acc': 0.0}
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                with torch.amp.autocast('cuda'):
                    output = model(data)
                    loss = criterion(output, target)
    
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                scheduler.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

            # Update progress
            epoch_loss = total_loss / len(train_loader)
            epoch_acc = correct / total
            progress_dict[gpu_id] = {
                'epoch': epoch + 1,
                'loss': epoch_loss,
                'acc': epoch_acc
            }
            
        # Final evaluation and save
        test_loss, test_acc = inference(model, test_loader, device)
        train_loss, train_acc = inference(model, train_loader, device)
        
        save_path = f"{log_dir}/models/model_{model_idx}.pkl"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epochs,
            'batch_size': train_config['batch_size']
        }, save_path)
        
        shared_dict[model_idx] = {
            'train_acc': float(train_acc),
            'test_acc': float(test_acc),
            'train_loss': float(train_loss),
            'test_loss': float(test_loss),
            'model_path': save_path,
            'num_train': len(train_indices),
            'final_batch_size': train_config['batch_size']
        }
        
    except Exception as e:
        print(f"\nError on GPU {gpu_id}: {str(e)}", flush=True)
        shared_dict[model_idx] = {'error': str(e)}

def parallel_prepare_models(
    log_dir: str,
    dataset: torch.utils.data.Dataset,
    data_split_info: List[Dict],
    all_memberships: Any,
    configs: Dict,
    logger: logging.Logger,
    num_gpus: int = 4
) -> List[torch.nn.Module]:
    """Train models in parallel across multiple GPUs."""
    experiment_dir = f"{log_dir}/models"
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Training {len(data_split_info)} models using {num_gpus} GPUs")
    
    # Save memberships
    np.save(f"{log_dir}/memberships.npy", all_memberships)
    
    # Setup shared dictionaries for results and progress
    manager = mp.Manager()
    shared_dict = manager.dict()
    progress_dict = manager.dict()
    stop_monitor = manager.Event()
    
    # Start progress monitoring in a separate thread
    monitor_thread = Thread(target=monitor_progress, args=(progress_dict, stop_monitor))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Start training processes
    processes = []
    for idx, split_info in enumerate(data_split_info):
        gpu_id = idx % num_gpus
        p = mp.Process(
            target=train_model_on_gpu,
            args=(
                gpu_id,
                configs['train']['model_name'],
                configs['data']['dataset'],
                dataset,
                split_info['train'],
                split_info['test'],
                copy.deepcopy(configs['train']),
                shared_dict,
                progress_dict,
                idx,
                log_dir
            )
        )
        p.start()
        processes.append(p)
        
        # If we've filled all GPUs or this is the last model, wait for completion
        if (idx + 1) % num_gpus == 0 or idx == len(data_split_info) - 1:
            for p in processes:
                p.join()
            processes = []
    
    # Stop the monitor thread
    stop_monitor.set()
    monitor_thread.join()
    
    # Save metadata and return models
    metadata = {}
    for idx in range(len(data_split_info)):
        if idx in shared_dict:
            if 'error' not in shared_dict[idx]:
                metadata[str(idx)] = {
                    **shared_dict[idx],
                    "optimizer": "AdamW",  # Updated from SGD
                    "batch_size": configs['train']['batch_size'],
                    "epochs": configs['train']['epochs'],
                    "model_name": configs['train']['model_name'],
                    "learning_rate": configs['train']['learning_rate'],
                    "weight_decay": configs['train']['weight_decay'],
                    "dataset": configs['data']['dataset']
                }
            else:
                logger.error(f"Model {idx} failed with error: {shared_dict[idx]['error']}")

    with open(f"{log_dir}/models/models_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Load and return all models
    model_list = []
    for idx in range(len(data_split_info)):
        if idx in shared_dict and 'error' not in shared_dict[idx]:
            model = get_model(configs['train']['model_name'], configs['data']['dataset'], configs)
            saved_data = torch.load(shared_dict[idx]['model_path'])
            model.load_state_dict(saved_data['model_state_dict'])
            model_list.append(model)
    
    return model_list