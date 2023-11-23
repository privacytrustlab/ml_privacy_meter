#!/bin/bash

## Training
# CIFAR-10
prefix="cifar10"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."
else
    echo "Folder 'logs/${prefix}' already exists."
fi
n_models_end=255 # train 1 target models. Change to 255 to train the full set of 256 models
for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

# CIFAR-100
# training 1 target model
prefix="cifar100"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."
else
    echo "Folder 'logs/${prefix}' already exists."
fi
n_models_end=255 # train 1 target models. Change to 255 to train the full set of 256 models
for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar100 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done


# CINIC-10
# training 1 target model
prefix="cinic10"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."
    
else
    echo "Folder 'logs/${prefix}' already exists."
fi
n_models_end=255 # train 1 target models. Change to 255 to train the full set of 256 models
for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='1' python3 -u train.py --dataset=cinic10 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done


# Purchase-100
# training 1 target model
prefix="purchase100"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."
else
    echo "Folder 'logs/${prefix}' already exists."
fi
n_models_end=255 # train 1 target models. Change to 255 to train the full set of 256 models
for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=purchase100 --epochs=50 --save_steps=50 --arch mlp --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done


## Infer
# CIFAR-10
# infer for target models
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10/ --aug=18 --dataset=cifar10

# CIFAR-100
# infer for target models
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar100/ --aug=2 --dataset=cifar100

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar100/ --aug=18 --dataset=cifar100

# CINIC-10
# infer for target models
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10/ --aug=2 --dataset=cinic10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10/ --aug=18 --dataset=cinic10


# Purchase-100
# infer for target models
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/purchase100/ --aug=2 --dataset=purchase100 # no augmentation possible here

