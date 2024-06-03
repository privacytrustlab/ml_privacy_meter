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
n_models_end=0 # train 1 target models. Change to 255 to train the full set of 256 models
for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

# training 2 ref models
prefix="cifar10_2"
if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."
else
    echo "Folder 'logs/${prefix}' already exists."
fi
n_models_end=1 # train 2 reference models
for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments 2 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
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
n_models_end=0 # train 1 target models. Change to 255 to train the full set of 256 models
for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar100 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

# # training 2 ref models
prefix="cifar100_2"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."
else
    echo "Folder 'logs/${prefix}' already exists."
fi
n_models_end=1 # train 2 reference models
for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar100 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments 2 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

## Infer
# CIFAR-10
# infer for target models
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10/ --aug=18 --dataset=cifar10

# infer for reference models
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10_2/ --aug=2 --dataset=cifar10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10_2/ --aug=18 --dataset=cifar10
# CIFAR-100

# infer for target models
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar100/ --aug=2 --dataset=cifar100

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar100/ --aug=18 --dataset=cifar100

# infer for reference models
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar100_2/ --aug=2 --dataset=cifar100

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar100_2/ --aug=18 --dataset=cifar100
