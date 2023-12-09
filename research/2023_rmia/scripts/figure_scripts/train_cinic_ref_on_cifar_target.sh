#!/bin/bash

# training 1 target model

prefix="cifar10"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."
else
    echo "Folder 'logs/${prefix}' already exists."
fi

n_models_end=9 # train 1 target models. Change to 255 to train the full set of 256 models

for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

## training CINIC-10 Reference (Can be copy pasted and renamed from CINIC-10 original trained models)

# # training 2 ref models
prefix="cinic10_2_on_cifar"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."

    cp exp/cinic10/*.npy exp/"${prefix}" # copy the training files from cinic10 to this set of models
else
    echo "Folder 'logs/${prefix}' already exists."
fi

n_models_end=1 # train 2 reference models

for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='1' python3 -u train.py --dataset=cinic10 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments 2 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done


# training 4 ref models
prefix="cinic10_4_on_cifar"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."

    cp exp/cinic10/*.npy exp/"${prefix}"
else
    echo "Folder 'logs/${prefix}' already exists."
fi

n_models_end=3 # train 4 reference models

for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='1' python3 -u train.py --dataset=cinic10 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments 4 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done


# training 8 ref models
prefix="cinic10_8_on_cifar"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."

    cp exp/cinic10/*.npy exp/"${prefix}"
else
    echo "Folder 'logs/${prefix}' already exists."
fi

n_models_end=7 # train 8 reference models

for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='1' python3 -u train.py --dataset=cinic10 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments 8 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

infer on CIFAR-10

# the following replaces and overwrites CINIC-10 x_train, y_train files with CIFAR-10's
cp exp/cifar10/*.npy exp/cinic10_2_on_cifar

cp exp/cifar10/*.npy exp/cinic10_4_on_cifar

cp exp/cifar10/*.npy exp/cinic10_8_on_cifar

# then infer on it
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_2_on_cifar/ --aug=2 --dataset=cifar10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_2_on_cifar/ --aug=18 --dataset=cifar10


CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_4_on_cifar/ --aug=2 --dataset=cifar10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_4_on_cifar/ --aug=18 --dataset=cifar10


CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_8_on_cifar/ --aug=2 --dataset=cifar10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_8_on_cifar/ --aug=18 --dataset=cifar10