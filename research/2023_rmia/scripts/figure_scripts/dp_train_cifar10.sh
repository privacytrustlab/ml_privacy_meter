#!/bin/bash

# training 256 target model


noise_multiplier=0.0 
l2_norm_clip=10
bs=1024
lr=0.1
prefix="dp_cifar10_noise_${noise_multiplier}_c_${l2_norm_clip}"

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
    train="CUDA_VISIBLE_DEVICES='1' python3 -u dp_train.py --lr=${lr} --batch=${bs} --l2_norm_clip=${l2_norm_clip} --noise_multiplier=${noise_multiplier} --dataset=cifar10 --epochs=100 --save_steps=100 --arch cnn32-3-max --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

CUDA_VISIBLE_DEVICES='1' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='1' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10


# # training 4 ref models
prefix="dp_cifar10_4_noise_${noise_multiplier}_c_${l2_norm_clip}"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."
else
    echo "Folder 'logs/${prefix}' already exists."
fi

n_models_end=3 # train 4 reference models

for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='1' python3 -u dp_train.py --lr=${lr} --batch=${bs} --l2_norm_clip=${l2_norm_clip} --noise_multiplier=${noise_multiplier} --dataset=cifar10 --epochs=100 --save_steps=100 --arch cnn32-3-max --num_experiments 4 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

CUDA_VISIBLE_DEVICES='1' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='1' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10




noise_multiplier=0.2 
l2_norm_clip=5
bs=1024
lr=0.1
prefix="dp_cifar10_noise_${noise_multiplier}_c_${l2_norm_clip}"

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
    train="CUDA_VISIBLE_DEVICES='1' python3 -u dp_train.py --lr=${lr} --batch=${bs} --l2_norm_clip=${l2_norm_clip} --noise_multiplier=${noise_multiplier} --dataset=cifar10 --epochs=100 --save_steps=100 --arch cnn32-3-max --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

CUDA_VISIBLE_DEVICES='1' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='1' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10


# # training 4 ref models
prefix="dp_cifar10_4_noise_${noise_multiplier}_c_${l2_norm_clip}"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."
else
    echo "Folder 'logs/${prefix}' already exists."
fi

n_models_end=3 # train 4 reference models

for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='1' python3 -u dp_train.py --lr=${lr} --batch=${bs} --l2_norm_clip=${l2_norm_clip} --noise_multiplier=${noise_multiplier} --dataset=cifar10 --epochs=100 --save_steps=100 --arch cnn32-3-max --num_experiments 4 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

CUDA_VISIBLE_DEVICES='1' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='1' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10









noise_multiplier=0.8 
l2_norm_clip=1
bs=1024
lr=0.1
prefix="dp_cifar10_noise_${noise_multiplier}_c_${l2_norm_clip}"

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
    train="CUDA_VISIBLE_DEVICES='1' python3 -u dp_train.py --lr=${lr} --batch=${bs} --l2_norm_clip=${l2_norm_clip} --noise_multiplier=${noise_multiplier} --dataset=cifar10 --epochs=100 --save_steps=100 --arch cnn32-3-max --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

CUDA_VISIBLE_DEVICES='1' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='1' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10


# # training 4 ref models
prefix="dp_cifar10_4_noise_${noise_multiplier}_c_${l2_norm_clip}"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."
else
    echo "Folder 'logs/${prefix}' already exists."
fi

n_models_end=3 # train 4 reference models

for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='1' python3 -u dp_train.py --lr=${lr} --batch=${bs} --l2_norm_clip=${l2_norm_clip} --noise_multiplier=${noise_multiplier} --dataset=cifar10 --epochs=100 --save_steps=100 --arch cnn32-3-max --num_experiments 4 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

CUDA_VISIBLE_DEVICES='1' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='1' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10