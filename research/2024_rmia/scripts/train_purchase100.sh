#!/bin/bash

# training 256 target model
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


# training 2 ref models
prefix="purchase100_2"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."

    cp exp/purchase100/*.npy exp/"${prefix}"
else
    echo "Folder 'logs/${prefix}' already exists."
fi

n_models_end=1 # train 2 reference models

for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=purchase100 --epochs=50 --save_steps=50 --arch mlp --num_experiments 2 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done


# training 4 ref models
prefix="purchase100_4"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."

    cp exp/purchase100/*.npy exp/"${prefix}"
else
    echo "Folder 'logs/${prefix}' already exists."
fi

n_models_end=3 # train 4 reference models

for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=purchase100 --epochs=50 --save_steps=50 --arch mlp --num_experiments 4 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done


# training 8 ref models
prefix="purchase100_8"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."

    cp exp/purchase100/*.npy exp/"${prefix}"
else
    echo "Folder 'logs/${prefix}' already exists."
fi

n_models_end=7 # train 8 reference models

for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=purchase100 --epochs=50 --save_steps=50 --arch mlp --num_experiments 8 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done