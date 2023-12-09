#!/bin/bash

# training 256 target model
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


# # training 2 ref models
prefix="cinic10_2"

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
prefix="cinic10_4"

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
prefix="cinic10_8"

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