#!/bin/bash

max_depth=3
# training 256 target model
prefix="purchase100_gbdt_d_${max_depth}"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."

    cp exp/purchase100/*.npy exp/"${prefix}"
else
    echo "Folder 'logs/${prefix}' already exists."
fi

n_models_end=0 # train 1 target models. Change to 255 to train the full set of 256 models

for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train_infer_boosted_tree.py --max_depth=${max_depth} --dataset=purchase100 --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done


# training 2 ref models
prefix="purchase100_2_gbdt_d_${max_depth}"

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
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train_infer_boosted_tree.py --max_depth=${max_depth} --dataset=purchase100 --num_experiments 2 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done



max_depth=5
# training 256 target model
prefix="purchase100_gbdt_d_${max_depth}"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."

    cp exp/purchase100/*.npy exp/"${prefix}"
else
    echo "Folder 'logs/${prefix}' already exists."
fi

n_models_end=0 # train 1 target models. Change to 255 to train the full set of 256 models

for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train_infer_boosted_tree.py --max_depth=${max_depth} --dataset=purchase100 --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done


# training 2 ref models
prefix="purchase100_2_gbdt_d_${max_depth}"

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
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train_infer_boosted_tree.py --max_depth=${max_depth} --dataset=purchase100 --num_experiments 2 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done


max_depth=7
# training 256 target model
prefix="purchase100_gbdt_d_${max_depth}"

if [ ! -d "logs/${prefix}" ]; then
    # If it doesn't exist, create the folder
    mkdir "logs/${prefix}"
    mkdir "exp/${prefix}"
    echo "Folder 'logs/${prefix}' created."

    cp exp/purchase100/*.npy exp/"${prefix}"
else
    echo "Folder 'logs/${prefix}' already exists."
fi

n_models_end=0 # train 1 target models. Change to 255 to train the full set of 256 models

for model in $(seq 0 1 $n_models_end);
do
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train_infer_boosted_tree.py --max_depth=${max_depth} --dataset=purchase100 --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done


# training 2 ref models
prefix="purchase100_2_gbdt_d_${max_depth}"

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
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train_infer_boosted_tree.py --max_depth=${max_depth} --dataset=purchase100 --num_experiments 2 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

