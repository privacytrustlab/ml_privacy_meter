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



suffix="cnn16"
arch="cnn16-3-max"
epochs=100
# training 1 target model
prefix="cifar10_${suffix}"

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
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=${epochs} --save_steps=100 --arch ${arch} --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10

# training 2 ref models
prefix="cifar10_2_${suffix}"

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
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=${epochs} --save_steps=100 --arch ${arch} --num_experiments 2 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10







suffix="cnn32"
arch="cnn32-3-max"
epochs=100
# training 1 target model
prefix="cifar10_${suffix}"

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
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=${epochs} --save_steps=100 --arch ${arch} --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10

# training 2 ref models
prefix="cifar10_2_${suffix}"

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
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=${epochs} --save_steps=100 --arch ${arch} --num_experiments 2 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10




suffix="cnn64"
arch="cnn64-3-max"
epochs=100
# training 1 target model
prefix="cifar10_${suffix}"

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
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=${epochs} --save_steps=100 --arch ${arch} --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10

# training 2 ref models
prefix="cifar10_2_${suffix}"

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
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=${epochs} --save_steps=100 --arch ${arch} --num_experiments 2 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10






suffix="wrn28_1"
arch="wrn28-1"
epochs=100
# training 1 target model
prefix="cifar10_${suffix}"

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
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=${epochs} --save_steps=100 --arch ${arch} --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10

# training 2 ref models
prefix="cifar10_2_${suffix}"

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
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=${epochs} --save_steps=100 --arch ${arch} --num_experiments 2 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done



CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10




suffix="wrn28_10"
arch="wrn28-10"
epochs=100
# training 1 target model
prefix="cifar10_${suffix}"

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
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=${epochs} --save_steps=100 --arch ${arch} --num_experiments 256 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 --bs=1000 # contains the original query

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10 --bs=1000

# training 2 ref models
prefix="cifar10_2_${suffix}"

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
    train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=${epochs} --save_steps=100 --arch ${arch} --num_experiments 2 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
    eval ${train}
done

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=2 --dataset=cifar10 --bs=1000 # contains the original query

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/${prefix}/ --aug=18 --dataset=cifar10 --bs=1000