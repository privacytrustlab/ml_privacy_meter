#!/bin/bash

#export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"

# infer for target models
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar100/ --aug=2 --dataset=cifar100

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar100/ --aug=18 --dataset=cifar100

# infer for reference models
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar100_2/ --aug=2 --dataset=cifar100

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar100_2/ --aug=18 --dataset=cifar100


CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar100_4/ --aug=2 --dataset=cifar100

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar100_4/ --aug=18 --dataset=cifar100


CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar100_8/ --aug=2 --dataset=cifar100

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar100_8/ --aug=18 --dataset=cifar100