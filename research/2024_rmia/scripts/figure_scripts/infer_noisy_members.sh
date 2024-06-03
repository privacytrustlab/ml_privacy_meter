#!/bin/bash

#export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"

# infer for target models
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10_noisy_members_scale_0.1/ --aug=2 --dataset=cifar10 # contains the original query

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10_noisy_members_scale_0.1/ --aug=18 --dataset=cifar10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10_noisy_members_scale_0.4/ --aug=2 --dataset=cifar10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10_noisy_members_scale_0.4/ --aug=18 --dataset=cifar10
