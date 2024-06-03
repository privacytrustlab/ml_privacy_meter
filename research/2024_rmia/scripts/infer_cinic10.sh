#!/bin/bash

#export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"

# infer for target models
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10/ --aug=2 --dataset=cinic10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10/ --aug=18 --dataset=cinic10

# infer for reference models
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_2/ --aug=2 --dataset=cinic10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_2/ --aug=18 --dataset=cinic10


CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_4/ --aug=2 --dataset=cinic10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_4/ --aug=18 --dataset=cinic10


CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_8/ --aug=2 --dataset=cinic10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_8/ --aug=18 --dataset=cinic10
