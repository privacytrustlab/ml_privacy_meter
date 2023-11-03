#!/bin/bash

# training
bash train_cifar10.sh
bash train_cifar100.sh
bash train_cinic10.sh

# inferring
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10/ --aug=18 --dataset=cifar10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar100/ --aug=18 --dataset=cifar100

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10/ --aug=18 --dataset=cinic10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/purchase100/ --aug=2 --dataset=purchase100 # no augmentation possible here