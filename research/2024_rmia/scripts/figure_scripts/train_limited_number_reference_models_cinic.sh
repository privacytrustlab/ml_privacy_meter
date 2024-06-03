#!/bin/bash

# training
bash train_cinic10.sh

# infer for target and reference models

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10/ --aug=2 --dataset=cinic10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10/ --aug=18 --dataset=cinic10


CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_2/ --aug=2 --dataset=cinic10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_2/ --aug=18 --dataset=cinic10


CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_4/ --aug=2 --dataset=cinic10

CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cinic10_4/ --aug=18 --dataset=cinic10
