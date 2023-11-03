#!/bin/bash

# training
bash train_cifar10.sh
bash train_cifar100.sh
bash train_cinic10.sh

# inferring
bash infer_cifar10.sh
bash infer_cifar100.sh
bash infer_cinic10.sh