#!/bin/bash

target_idx=0

gammas=(1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0)

for gamma in "${gammas[@]}";
do
    ### no augmentation
    augmentation=none 
    nb_augmentation=2
    python main.py --cf attack_configs/cifar10/rmia_online_254_ref_models.yaml --audit.target_idx "${target_idx}" --audit.gamma "${gamma}" --audit.report_log "report_rmia_gamma_${gamma}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}"
done