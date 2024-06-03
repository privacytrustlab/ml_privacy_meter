#!/bin/bash

target_idx="ten"

gammas=(1 2 4 8 16 32 64)

for gamma in "${gammas[@]}";
do
    augmentation=augmented 
    nb_augmentation=18
    python main.py --cf attack_configs/cifar10/rmia_online_254_ref_models.yaml --audit.target_idx "${target_idx}" --audit.gamma "${gamma}" --audit.report_log "aug_18_report_rmia_gamma_${gamma}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}"
done

python plot.py --figure 101