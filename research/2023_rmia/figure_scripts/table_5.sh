#!/bin/bash
# RMIA with various number of z samples
target_idx="ten"

zs=(25 250 1250 2500 6250 12500 25000)

for top_k in "${zs[@]}";
do
    ### no augmentation
    augmentation=none 
    nb_augmentation=2
    python main.py --cf attack_configs/cifar10/rmia_online_254_ref_models.yaml --audit.target_idx "${target_idx}" --audit.top_k "${top_k}" --audit.report_log "report_rmia_z_${top_k}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}"
done