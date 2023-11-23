#!/bin/bash

target_idx="ten"

as=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for a in "${as[@]}";
do
    augmentation=augmented 
    nb_augmentation=18
    python main.py --cf attack_configs/cifar10/rmia_online_254_ref_models.yaml --audit.target_idx "${target_idx}" --audit.offline_a "${a}" --audit.report_log "report_rmia_a_${a}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}" --audit.offline "True"
done

for a in "${as[@]}";
do
    augmentation=augmented 
    nb_augmentation=18
    python main.py --cf attack_configs/cifar100/rmia_online_254_ref_models.yaml --audit.target_idx "${target_idx}" --audit.offline_a "${a}" --audit.report_log "report_rmia_a_${a}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}" --audit.offline "True"
done

for a in "${as[@]}";
do
    augmentation=augmented 
    nb_augmentation=18
    python main.py --cf attack_configs/cinic10/rmia_online_254_ref_models.yaml --audit.target_idx "${target_idx}" --audit.offline_a "${a}" --audit.report_log "report_rmia_a_${a}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}" --audit.offline "True"
done

for a in "${as[@]}";
do
    augmentation=none 
    nb_augmentation=2
    python main.py --cf attack_configs/purchase100/rmia_online_254_ref_models.yaml --audit.target_idx "${target_idx}" --audit.offline_a "${a}" --audit.report_log "report_rmia_a_${a}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}" --audit.offline "True"
done

python plot.py --figure 102