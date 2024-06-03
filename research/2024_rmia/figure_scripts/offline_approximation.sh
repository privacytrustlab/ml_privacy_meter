#!/bin/bash
# to vizualize p_IN and p_OUT using one query and all reference models
target_idx=0
datasets=("cifar10" "cifar100" "cinic10" "purchase100")

for dataset in "${datasets[@]}";
do
    ### no augmentation
    augmentation=none 
    nb_augmentation=2
    prefix="no_aug_"
    
    python main.py --cf "attack_configs/${dataset}/rmia_online_254_ref_models.yaml" --audit.report_log "${prefix}report_rmia" --audit.target_idx "${target_idx}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}"
done