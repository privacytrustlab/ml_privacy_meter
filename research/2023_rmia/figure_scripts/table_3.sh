#!/bin/bash

### using all 256 ref model
target_idx="ten" # average over 10 target models from target_dir (e.g. model 0 to 9)
datasets=("cifar10" "cifar100" "cinic10" "purchase100")

for dataset in "${datasets[@]}";
do
    ### no augmentation
    augmentation=none 
    nb_augmentation=2
    prefix="no_aug_"
    python main.py --cf "attack_configs/${dataset}/attack_R_offline_127_ref_models.yaml" --audit.report_log "${prefix}report_reference" --audit.target_idx "${target_idx}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}"
    python main.py --cf "attack_configs/${dataset}/attack_P.yaml" --audit.report_log "report_population" --audit.target_idx "${target_idx}"

    ### 18 queries
    augmentation=augmented 
    nb_augmentation=18
    prefix="18_aug_"
    python main.py --cf "attack_configs/${dataset}/lira_offline_127_ref_models.yaml" --audit.report_log "${prefix}report_lira_offline" --audit.target_idx "${target_idx}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}"
    python main.py --cf "attack_configs/${dataset}/rmia_offline_127_ref_models.yaml" --audit.report_log "${prefix}report_relative_offline" --audit.target_idx "${target_idx}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}"
    python main.py --cf "attack_configs/${dataset}/lira_online_254_ref_models.yaml" --audit.report_log "${prefix}report_lira_online" --audit.target_idx "${target_idx}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}"
    python main.py --cf "attack_configs/${dataset}/rmia_online_254_ref_models.yaml" --audit.report_log "${prefix}report_relative_online" --audit.target_idx "${target_idx}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}"
done

