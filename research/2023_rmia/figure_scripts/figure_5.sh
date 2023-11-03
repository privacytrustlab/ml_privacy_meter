#!/bin/bash

target_idx=0 # average over 10 target models from target_dir (e.g. model 0 to 9)
datasets=("cifar10" "cifar100" "cinic10" "purchase100")

for dataset in "${datasets[@]}";
do
    python main.py --cf "attack_configs/${dataset}/attack_P.yaml" --audit.report_log "report_population" --audit.target_idx "${target_idx}"

    prefix="18_aug_"
    python main.py --cf "attack_configs/${dataset}/attack_R_offline_127_ref_models.yaml" --audit.report_log "${prefix}report_reference" --audit.target_idx "${target_idx}"
    python main.py --cf "attack_configs/${dataset}/lira_offline_127_ref_models.yaml" --audit.report_log "${prefix}report_lira_offline" --audit.target_idx "${target_idx}"
    python main.py --cf "attack_configs/${dataset}/rmia_offline_127_ref_models.yaml" --audit.report_log "${prefix}report_relative_offline" --audit.target_idx "${target_idx}"
done

# plotting using matplotlib
python plot.py --figure 5