#!/bin/bash

target_idx=0 # average over 10 target models from target_dir (e.g. model 0 to 9)
datasets=("cifar10" "cifar100" "cinic10" "purchase100")

for dataset in "${datasets[@]}";
do
    python main.py --cf "attack_configs/${dataset}/rmia_direct_64.yaml" --audit.report_log "report_rmia_direct_64" --audit.target_idx "${target_idx}"
    python main.py --cf "attack_configs/${dataset}/rmia_online_254_ref_models.yaml" --audit.report_log "report_rmia_bayes_64"--audit.num_ref_models "64" --audit.target_idx "${target_idx}"
    python main.py --cf "attack_configs/${dataset}/rmia_direct_4.yaml" --audit.report_log "report_rmia_direct_4" --audit.target_idx "${target_idx}"
    python main.py --cf "attack_configs/${dataset}/rmia_online_254_ref_models.yaml" --audit.report_log "report_rmia_bayes_4"--audit.num_ref_models "4" --audit.target_idx "${target_idx}"
done

# plotting using matplotlib
python plot.py --figure 12
python plot.py --figure 13