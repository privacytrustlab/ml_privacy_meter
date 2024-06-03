#!/bin/bash
# Compares RMIA with prior works using limited number of reference models
target_idx="ten" # average over 10 target models from target_dir (e.g. model 0 to 9)
datasets=("cifar10" "cifar100" "cinic10")

for dataset in "${datasets[@]}";
do
    ### 1 ref model
    python main.py --cf "attack_configs/${dataset}/attack_R_offline_1_ref_models.yaml" --audit.target_idx "${target_idx}"
    python main.py --cf "attack_configs/${dataset}/lira_offline_1_ref_models.yaml" --audit.target_idx "${target_idx}"
    python main.py --cf "attack_configs/${dataset}/rmia_offline_1_ref_models.yaml" --audit.target_idx "${target_idx}"
    
    ### 2 ref models
    num_ref_models_offline="4"
    reference_dir_offline="scripts/exp/${dataset}_${num_ref_models_offline}"
    num_ref_models_online="2"
    
    reference_dir_online="scripts/exp/${dataset}_${num_ref_models_online}"

    python main.py --cf "attack_configs/${dataset}/attack_R_offline_2_ref_models.yaml" --audit.report_log "report_reference_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}"
    python main.py --cf "attack_configs/${dataset}/lira_offline_2_ref_models.yaml" --audit.report_log "report_lira_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}"
    python main.py --cf "attack_configs/${dataset}/rmia_offline_2_ref_models.yaml" --audit.report_log "report_relative_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}"
    python main.py --cf "attack_configs/${dataset}/lira_online_2_ref_models.yaml" --audit.report_log "report_lira_online_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_online}" --data.reference_dir "${reference_dir_online}"
    python main.py --cf "attack_configs/${dataset}/rmia_online_2_ref_models.yaml" --audit.report_log "report_relative_online_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_online}" --data.reference_dir "${reference_dir_online}"

    ### 4 ref models
    num_ref_models_offline="8"
    reference_dir_offline="scripts/exp/${dataset}_${num_ref_models_offline}"
    num_ref_models_online="4"
    reference_dir_online="scripts/exp/${dataset}_${num_ref_models_online}"

    python main.py --cf "attack_configs/${dataset}/attack_R_offline_2_ref_models.yaml" --audit.report_log "report_reference_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}"
    python main.py --cf "attack_configs/${dataset}/lira_offline_2_ref_models.yaml" --audit.report_log "report_lira_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}"
    python main.py --cf "attack_configs/${dataset}/rmia_offline_2_ref_models.yaml" --audit.report_log "report_relative_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}"
    python main.py --cf "attack_configs/${dataset}/lira_online_2_ref_models.yaml" --audit.report_log "report_lira_online_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_online}" --data.reference_dir "${reference_dir_online}"
    python main.py --cf "attack_configs/${dataset}/rmia_online_2_ref_models.yaml" --audit.report_log "report_relative_online_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_online}" --data.reference_dir "${reference_dir_online}"
done

