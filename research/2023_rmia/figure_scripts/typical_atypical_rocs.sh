#!/bin/bash

target_idx="ten" # average over 10 target models from target_dir (e.g. model 0 to 9)
datasets=("cifar10")

for dataset in "${datasets[@]}";
do  
    ## typical attack
    prefix="typical"

    ### 1 ref model
    num_ref_models_online="1"
    python main.py --cf "attack_configs/${dataset}/attack_P.yaml" --audit.target_idx "${target_idx}" --audit.report_log "${prefix}_report_population" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/attack_R_offline_1_ref_models.yaml" --audit.target_idx "${target_idx}" --audit.report_log "${prefix}_report_reference_${num_ref_models_online}_ref_model" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/lira_offline_1_ref_models.yaml" --audit.target_idx "${target_idx}" --audit.report_log "${prefix}_report_lira_offline_${num_ref_models_online}_ref_model" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/rmia_offline_1_ref_models.yaml" --audit.target_idx "${target_idx}" --audit.report_log "${prefix}_report_relative_offline_${num_ref_models_online}_ref_model" --audit.subset "${prefix}"
     
    ### 2 ref models
    num_ref_models_offline="4"
    reference_dir_offline="scripts/exp/${dataset}_${num_ref_models_offline}"
    num_ref_models_online="2"
    
    reference_dir_online="scripts/exp/${dataset}_${num_ref_models_online}"

    python main.py --cf "attack_configs/${dataset}/attack_R_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_reference_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/lira_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_lira_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/rmia_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_relative_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/lira_online_2_ref_models.yaml" --audit.report_log "${prefix}_report_lira_online_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_online}" --data.reference_dir "${reference_dir_online}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/rmia_online_2_ref_models.yaml" --audit.report_log "${prefix}_report_relative_online_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_online}" --data.reference_dir "${reference_dir_online}" --audit.subset "${prefix}"

    ### 4 ref models
    num_ref_models_offline="8"
    reference_dir_offline="scripts/exp/${dataset}_${num_ref_models_offline}"
    num_ref_models_online="4"
    reference_dir_online="scripts/exp/${dataset}_${num_ref_models_online}"

    python main.py --cf "attack_configs/${dataset}/attack_R_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_reference_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/lira_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_lira_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/rmia_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_relative_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/lira_online_2_ref_models.yaml" --audit.report_log "${prefix}_report_lira_online_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_online}" --data.reference_dir "${reference_dir_online}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/rmia_online_2_ref_models.yaml" --audit.report_log "${prefix}_report_relative_online_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_online}" --data.reference_dir "${reference_dir_online}" --audit.subset "${prefix}"

    ### Full ref models
    num_ref_models_offline="254"
    reference_dir_offline="scripts/exp/${dataset}"
    num_ref_models_online="127"
    reference_dir_online="scripts/exp/${dataset}"

    python main.py --cf "attack_configs/${dataset}/attack_R_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_reference_127_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "254" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/lira_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_lira_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/rmia_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_relative_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/lira_online_2_ref_models.yaml" --audit.report_log "${prefix}_report_lira_online_254_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "254" --data.reference_dir "${reference_dir_online}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/rmia_online_2_ref_models.yaml" --audit.report_log "${prefix}_report_relative_online_254_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "254" --data.reference_dir "${reference_dir_online}" --audit.subset "${prefix}"



    ## atypical attack
    prefix="atypical"

    ### 1 ref model
    num_ref_models_online="1"
    python main.py --cf "attack_configs/${dataset}/attack_P.yaml" --audit.target_idx "${target_idx}" --audit.report_log "${prefix}_report_population" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/attack_R_offline_1_ref_models.yaml" --audit.target_idx "${target_idx}" --audit.report_log "${prefix}_report_reference_${num_ref_models_online}_ref_model" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/lira_offline_1_ref_models.yaml" --audit.target_idx "${target_idx}" --audit.report_log "${prefix}_report_lira_offline_${num_ref_models_online}_ref_model" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/rmia_offline_1_ref_models.yaml" --audit.target_idx "${target_idx}" --audit.report_log "${prefix}_report_relative_offline_${num_ref_models_online}_ref_model" --audit.subset "${prefix}"
     
    ### 2 ref models
    num_ref_models_offline="4"
    reference_dir_offline="scripts/exp/${dataset}_${num_ref_models_offline}"
    num_ref_models_online="2"
    
    reference_dir_online="scripts/exp/${dataset}_${num_ref_models_online}"

    python main.py --cf "attack_configs/${dataset}/attack_R_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_reference_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/lira_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_lira_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/rmia_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_relative_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/lira_online_2_ref_models.yaml" --audit.report_log "${prefix}_report_lira_online_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_online}" --data.reference_dir "${reference_dir_online}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/rmia_online_2_ref_models.yaml" --audit.report_log "${prefix}_report_relative_online_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_online}" --data.reference_dir "${reference_dir_online}" --audit.subset "${prefix}"

    ### 4 ref models
    num_ref_models_offline="8"
    reference_dir_offline="scripts/exp/${dataset}_${num_ref_models_offline}"
    num_ref_models_online="4"
    reference_dir_online="scripts/exp/${dataset}_${num_ref_models_online}"

    python main.py --cf "attack_configs/${dataset}/attack_R_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_reference_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/lira_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_lira_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/rmia_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_relative_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/lira_online_2_ref_models.yaml" --audit.report_log "${prefix}_report_lira_online_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_online}" --data.reference_dir "${reference_dir_online}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/rmia_online_2_ref_models.yaml" --audit.report_log "${prefix}_report_relative_online_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_online}" --data.reference_dir "${reference_dir_online}" --audit.subset "${prefix}"


    ### Full ref models
    num_ref_models_offline="254"
    reference_dir_offline="scripts/exp/${dataset}"
    num_ref_models_online="127"
    reference_dir_online="scripts/exp/${dataset}"

    python main.py --cf "attack_configs/${dataset}/attack_R_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_reference_127_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "254" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/lira_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_lira_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/rmia_offline_2_ref_models.yaml" --audit.report_log "${prefix}_report_relative_offline_${num_ref_models_online}_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models_offline}" --data.reference_dir "${reference_dir_offline}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/lira_online_2_ref_models.yaml" --audit.report_log "${prefix}_report_lira_online_254_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "254" --data.reference_dir "${reference_dir_online}" --audit.subset "${prefix}"
    python main.py --cf "attack_configs/${dataset}/rmia_online_2_ref_models.yaml" --audit.report_log "${prefix}_report_relative_online_254_ref_model" --audit.target_idx "${target_idx}" --audit.num_ref_models "254" --data.reference_dir "${reference_dir_online}" --audit.subset "${prefix}"

done


python plot.py --figure 104