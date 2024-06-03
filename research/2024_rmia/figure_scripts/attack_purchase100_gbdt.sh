#!/bin/bash

depth=3
log_dir="demo_purchase100_gbdt_d_${depth}"
target_dir="scripts/exp/purchase100_gbdt_d_${depth}"
reference_dir="scripts/exp/purchase100_2_gbdt_d_${depth}"

python main.py --cf attack_configs/purchase100/attack_P.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}

python main.py --cf attack_configs/purchase100/attack_R_offline_1_ref_models.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}
python main.py --cf attack_configs/purchase100/lira_offline_1_ref_models.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}
python main.py --cf attack_configs/purchase100/rmia_offline_1_ref_models.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}

python main.py --cf attack_configs/purchase100/lira_online_2_ref_models.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}
python main.py --cf attack_configs/purchase100/rmia_online_2_ref_models.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}


depth=5
log_dir="demo_purchase100_gbdt_d_${depth}"
target_dir="scripts/exp/purchase100_gbdt_d_${depth}"
reference_dir="scripts/exp/purchase100_2_gbdt_d_${depth}"

python main.py --cf attack_configs/purchase100/attack_P.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}

python main.py --cf attack_configs/purchase100/attack_R_offline_1_ref_models.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}
python main.py --cf attack_configs/purchase100/lira_offline_1_ref_models.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}
python main.py --cf attack_configs/purchase100/rmia_offline_1_ref_models.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}

python main.py --cf attack_configs/purchase100/lira_online_2_ref_models.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}
python main.py --cf attack_configs/purchase100/rmia_online_2_ref_models.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}



depth=7
log_dir="demo_purchase100_gbdt_d_${depth}"
target_dir="scripts/exp/purchase100_gbdt_d_${depth}"
reference_dir="scripts/exp/purchase100_2_gbdt_d_${depth}"

python main.py --cf attack_configs/purchase100/attack_P.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}

python main.py --cf attack_configs/purchase100/attack_R_offline_1_ref_models.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}
python main.py --cf attack_configs/purchase100/lira_offline_1_ref_models.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}
python main.py --cf attack_configs/purchase100/rmia_offline_1_ref_models.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}

python main.py --cf attack_configs/purchase100/lira_online_2_ref_models.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}
python main.py --cf attack_configs/purchase100/rmia_online_2_ref_models.yaml --run.log_dir ${log_dir} --data.target_dir ${target_dir} --data.reference_dir ${reference_dir}

python plot.py --figure 204