#!/bin/bash

folder="cifar10_diff_arch/cnn32_dp_noise_0.0_c_10"
python main.py --cf "attack_configs/${folder}/attack_P.yaml"

python main.py --cf "attack_configs/${folder}/attack_R_offline_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_offline_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_online_4_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_offline_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_online_4_ref_models.yaml"


folder="cifar10_diff_arch/cnn32_dp_noise_0.2_c_5"
python main.py --cf "attack_configs/${folder}/attack_P.yaml"

python main.py --cf "attack_configs/${folder}/attack_R_offline_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_offline_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_online_4_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_offline_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_online_4_ref_models.yaml"


folder="cifar10_diff_arch/cnn32_dp_noise_0.8_c_1"
python main.py --cf "attack_configs/${folder}/attack_P.yaml"

python main.py --cf "attack_configs/${folder}/attack_R_offline_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_offline_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_online_4_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_offline_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_online_4_ref_models.yaml"

python plot.py --figure 203