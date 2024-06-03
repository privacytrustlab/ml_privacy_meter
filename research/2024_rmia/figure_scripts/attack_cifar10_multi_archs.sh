#!/bin/bash

folder="cifar10_diff_arch/cnn16"
python main.py --cf "attack_configs/${folder}/attack_P.yaml"

python main.py --cf "attack_configs/${folder}/attack_R_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_online_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_online_2_ref_models.yaml"


folder="cifar10_diff_arch/cnn16_on_wrn28-2"
python main.py --cf "attack_configs/${folder}/attack_P.yaml"

python main.py --cf "attack_configs/${folder}/attack_R_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_online_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_online_2_ref_models.yaml"




folder="cifar10_diff_arch/cnn32"
python main.py --cf "attack_configs/${folder}/attack_P.yaml"

python main.py --cf "attack_configs/${folder}/attack_R_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_online_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_online_2_ref_models.yaml"


folder="cifar10_diff_arch/cnn32_on_wrn28-2"
python main.py --cf "attack_configs/${folder}/attack_P.yaml"

python main.py --cf "attack_configs/${folder}/attack_R_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_online_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_online_2_ref_models.yaml"




folder="cifar10_diff_arch/cnn64"
python main.py --cf "attack_configs/${folder}/attack_P.yaml"

python main.py --cf "attack_configs/${folder}/attack_R_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_online_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_online_2_ref_models.yaml"

folder="cifar10_diff_arch/cnn64_on_wrn28-2"
python main.py --cf "attack_configs/${folder}/attack_P.yaml"

python main.py --cf "attack_configs/${folder}/attack_R_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_online_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_online_2_ref_models.yaml"




folder="cifar10_diff_arch/wrn28-1"
python main.py --cf "attack_configs/${folder}/attack_P.yaml"

python main.py --cf "attack_configs/${folder}/attack_R_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_online_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_online_2_ref_models.yaml"

folder="cifar10_diff_arch/wrn28-1_on_wrn28-2"
python main.py --cf "attack_configs/${folder}/attack_P.yaml"

python main.py --cf "attack_configs/${folder}/attack_R_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_online_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_online_2_ref_models.yaml"




folder="cifar10_diff_arch/wrn28-10"
python main.py --cf "attack_configs/${folder}/attack_P.yaml"

python main.py --cf "attack_configs/${folder}/attack_R_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_online_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_online_2_ref_models.yaml"


folder="cifar10_diff_arch/wrn28-10_on_wrn28-2"
python main.py --cf "attack_configs/${folder}/attack_P.yaml"

python main.py --cf "attack_configs/${folder}/attack_R_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/lira_online_2_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_offline_1_ref_models.yaml"
python main.py --cf "attack_configs/${folder}/rmia_online_2_ref_models.yaml"

python plot.py --figure 201
python plot.py --figure 202