#!/bin/bash
# Compares RMIA with prior works using reference models trained on CINIC-10 to attack target on CIFAR-10
target_idx="ten"

num_ref_models=2
python main.py --cf attack_configs/cifar10/attack_R_offline_1_ref_models.yaml --audit.report_log cinic_ref_reference_1 --audit.allout "True" --data.reference_dir scripts/exp/cinic10_2_on_cifar --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models}"
python main.py --cf attack_configs/cifar10/lira_offline_1_ref_models.yaml --audit.report_log cinic_lira_offline_1 --audit.allout "True" --data.reference_dir scripts/exp/cinic10_2_on_cifar --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models}"
python main.py --cf attack_configs/cifar10/rmia_offline_1_ref_models.yaml --audit.report_log cinic_rmia_offline_1 --audit.allout "True" --data.reference_dir scripts/exp/cinic10_2_on_cifar --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models}"

num_ref_models=4
python main.py --cf attack_configs/cifar10/attack_R_offline_2_ref_models.yaml --audit.report_log cinic_ref_reference_2 --audit.allout "True" --data.reference_dir scripts/exp/cinic10_4_on_cifar --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models}"
python main.py --cf attack_configs/cifar10/lira_offline_2_ref_models.yaml --audit.report_log cinic_lira_offline_2 --audit.allout "True" --data.reference_dir scripts/exp/cinic10_4_on_cifar --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models}"
python main.py --cf attack_configs/cifar10/rmia_offline_2_ref_models.yaml --audit.report_log cinic_rmia_offline_2 --audit.allout "True" --data.reference_dir scripts/exp/cinic10_4_on_cifar --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models}"

num_ref_models=8
python main.py --cf attack_configs/cifar10/attack_R_offline_2_ref_models.yaml --audit.report_log cinic_ref_reference_4 --audit.allout "True" --data.reference_dir scripts/exp/cinic10_8_on_cifar --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models}"
python main.py --cf attack_configs/cifar10/lira_offline_2_ref_models.yaml --audit.report_log cinic_lira_offline_4 --audit.allout "True" --data.reference_dir scripts/exp/cinic10_8_on_cifar --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models}"
python main.py --cf attack_configs/cifar10/rmia_offline_2_ref_models.yaml --audit.report_log cinic_rmia_offline_4 --audit.allout "True" --data.reference_dir scripts/exp/cinic10_8_on_cifar --audit.target_idx "${target_idx}" --audit.num_ref_models "${num_ref_models}"