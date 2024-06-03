#!/bin/bash

python main.py --cf attack_configs/purchase100/attack_P.yaml

python main.py --cf attack_configs/purchase100/attack_R_offline_1_ref_models.yaml
python main.py --cf attack_configs/purchase100/lira_offline_1_ref_models.yaml
python main.py --cf attack_configs/purchase100/rmia_offline_1_ref_models.yaml

python main.py --cf attack_configs/purchase100/lira_online_2_ref_models.yaml
python main.py --cf attack_configs/purchase100/rmia_online_2_ref_models.yaml

# plotting using matplotlib
python plot.py --log_dir "demo_purchase100"