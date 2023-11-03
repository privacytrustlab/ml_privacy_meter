#!/bin/bash

### Attack-P
python main.py --cf attack_configs/cinic10/attack_P.yaml

### 1 ref model
python main.py --cf attack_configs/cinic10/attack_R_offline_1_ref_models.yaml
python main.py --cf attack_configs/cinic10/lira_offline_1_ref_models.yaml
python main.py --cf attack_configs/cinic10/rmia_offline_1_ref_models.yaml

### 2 ref models
python main.py --cf attack_configs/cinic10/attack_R_offline_2_ref_models.yaml
python main.py --cf attack_configs/cinic10/lira_online_2_ref_models.yaml
python main.py --cf attack_configs/cinic10/rmia_online_2_ref_models.yaml

### 254 ref models
python main.py --cf attack_configs/cinic10/attack_R_offline_127_ref_models.yaml
python main.py --cf attack_configs/cinic10/lira_online_254_ref_models.yaml
python main.py --cf attack_configs/cinic10/rmia_online_254_ref_models.yaml

# plotting using matplotlib
python plot.py --figure 2