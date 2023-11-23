#!/bin/bash
# compares RMIA and prior works on only 1 reference model

python main.py --cf attack_configs/cifar10/attack_P.yaml

python main.py --cf attack_configs/cifar10/attack_R_offline_1_ref_models.yaml
python main.py --cf attack_configs/cifar10/lira_offline_1_ref_models.yaml
python main.py --cf attack_configs/cifar10/rmia_offline_1_ref_models.yaml

# plotting using matplotlib
python plot.py --log_dir "demo_cifar10"




python main.py --cf attack_configs/cifar100/attack_P.yaml

python main.py --cf attack_configs/cifar100/attack_R_offline_1_ref_models.yaml
python main.py --cf attack_configs/cifar100/lira_offline_1_ref_models.yaml
python main.py --cf attack_configs/cifar100/rmia_offline_1_ref_models.yaml

# plotting using matplotlib
python plot.py --figure 1