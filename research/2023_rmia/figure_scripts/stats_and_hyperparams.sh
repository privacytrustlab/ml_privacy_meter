#!/bin/bash

target_idx="ten"

ns=(3 4 5 6 )

for n in "${ns[@]}";
do
    augmentation=augmented
    nb_augmentation=18
    python main.py --cf attack_configs/cifar10/rmia_online_254_ref_models.yaml --audit.target_idx "${target_idx}" --audit.taylor_n "${n}" --audit.report_log "report_rmia_taylor_n_${n}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}"
done

ms=(0.4 0.5 0.6 0.7 0.8)

for m in "${ms[@]}";
do
    augmentation=augmented 
    nb_augmentation=18
    python main.py --cf attack_configs/cifar10/rmia_online_254_ref_models.yaml --audit.target_idx "${target_idx}" --audit.taylor_m "${m}" --audit.report_log "report_rmia_taylor_m_${m}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}"
done

temperatures=(1.0 1.5 2.0 2.5 3.0)

for temperature in "${temperatures[@]}";
do
    augmentation=augmented 
    nb_augmentation=18
    python main.py --cf attack_configs/cifar10/rmia_online_254_ref_models.yaml --audit.target_idx "${target_idx}" --audit.temperature "${temperature}" --audit.report_log "report_rmia_taylor_temperature_${temperature}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}"
done

python plot.py --figure 105