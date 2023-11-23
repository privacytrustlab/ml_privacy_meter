#!/bin/bash
# RMIA with different ways to compute p(x|theta)
target_idx="ten"

### no augmentation
augmentation=none 
nb_augmentation=2

signal="softmax_relative"
python main.py --cf attack_configs/cifar10/rmia_online_254_ref_models.yaml --audit.target_idx "${target_idx}" --audit.signal "${signal}" --audit.report_log "softmax"

signal="sm_softmax_relative"
python main.py --cf attack_configs/cifar10/rmia_online_254_ref_models.yaml --audit.target_idx "${target_idx}" --audit.signal "${signal}" --audit.report_log "sm_softmax"

signal="taylor_softmax_relative"
python main.py --cf attack_configs/cifar10/rmia_online_254_ref_models.yaml --audit.target_idx "${target_idx}" --audit.signal "${signal}" --audit.report_log "taylor_softmax"

signal="sm_taylor_softmax_relative"
python main.py --cf attack_configs/cifar10/rmia_online_254_ref_models.yaml --audit.target_idx "${target_idx}" --audit.signal "${signal}" --audit.report_log "sm_taylor_softmax"

