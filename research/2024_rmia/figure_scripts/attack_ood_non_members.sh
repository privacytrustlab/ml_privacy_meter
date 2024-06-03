# OOD non members
target_idx="ten"

as=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for a in "${as[@]}";
do
    augmentation=none 
    nb_augmentation=2
    python main_ood.py --cf attack_configs/cifar10/rmia_offline_127_ref_models_ood.yaml --audit.target_idx "${target_idx}" --audit.offline_a "${a}" --audit.report_log "report_relative_offline_full_ref_model_ood_a_${a}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}" --audit.offline "True"
    python main_ood.py --cf attack_configs/cifar10/rmia_offline_127_ref_models_noise.yaml --audit.target_idx "${target_idx}" --audit.offline_a "${a}" --audit.report_log "report_relative_offline_full_ref_model_noise_a_${a}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}" --audit.offline "True"
done

augmentation=none 
nb_augmentation=2
python main_ood.py --cf attack_configs/cifar10/rmia_offline_127_ref_models_ood.yaml --audit.target_idx "${target_idx}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}" --audit.offline "True"
python main_ood.py --cf attack_configs/cifar10/rmia_offline_127_ref_models_noise.yaml --audit.target_idx "${target_idx}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}" --audit.offline "True"

python main_ood.py --cf attack_configs/cifar10/attack_R_offline_127_ref_models_ood.yaml --audit.target_idx "${target_idx}" --audit.report_log "report_reference_full_ref_model_ood" --audit.offline "True"
python main_ood.py --cf attack_configs/cifar10/attack_R_offline_127_ref_models_noise.yaml --audit.target_idx "${target_idx}" --audit.report_log "report_reference_full_ref_model_noise" --audit.offline "True"

python main_ood.py --cf attack_configs/cifar10/attack_P_ood.yaml --audit.target_idx "${target_idx}" --audit.report_log "report_population_ood" --audit.offline "True"
python main_ood.py --cf attack_configs/cifar10/attack_P_noise.yaml --audit.target_idx "${target_idx}" --audit.report_log "report_population_noise" --audit.offline "True"

python main_ood.py --cf attack_configs/cifar10/lira_offline_127_ref_models_ood.yaml --audit.target_idx "${target_idx}" --audit.report_log "report_lira_offline_full_ref_model_ood" --audit.offline "True"
python main_ood.py --cf attack_configs/cifar10/lira_offline_127_ref_models_noise.yaml --audit.target_idx "${target_idx}" --audit.report_log "report_lira_offline_full_ref_model_noise" --audit.offline "True"

python main_ood.py --cf attack_configs/cifar10/rmia_offline_127_ref_models_ood.yaml --audit.target_idx "${target_idx}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}" --audit.offline "True"
python main_ood.py --cf attack_configs/cifar10/rmia_offline_127_ref_models_noise.yaml --audit.target_idx "${target_idx}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}" --audit.offline "True"

python plot.py --figure 301
python plot.py --figure 302

# Noisy members as Non members 
extra_target_dir="scripts/exp/cifar10_noisy_members_scale_0.1"
extra_reference_dir="scripts/exp/cifar10_noisy_members_scale_0.1"
prefix="cifar10_noisy_members_scale_0.1"
python main_noise_ood.py --cf attack_configs/cifar10/attack_R_offline_127_ref_models_noise.yaml --audit.report_log "${prefix}_reference" --audit.target_idx "${target_idx}" --audit.offline "True" --data.extra_target_dir "${extra_target_dir}" --data.extra_reference_dir "${extra_reference_dir}"
python main_noise_ood.py --cf attack_configs/cifar10/attack_P_noise.yaml --audit.report_log "${prefix}_population" --audit.target_idx "${target_idx}" --audit.offline "True" --data.extra_target_dir "${extra_target_dir}" --data.extra_reference_dir "${extra_reference_dir}"
python main_noise_ood.py --cf attack_configs/cifar10/lira_offline_127_ref_models_noise.yaml --audit.report_log "${prefix}_lira_offline" --audit.target_idx "${target_idx}" --audit.offline "True" --data.extra_target_dir "${extra_target_dir}" --data.extra_reference_dir "${extra_reference_dir}"

extra_target_dir="scripts/exp/cifar10_noisy_members_scale_0.4"
extra_reference_dir="scripts/exp/cifar10_noisy_members_scale_0.4"
prefix="cifar10_noisy_members_scale_0.4"
python main_noise_ood.py --cf attack_configs/cifar10/attack_R_offline_127_ref_models_noise.yaml --audit.report_log "${prefix}_reference" --audit.target_idx "${target_idx}" --audit.offline "True" --data.extra_target_dir "${extra_target_dir}" --data.extra_reference_dir "${extra_reference_dir}"
python main_noise_ood.py --cf attack_configs/cifar10/attack_P_noise.yaml --audit.report_log "${prefix}_population" --audit.target_idx "${target_idx}" --audit.offline "True" --data.extra_target_dir "${extra_target_dir}" --data.extra_reference_dir "${extra_reference_dir}"
python main_noise_ood.py --cf attack_configs/cifar10/lira_offline_127_ref_models_noise.yaml --audit.report_log "${prefix}_lira_offline" --audit.target_idx "${target_idx}"  --audit.offline "True" --data.extra_target_dir "${extra_target_dir}" --data.extra_reference_dir "${extra_reference_dir}"

as=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for a in "${as[@]}";
do
    extra_target_dir="scripts/exp/cifar10_noisy_members_scale_0.1"
    extra_reference_dir="scripts/exp/cifar10_noisy_members_scale_0.1"
    prefix="cifar10_noisy_members_scale_0.1"
    python main_noise_ood.py --cf attack_configs/cifar10/rmia_offline_127_ref_models_noise.yaml --audit.offline_a "${a}" --audit.report_log "${prefix}_rmia_a_${a}" --audit.target_idx "${target_idx}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}" --audit.offline "True" --data.extra_target_dir "${extra_target_dir}" --data.extra_reference_dir "${extra_reference_dir}"

    extra_target_dir="scripts/exp/cifar10_noisy_members_scale_0.4"
    extra_reference_dir="scripts/exp/cifar10_noisy_members_scale_0.4"
    prefix="cifar10_noisy_members_scale_0.4"
    python main_noise_ood.py --cf attack_configs/cifar10/rmia_offline_127_ref_models_noise.yaml --audit.offline_a "${a}" --audit.report_log "${prefix}_rmia_a_${a}" --audit.target_idx "${target_idx}" --audit.augmentation "${augmentation}" --audit.nb_augmentation "${nb_augmentation}" --audit.offline "True" --data.extra_target_dir "${extra_target_dir}" --data.extra_reference_dir "${extra_reference_dir}"
done

python plot.py --figure 401