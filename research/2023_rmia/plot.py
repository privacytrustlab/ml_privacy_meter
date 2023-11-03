"""This file contains ploting functions."""
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import auc


def plot_roc(fpr_list, tpr_list, roc_auc, path, log=False):
    range01 = np.linspace(0, 1)
    plt.fill_between(fpr_list, tpr_list, alpha=0.15)
    plt.plot(range01, range01, "--", label="Random guess")
    plt.plot(fpr_list, tpr_list, label="ROC curve")
    if log:
        plt.xlim([10e-6, 1])
        plt.ylim([10e-6, 1])
        plt.xscale("log")
        plt.yscale("log")
    else:
        plt.xlim([0, 1])
        plt.ylim([0, 1])
    plt.grid()
    plt.legend()
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    plt.title("ROC curve")
    plt.text(
        0.7,
        0.3,
        f"AUC = {roc_auc:.03f}",
        horizontalalignment="center",
        verticalalignment="center",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.savefig(
        fname=path,
        dpi=1000,
    )
    plt.clf()


def get_fpr_tpr(log_dir, report_folder, model_idx):
                
    if model_idx == "ten" or model_idx == "fifty" :
        if model_idx == "ten":
            nbmodels = 10
        elif model_idx == "fifty":
            nbmodels = 50
        all_fpr, all_tpr = [], []
        for k in range(nbmodels):
            filepath = f"{log_dir}/{report_folder}/attack_stats_{k}.npz"
            
            if os.path.isfile(filepath):
                stats = np.load(filepath, allow_pickle=True)
                fpr_list, tpr_list = stats["fpr_list"], stats["tpr_list"]
                all_fpr.append(fpr_list)
                all_tpr.append(tpr_list)
 
            else:
                print(f"{report_folder} NOT FOUND")
        
        if len(all_fpr) > 0:
            return np.mean(all_fpr, axis=0), np.mean(all_tpr, axis=0)
    else:
        filepath = f"{log_dir}/{report_folder}/attack_stats_{model_idx}.npz"

        if os.path.isfile(filepath):
            stats = np.load(filepath, allow_pickle=True)
            try :
                fpr_list, tpr_list = stats["fpr_list"], stats["tpr_list"]

                return fpr_list, tpr_list

            except Exception as e:
                print(e)
        else:
            print(f"{report_folder} NOT FOUND")

def metric_results(fpr_list, tpr_list):
    fprs = [0.01,0.001,0.0001,0.00001,0.0] # 1%, 0.1%, 0.01%, 0.001%, 0%
    tpr_dict = {}
    acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
    roc_auc = auc(fpr_list, tpr_list)

    for fpr in fprs:
        tpr_dict[fpr] = tpr_list[np.where(fpr_list <= fpr)[0][-1]]

    return roc_auc, acc, tpr_dict

def get_rocs_from(model_idx, attack_list_and_paths, log_dir, save_path, log=False):
    attacks_stats = {}
    aucs_list = {}
    nb_decimals = 3

    range01 = np.linspace(0, 1, 100)
    
    if log:
        plt.xlim([10e-6, 1])
        plt.ylim([10e-6, 1])
        plt.xscale("log")
        plt.yscale("log")
    else:
        plt.xlim([0, 1])
        plt.ylim([0, 1])
    
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    plt.title("ROC curves")

    plt.plot(range01, range01, "--", label="Random guess")

    for (attack, pa) in attack_list_and_paths:
        attacks_stats[attack] = get_fpr_tpr(log_dir=log_dir, report_folder=pa, model_idx=model_idx)
        aucs_list[attack] = np.round(metric_results(attacks_stats[attack][0], attacks_stats[attack][1])[0], nb_decimals)
        
        plt.plot(attacks_stats[attack][0], attacks_stats[attack][1], label=f"{attack} - AUC: {aucs_list[attack]}")
    
    plt.grid()
    plt.legend()
    plt.savefig(
        fname=save_path,
        dpi=1000,
    )
    
    plt.clf()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        default="demo_cifar10",
        help="Folder containing the attack results",
    )

    parser.add_argument(
        "--figure",
        type=int,
        default=0,
        help="Figure to plot (It needs to have executed scripts/figure_scripts/figure_N.sh then figure_scripts/figure_N.sh)",
    )

    # Load the parameters
    args, unknown = parser.parse_known_args()

    # this is hardcoded, change it to compare other attacks

    if args.figure == 0:

        # 1 Reference models offline
        model_idx = 0
        attack_list_and_paths = [
            ("RMIA", "report_relative_offline_1_ref_model"),
            ("Attack-R", "report_reference_1_ref_model"),
            ("Attack-P", "report_population"),
            ("LiRA", "report_lira_offline_1_ref_model"),
        ]

        get_rocs_from(model_idx, attack_list_and_paths, args.log_dir, f'{args.log_dir}/rocs_offline.png', log=False)
        get_rocs_from(model_idx, attack_list_and_paths, args.log_dir, f'{args.log_dir}/rocs_offline_log.png', log=True)

        # 2 Reference models online
        attack_list_and_paths = [
            ("RMIA", "report_relative_online_2_ref_model"),
            ("LiRA", "report_lira_online_2_ref_model"),
            # Note for attack-R, you need to train 4 reference models and use half as reference models.
        ]

        get_rocs_from(model_idx, attack_list_and_paths, args.log_dir, f'{args.log_dir}/rocs_online.png', log=False)
        get_rocs_from(model_idx, attack_list_and_paths, args.log_dir, f'{args.log_dir}/rocs_online_log.png', log=True) 
    elif args.figure == 1: # CIFAR-10 CIFAR-100
        log_dir = "demo_cifar10"
        # 1 Reference models offline
        model_idx = 0
        attack_list_and_paths = [
            ("RMIA", "report_relative_offline_1_ref_model"),
            ("Attack-R", "report_reference_1_ref_model"),
            ("Attack-P", "report_population"),
            ("LiRA", "report_lira_offline_1_ref_model"),
        ]

        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_1_rocs_offline.png', log=False)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_1_rocs_offline_log.png', log=True)

        log_dir = "demo_cifar100"
        # 1 Reference models offline
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_1_rocs_offline.png', log=False)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_1_rocs_offline_log.png', log=True)
    elif args.figure == 2: # CINIC-10 on 1, 2, 254 models
        log_dir = "demo_cinic10"
        # 1 Reference models offline
        model_idx = 0
        attack_list_and_paths = [
            ("RMIA", "report_relative_offline_1_ref_model"),
            ("Attack-R", "report_reference_1_ref_model"),
            ("Attack-P", "report_population"),
            ("LiRA", "report_lira_offline_1_ref_model"),
        ]

        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_2_a_rocs_offline.png', log=False)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_2_a_rocs_offline_log.png', log=True)

        # 2 Reference models online
        model_idx = 0
        attack_list_and_paths = [
            ("RMIA", "report_relative_online_2_ref_model"),
            ("LiRA", "report_lira_online_2_ref_model"),
            ("Attack-R", "report_reference_2_ref_model"),
            ("Attack-P", "report_population"),
        ]

        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_2_b_rocs_offline.png', log=False)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_2_b_rocs_offline_log.png', log=True)

        # 254 Reference models online
        model_idx = 0
        attack_list_and_paths = [
            ("RMIA", "report_relative_online_full_ref_model"),
            ("LiRA", "report_lira_online_full_ref_model"),
            ("Attack-R", "report_reference_full_ref_model"),
            ("Attack-P", "report_population"),
        ]

        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_2_c_rocs_offline.png', log=False)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_2_c_rocs_offline_log.png', log=True)
    elif args.figure == 3: # ROCS 1 OUT
        datasets = ["demo_cifar10", "demo_cifar100", "demo_cinic10", "demo_purchase100"]
        for log_dir in datasets:
            # 1 Reference models offline
            model_idx = 0
            attack_list_and_paths = [
                ("RMIA", "report_relative_offline_1_ref_model"),
                ("Attack-R", "report_reference_1_ref_model"),
                ("Attack-P", "report_population"),
                ("LiRA", "report_lira_offline_1_ref_model"),
            ]
            get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_3_rocs_offline.png', log=False)
            get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_3_rocs_offline_log.png', log=True)
    elif args.figure == 4: # ROCS 1 IN 1 OUT
        datasets = ["demo_cifar10", "demo_cifar100", "demo_cinic10", "demo_purchase100"]
        for log_dir in datasets:
            # 1 Reference models offline
            model_idx = 0
            attack_list_and_paths = [
                ("RMIA", "report_relative_online_2_ref_model"),
                ("LiRA", "report_lira_online_2_ref_model"),
                ("Attack-R", "report_reference_2_ref_model"),
                ("Attack-P", "report_population"),
            ]
            get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_4_rocs_offline.png', log=False)
            get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_4_rocs_offline_log.png', log=True)
    elif args.figure == 5: # ROCS 127 OUT
        datasets = ["demo_cifar10", "demo_cifar100", "demo_cinic10", "demo_purchase100"]
        for log_dir in datasets:
            # 1 Reference models offline
            model_idx = 0
            attack_list_and_paths = [
                ("RMIA", "report_relative_offline_full_ref_model"),
                ("LiRA", "report_lira_offline_full_ref_model"),
                ("Attack-R", "report_reference_full_ref_model"),
                ("Attack-P", "report_population"),
            ]
            get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_5_rocs_offline.png', log=False)
            get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_5_rocs_offline_log.png', log=True)
    elif args.figure == 6: # ROCS 127 IN 127 OUT
        datasets = ["demo_cifar10", "demo_cifar100", "demo_cinic10", "demo_purchase100"]
        for log_dir in datasets:
            # 1 Reference models offline
            model_idx = 0
            attack_list_and_paths = [
                ("RMIA", "report_relative_online_full_ref_model"),
                ("LiRA", "report_lira_online_full_ref_model"),
                ("Attack-R", "report_reference_full_ref_model"),
                ("Attack-P", "report_population"),
            ]
            get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_6_rocs_offline.png', log=False)
            get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_6_rocs_offline_log.png', log=True)
    elif args.figure == 12: # RMIA direct 64 total models (32 models per gaussian)
        datasets = ["demo_cifar10", "demo_cifar100", "demo_cinic10", "demo_purchase100"]
        for log_dir in datasets:
            # 1 Reference models offline
            model_idx = 0
            attack_list_and_paths = [
                ("RMIA-Bayes", "report_rmia_bayes_64"),
                ("RMIA-direct", "report_rmia_direct_64"),
            ]
            get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_12_rocs_offline.png', log=False)
            get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_12_rocs_offline_log.png', log=True)        
    elif args.figure == 13: # RMIA direct 4 total models (2 models per gaussian)
        datasets = ["demo_cifar10", "demo_cifar100", "demo_cinic10", "demo_purchase100"]
        for log_dir in datasets:
            # 1 Reference models offline
            model_idx = 0
            attack_list_and_paths = [
                ("RMIA-Bayes", "report_rmia_bayes_4"),
                ("RMIA-direct", "report_rmia_direct_4"),
            ]
            get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_13_rocs_offline.png', log=False)
            get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/figure_13_rocs_offline_log.png', log=True)

    else:
        raise Exception("Figure number is not implemented or incorrect")