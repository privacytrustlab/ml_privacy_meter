"""This file contains ploting functions."""
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import auc
import pandas as pd


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

def reformat_fpr_tpr(x, fpr_list, tpr_list):
    """
    convert the tpr list into a tpr_list of same shape than x
    """
    returned_tpr = []
    for fpr in x:
        returned_tpr.append(tpr_list[np.where(fpr_list <= fpr)[0][-1]])
    return np.array(x), np.array(returned_tpr)

def metric_results(fpr_list, tpr_list):
    fprs = [0.01,0.001,0.0001,0.00001,0.0] # 1%, 0.1%, 0.01%, 0.001%, 0%
    tpr_dict = {}
    acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
    roc_auc = auc(fpr_list, tpr_list)

    for fpr in fprs:
        tpr_dict[fpr] = tpr_list[np.where(fpr_list <= fpr)[0][-1]]

    return roc_auc, acc, tpr_dict

def get_results(log_dir, report_folder, model_idx):
    # log_dir="demo_carlini"
    # report_folder = f"report_online"
    # model_idx="ten"
                
    if model_idx == "ten" or model_idx == "fifty" or model_idx == "all" :
        if model_idx == "ten":
            nbmodels = 10
        elif model_idx == "fifty":
            nbmodels = 50
        elif model_idx == "all":
            nbmodels = 256

        aucs, accs = [], []
        onep, tenth, hundreth, thousandth, zeros = [], [], [], [], []
        for k in range(nbmodels):
            filepath = f"{log_dir}/{report_folder}/attack_stats_{k}.npz"
            
            if os.path.isfile(filepath):
                stats = np.load(filepath, allow_pickle=True)
                
                aucs.append(stats["all_aucs"])
                accs.append(stats["all_accs"])

                fpr_list, tpr_list = stats["fpr_list"], stats["tpr_list"]

                roc_auc, acc, tpr_dict = metric_results(fpr_list, tpr_list)

                onep.append(tpr_dict[0.01])
                tenth.append(tpr_dict[0.001])
                hundreth.append(tpr_dict[0.0001])
                thousandth.append(tpr_dict[0.00001])
                zeros.append(tpr_dict[0.0])
            else:
                print(f"{report_folder} NOT FOUND")
        
        if len(aucs) > 0:
            results = {}

            results["auc"] = [np.mean(aucs), np.std(aucs)]
            results["acc"] = [np.mean(accs), np.std(accs)]
            results[0.01] = [np.mean(onep), np.std(onep)]
            results[0.001] = [np.mean(tenth), np.std(tenth)]
            results[0.0001] = [np.mean(hundreth), np.std(hundreth)]
            results[0.00001] = [np.mean(thousandth), np.std(thousandth)]
            results[0.0] = [np.mean(zeros), np.std(zeros)]

            return results
    else:
        filepath = f"{log_dir}/{report_folder}/attack_stats_{model_idx}.npz"

        if os.path.isfile(filepath):
            stats = np.load(filepath, allow_pickle=True)
            try :
                fpr_list, tpr_list = stats["fpr_list"], stats["tpr_list"]
                roc_auc, acc, tpr_dict = metric_results(fpr_list, tpr_list)

                results = {}

                results["auc"] = [stats["all_aucs"], 0.0]
                results["acc"] = [stats["all_accs"], 0.0]
                results[0.01] = [tpr_dict[0.01], 0.0]
                results[0.001] = [tpr_dict[0.001], 0.0]
                results[0.0001] = [tpr_dict[0.0001], 0.0]
                results[0.00001] = [tpr_dict[0.00001], 0.0]
                results[0.0] = [tpr_dict[0.0], 0.0]

                return results 

            except Exception as e:
                print(e)
        else:
            print(f"{report_folder} model {model_idx} NOT FOUND")


def save_roc_to_csv(attacks_stats, attack_list, folder, path, x = np.logspace(-6, 0, 1000)):
    if isinstance(attack_list[0], str):

        data = {}
        aucs = {}
        accs = {}
        tenth = {}
        hundredth = {}
        zero = {}
        nb_decimals = 3
        for attack in attack_list: # fpr: attacks_stats[attack][0], tpr: attacks_stats[attack][1], betas_dict: attacks_stats[attack][2]
            data[attack] = reformat_fpr_tpr(x, attacks_stats[attack][0], attacks_stats[attack][1])[1]
            aucs[attack] = [np.round(metric_results(attacks_stats[attack][0], attacks_stats[attack][1])[0], nb_decimals)]
            accs[attack] = [np.round(metric_results(attacks_stats[attack][0], attacks_stats[attack][1])[1], nb_decimals)]
            tenth[attack] = reformat_fpr_tpr([0.001], attacks_stats[attack][0], attacks_stats[attack][1])[1]
            hundredth[attack] = reformat_fpr_tpr([0.0001], attacks_stats[attack][0], attacks_stats[attack][1])[1]
            zero[attack] = reformat_fpr_tpr([0.0], attacks_stats[attack][0], attacks_stats[attack][1])[1]
        
        data["random"] = x
        aucs["random"] = [0.5]
        accs["random"] = [0.5]
        tenth["random"] = [0.001]
        hundredth["random"] = [0.0001]
        zero["random"] = [0.0]

        attack_comparison_data = pd.DataFrame(data)
        attack_comparison_data.to_csv(f"{folder}/{path}_roc.csv", sep=',')

        attack_comparison_auc = pd.DataFrame(aucs)
        attack_comparison_auc.to_csv(f"{folder}/{path}_auc.csv", sep=',')

    
    if isinstance(attack_list[0], float) or isinstance(attack_list[0], int): # in case of attack are different gammas/hyperparams
        data = {}
        aucs = []
        accs = []
        tenth = []
        hundredth = []
        zero = []
        nb_decimals = 3
        for attack in attack_list:
            data[attack] = reformat_fpr_tpr(x, attacks_stats[attack][0], attacks_stats[attack][1])[1]
            aucs.append(np.round(metric_results(attacks_stats[attack][0], attacks_stats[attack][1])[0], nb_decimals))
            accs.append(np.round(metric_results(attacks_stats[attack][0], attacks_stats[attack][1])[1], nb_decimals))
            tenth.append(reformat_fpr_tpr([0.001], attacks_stats[attack][0], attacks_stats[attack][1])[1][0])
            hundredth.append(reformat_fpr_tpr([0.0001], attacks_stats[attack][0], attacks_stats[attack][1])[1][0])
            zero.append(reformat_fpr_tpr([0.0], attacks_stats[attack][0], attacks_stats[attack][1])[1][0])

        final_dict = {
            "x":attack_list,
            "auc":aucs,
            "acc":accs,
            "tenth":tenth,
            "hundredth":hundredth,
            "zero":zero
        }
        print(len(x), len(aucs), len(accs))

        data["random"] = x

        attack_comparison_data = pd.DataFrame(data)
        attack_comparison_data.to_csv(f"{folder}/{path}_roc.csv", sep=',')

        attack_comparison_data = pd.DataFrame(final_dict)
        attack_comparison_data.to_csv(f"{folder}/{path}_all_data.csv", sep=',')


def get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder):
    attacks_stats = {}
    attack_list = []
    for (attack, pa) in attack_list_and_paths:
        attacks_stats[attack] = get_fpr_tpr(log_dir=log_dir, report_folder=pa, model_idx=model_idx)
        attack_list.append(attack)

    save_roc_to_csv(attacks_stats, attack_list, folder, path, x = np.logspace(-6, 0, 1000))

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
    
    elif args.figure == 101: # AUC vs. gamma
        log_dir = "demo_cifar10"
        model_idx = "ten"
        gammas = [1, 2, 4, 8, 16, 32, 64]

        attacks_stats = {}
        attack_list = []
        final_dict = {}
        
        attack_list_and_paths = [
            (gamma, f"aug_18_report_rmia_gamma_{gamma}") for gamma in gammas
        ]

        for (attack, path) in attack_list_and_paths:
            attacks_stats[attack] = get_results(log_dir=log_dir, report_folder=path, model_idx=model_idx)

            final_dict[attack] = np.array([
                attacks_stats[attack]["auc"][0]*100,attacks_stats[attack]["auc"][1]*100, 
                attacks_stats[attack][0.001][0]*100,attacks_stats[attack][0.001][1]*100,
                attacks_stats[attack][0.0001][0]*100,attacks_stats[attack][0.0001][1]*100,
                attacks_stats[attack][0.0][0]*100,attacks_stats[attack][0.0][1]*100,
            ])

            final_dict[attack] = ["%.2f" % x for x in final_dict[attack]]

        folder = "../../paper/data"
        path = "stats_and_gammas"

        attack_comparison_data = pd.DataFrame(final_dict).T
        attack_comparison_data.to_csv(f"{folder}/{path}.csv", sep=',')
        
            
    elif args.figure == 102: # AUC vs. offline_a
        ## offline a
        log_dir = "demo_cifar10"
        model_idx = "ten"
        all_as = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        attacks_stats = {}
        attack_list = []
        final_dict = {}
        
        attack_list_and_paths = [
            (a, f"report_rmia_a_{a}") for a in all_as
        ]

        for (attack, path) in attack_list_and_paths:
            attacks_stats[attack] = get_results(log_dir=log_dir, report_folder=path, model_idx=model_idx)

            final_dict[attack] = np.array([
                attacks_stats[attack]["auc"][0]*100,attacks_stats[attack]["auc"][1]*100, 
                attacks_stats[attack][0.001][0]*100,attacks_stats[attack][0.001][1]*100,
                attacks_stats[attack][0.0001][0]*100,attacks_stats[attack][0.0001][1]*100,
                attacks_stats[attack][0.0][0]*100,attacks_stats[attack][0.0][1]*100,
            ])

            final_dict[attack] = ["%.2f" % x for x in final_dict[attack]]

        folder = "../../paper/data"
        path = "stats_and_offline_as"

        attack_comparison_data = pd.DataFrame(final_dict).T
        attack_comparison_data.to_csv(f"{folder}/{path}.csv", sep=',')






        ## offline a
        log_dir = "demo_cifar100"
        model_idx = "ten"
        all_as = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        attacks_stats = {}
        attack_list = []
        final_dict = {}
        
        attack_list_and_paths = [
            (a, f"report_rmia_a_{a}") for a in all_as
        ]

        for (attack, path) in attack_list_and_paths:
            attacks_stats[attack] = get_results(log_dir=log_dir, report_folder=path, model_idx=model_idx)

            final_dict[attack] = np.array([
                attacks_stats[attack]["auc"][0]*100,attacks_stats[attack]["auc"][1]*100, 
                attacks_stats[attack][0.001][0]*100,attacks_stats[attack][0.001][1]*100,
                attacks_stats[attack][0.0001][0]*100,attacks_stats[attack][0.0001][1]*100,
                attacks_stats[attack][0.0][0]*100,attacks_stats[attack][0.0][1]*100,
            ])

            final_dict[attack] = ["%.2f" % x for x in final_dict[attack]]

        folder = "../../paper/data"
        path = "stats_and_offline_as_cifar100"

        attack_comparison_data = pd.DataFrame(final_dict).T
        attack_comparison_data.to_csv(f"{folder}/{path}.csv", sep=',')








        ## offline a
        log_dir = "demo_cinic10"
        model_idx = "ten"
        all_as = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        attacks_stats = {}
        attack_list = []
        final_dict = {}
        
        attack_list_and_paths = [
            (a, f"report_rmia_a_{a}") for a in all_as
        ]

        for (attack, path) in attack_list_and_paths:
            attacks_stats[attack] = get_results(log_dir=log_dir, report_folder=path, model_idx=model_idx)

            final_dict[attack] = np.array([
                attacks_stats[attack]["auc"][0]*100,attacks_stats[attack]["auc"][1]*100, 
                attacks_stats[attack][0.001][0]*100,attacks_stats[attack][0.001][1]*100,
                attacks_stats[attack][0.0001][0]*100,attacks_stats[attack][0.0001][1]*100,
                attacks_stats[attack][0.0][0]*100,attacks_stats[attack][0.0][1]*100,
            ])

            final_dict[attack] = ["%.2f" % x for x in final_dict[attack]]

        folder = "../../paper/data"
        path = "stats_and_offline_as_cinic10"

        attack_comparison_data = pd.DataFrame(final_dict).T
        attack_comparison_data.to_csv(f"{folder}/{path}.csv", sep=',')






        ## offline a
        log_dir = "demo_purchase100"
        model_idx = "ten"
        all_as = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        attacks_stats = {}
        attack_list = []
        final_dict = {}
        
        attack_list_and_paths = [
            (a, f"report_rmia_a_{a}") for a in all_as
        ]

        for (attack, path) in attack_list_and_paths:
            attacks_stats[attack] = get_results(log_dir=log_dir, report_folder=path, model_idx=model_idx)

            final_dict[attack] = np.array([
                attacks_stats[attack]["auc"][0]*100,attacks_stats[attack]["auc"][1]*100, 
                attacks_stats[attack][0.001][0]*100,attacks_stats[attack][0.001][1]*100,
                attacks_stats[attack][0.0001][0]*100,attacks_stats[attack][0.0001][1]*100,
                attacks_stats[attack][0.0][0]*100,attacks_stats[attack][0.0][1]*100,
            ])

            final_dict[attack] = ["%.2f" % x for x in final_dict[attack]]

        folder = "../../paper/data"
        path = "stats_and_offline_as_purchase100"

        attack_comparison_data = pd.DataFrame(final_dict).T
        attack_comparison_data.to_csv(f"{folder}/{path}.csv", sep=',')


    elif args.figure == 103: 
        pass
    elif args.figure == 104: # ROC on typical and atypical data
        log_dir = "demo_cifar10"
        model_idx = "ten" # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        prefix = "typical"
        attack_list_and_paths = [
                        (f"lira 1", f"{prefix}_report_lira_offline_1_ref_model"),
                        (f"lira 2", f"{prefix}_report_lira_online_2_ref_model"),
                        (f"lira 4", f"{prefix}_report_lira_online_4_ref_model"),
                        (f"lira 254", f"{prefix}_report_lira_online_254_ref_model"),

                        (f"relative 1", f"{prefix}_report_relative_offline_1_ref_model"),
                        (f"relative 2", f"{prefix}_report_relative_online_2_ref_model"),
                        (f"relative 4", f"{prefix}_report_relative_online_4_ref_model"),
                        (f"relative 254", f"{prefix}_report_relative_online_254_ref_model"),

                        (f"reference 1", f"{prefix}_report_reference_1_ref_model"),
                        (f"reference 2", f"{prefix}_report_reference_2_ref_model"),
                        (f"reference 4", f"{prefix}_report_reference_4_ref_model"),
                        (f"reference 127", f"{prefix}_report_reference_127_ref_model"),

                        ("population", f"{prefix}_report_population"),
                    ]
        folder = "../../paper/data"
        path = "typical_rocs"
        get_csv_rocs_from(6, attack_list_and_paths, log_dir, path, folder)




        log_dir = "demo_cifar10"
        model_idx = "ten" # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        prefix = "atypical"
        attack_list_and_paths = [
                        (f"lira 1", f"{prefix}_report_lira_offline_1_ref_model"),
                        (f"lira 2", f"{prefix}_report_lira_online_2_ref_model"),
                        (f"lira 4", f"{prefix}_report_lira_online_4_ref_model"),
                        (f"lira 254", f"{prefix}_report_lira_online_254_ref_model"),

                        (f"relative 1", f"{prefix}_report_relative_offline_1_ref_model"),
                        (f"relative 2", f"{prefix}_report_relative_online_2_ref_model"),
                        (f"relative 4", f"{prefix}_report_relative_online_4_ref_model"),
                        (f"relative 254", f"{prefix}_report_relative_online_254_ref_model"),

                        (f"reference 1", f"{prefix}_report_reference_1_ref_model"),
                        (f"reference 2", f"{prefix}_report_reference_2_ref_model"),
                        (f"reference 4", f"{prefix}_report_reference_4_ref_model"),
                        (f"reference 127", f"{prefix}_report_reference_127_ref_model"),

                        ("population", f"{prefix}_report_population"),
                    ]
        folder = "../../paper/data"
        path = "atypical_rocs"
        get_csv_rocs_from(6, attack_list_and_paths, log_dir, path, folder)

    elif args.figure == 105: # AUC vs. hyperparameters
        ## taylor n 
        log_dir = "demo_cifar10"
        model_idx = "ten"
        hyperparams = [3, 4, 5, 6]

        attacks_stats = {}
        attack_list = []
        final_dict = {}
        
        attack_list_and_paths = [
            (param, f"report_rmia_taylor_n_{param}") for param in hyperparams
        ]

        for (attack, path) in attack_list_and_paths:
            attacks_stats[attack] = get_results(log_dir=log_dir, report_folder=path, model_idx=model_idx)

            final_dict[attack] = np.array([
                attacks_stats[attack]["auc"][0]*100,attacks_stats[attack]["auc"][1]*100, 
                attacks_stats[attack][0.001][0]*100,attacks_stats[attack][0.001][1]*100,
                attacks_stats[attack][0.0001][0]*100,attacks_stats[attack][0.0001][1]*100,
                attacks_stats[attack][0.0][0]*100,attacks_stats[attack][0.0][1]*100,
            ])

            final_dict[attack] = ["%.2f" % x for x in final_dict[attack]]

        folder = "../../paper/data"
        path = "stats_and_ns"

        attack_comparison_data = pd.DataFrame(final_dict).T
        attack_comparison_data.to_csv(f"{folder}/{path}.csv", sep=',')


        ## taylor m
        log_dir = "demo_cifar10"
        model_idx = "ten"
        hyperparams = [0.4, 0.5, 0.6, 0.7, 0.8]

        attacks_stats = {}
        attack_list = []
        final_dict = {}
        
        attack_list_and_paths = [
            (param, f"report_rmia_taylor_m_{param}") for param in hyperparams
        ]

        for (attack, path) in attack_list_and_paths:
            attacks_stats[attack] = get_results(log_dir=log_dir, report_folder=path, model_idx=model_idx)

            final_dict[attack] = np.array([
                attacks_stats[attack]["auc"][0]*100,attacks_stats[attack]["auc"][1]*100, 
                attacks_stats[attack][0.001][0]*100,attacks_stats[attack][0.001][1]*100,
                attacks_stats[attack][0.0001][0]*100,attacks_stats[attack][0.0001][1]*100,
                attacks_stats[attack][0.0][0]*100,attacks_stats[attack][0.0][1]*100,
            ])

            final_dict[attack] = ["%.2f" % x for x in final_dict[attack]]

        folder = "../../paper/data"
        path = "stats_and_ms"

        attack_comparison_data = pd.DataFrame(final_dict).T
        attack_comparison_data.to_csv(f"{folder}/{path}.csv", sep=',')




        ## temperature n 
        log_dir = "demo_cifar10"
        model_idx = "ten"
        hyperparams = [1.0, 1.5, 2.0, 2.5, 3.0]

        attacks_stats = {}
        attack_list = []
        final_dict = {}
        
        attack_list_and_paths = [
            (param, f"report_rmia_taylor_temperature_{param}") for param in hyperparams
        ]

        for (attack, path) in attack_list_and_paths:
            attacks_stats[attack] = get_results(log_dir=log_dir, report_folder=path, model_idx=model_idx)

            final_dict[attack] = np.array([
                attacks_stats[attack]["auc"][0]*100,attacks_stats[attack]["auc"][1]*100, 
                attacks_stats[attack][0.001][0]*100,attacks_stats[attack][0.001][1]*100,
                attacks_stats[attack][0.0001][0]*100,attacks_stats[attack][0.0001][1]*100,
                attacks_stats[attack][0.0][0]*100,attacks_stats[attack][0.0][1]*100,
            ])

            final_dict[attack] = ["%.2f" % x for x in final_dict[attack]]

        folder = "../../paper/data"
        path = "stats_and_temperature"

        attack_comparison_data = pd.DataFrame(final_dict).T
        attack_comparison_data.to_csv(f"{folder}/{path}.csv", sep=',')
    
    elif args.figure == 201: # ROCs on multiple architecture (same target and ref arch)
        suffix="cifar10_cnn16"
        log_dir = f"demo_{suffix}"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        attack_list_and_paths = [
                        (f"lira 1", f"report_lira_offline_1_ref_model"),
                        (f"lira 2", f"report_lira_online_2_ref_model"),
                        (f"relative 1", f"report_relative_offline_1_ref_model"),
                        (f"relative 2", f"report_relative_online_2_ref_model"),
                        (f"reference 1", f"report_reference_1_ref_model"),
                        ("population", f"report_population"),
                    ]
        folder = "../../paper/data/multi"
        path = suffix
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/multi_arch_{suffix}_log.png', log=True)

        suffix="cifar10_cnn32"
        log_dir = f"demo_{suffix}"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        attack_list_and_paths = [
                        (f"lira 1", f"report_lira_offline_1_ref_model"),
                        (f"lira 2", f"report_lira_online_2_ref_model"),
                        (f"relative 1", f"report_relative_offline_1_ref_model"),
                        (f"relative 2", f"report_relative_online_2_ref_model"),
                        (f"reference 1", f"report_reference_1_ref_model"),
                        ("population", f"report_population"),
                    ]
        folder = "../../paper/data/multi"
        path = suffix
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/multi_arch_{suffix}_log.png', log=True)

        suffix="cifar10_cnn64"
        log_dir = f"demo_{suffix}"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        attack_list_and_paths = [
                        (f"lira 1", f"report_lira_offline_1_ref_model"),
                        (f"lira 2", f"report_lira_online_2_ref_model"),
                        (f"relative 1", f"report_relative_offline_1_ref_model"),
                        (f"relative 2", f"report_relative_online_2_ref_model"),
                        (f"reference 1", f"report_reference_1_ref_model"),
                        ("population", f"report_population"),
                    ]
        folder = "../../paper/data/multi"
        path = suffix
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/multi_arch_{suffix}_log.png', log=True)

        suffix="cifar10_wrn28_1"
        log_dir = f"demo_{suffix}"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        attack_list_and_paths = [
                        (f"lira 1", f"report_lira_offline_1_ref_model"),
                        (f"lira 2", f"report_lira_online_2_ref_model"),
                        (f"relative 1", f"report_relative_offline_1_ref_model"),
                        (f"relative 2", f"report_relative_online_2_ref_model"),
                        (f"reference 1", f"report_reference_1_ref_model"),
                        ("population", f"report_population"),
                    ]
        folder = "../../paper/data/multi"
        path = suffix
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/multi_arch_{suffix}_log.png', log=True)

        log_dir = f"demo_cifar10"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        attack_list_and_paths = [
                        (f"lira 1", f"report_lira_offline_1_ref_model"),
                        (f"lira 2", f"report_lira_online_2_ref_model"),
                        (f"relative 1", f"report_relative_offline_1_ref_model"),
                        (f"relative 2", f"report_relative_online_2_ref_model"),
                        (f"reference 1", f"report_reference_1_ref_model"),
                        ("population", f"report_population"),
                    ]
        folder = "../../paper/data/multi"
        path = "cifar10_wrn28_2"
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/multi_arch_{suffix}_log.png', log=True)

        suffix="cifar10_wrn28_10"
        log_dir = f"demo_{suffix}"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        attack_list_and_paths = [
                        (f"lira 1", f"report_lira_offline_1_ref_model"),
                        (f"lira 2", f"report_lira_online_2_ref_model"),
                        (f"relative 1", f"report_relative_offline_1_ref_model"),
                        (f"relative 2", f"report_relative_online_2_ref_model"),
                        (f"reference 1", f"report_reference_1_ref_model"),
                        ("population", f"report_population"),
                    ]
        folder = "../../paper/data/multi"
        path = suffix
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/multi_arch_{suffix}_log.png', log=True)
    
    elif args.figure == 202: # ROCs on multiple architecture (wrn28-2 target and different arch)
        log_dir = "demo_cifar10"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        prefix = "cnn16"
        attack_list_and_paths = [
                        (f"lira 1", f"{prefix}_report_lira_offline_1_ref_model"),
                        (f"lira 2", f"{prefix}_report_lira_online_2_ref_model"),
                        (f"relative 1", f"{prefix}_report_relative_offline_1_ref_model"),
                        (f"relative 2", f"{prefix}_report_relative_online_2_ref_model"),
                        (f"reference 1", f"{prefix}_report_reference_1_ref_model"),
                        ("population", f"{prefix}_report_population"),
                    ]
        folder = "../../paper/data/multi"
        path = f"wrn28-2_{prefix}"
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/multi_arch_same_target_{prefix}_log.png', log=True)


        log_dir = "demo_cifar10"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        prefix = "cnn32"
        attack_list_and_paths = [
                        (f"lira 1", f"{prefix}_report_lira_offline_1_ref_model"),
                        (f"lira 2", f"{prefix}_report_lira_online_2_ref_model"),
                        (f"relative 1", f"{prefix}_report_relative_offline_1_ref_model"),
                        (f"relative 2", f"{prefix}_report_relative_online_2_ref_model"),
                        (f"reference 1", f"{prefix}_report_reference_1_ref_model"),
                        ("population", f"{prefix}_report_population"),
                    ]
        folder = "../../paper/data/multi"
        path = f"wrn28-2_{prefix}"
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/multi_arch_same_target_{prefix}_log.png', log=True)

        log_dir = "demo_cifar10"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        prefix = "cnn64"
        attack_list_and_paths = [
                        (f"lira 1", f"{prefix}_report_lira_offline_1_ref_model"),
                        (f"lira 2", f"{prefix}_report_lira_online_2_ref_model"),
                        (f"relative 1", f"{prefix}_report_relative_offline_1_ref_model"),
                        (f"relative 2", f"{prefix}_report_relative_online_2_ref_model"),
                        (f"reference 1", f"{prefix}_report_reference_1_ref_model"),
                        ("population", f"{prefix}_report_population"),
                    ]
        folder = "../../paper/data/multi"
        path = f"wrn28-2_{prefix}"
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/multi_arch_same_target_{prefix}_log.png', log=True)

        log_dir = "demo_cifar10"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        prefix = "wrn28-1"
        attack_list_and_paths = [
                        (f"lira 1", f"{prefix}_report_lira_offline_1_ref_model"),
                        (f"lira 2", f"{prefix}_report_lira_online_2_ref_model"),
                        (f"relative 1", f"{prefix}_report_relative_offline_1_ref_model"),
                        (f"relative 2", f"{prefix}_report_relative_online_2_ref_model"),
                        (f"reference 1", f"{prefix}_report_reference_1_ref_model"),
                        ("population", f"{prefix}_report_population"),
                    ]
        folder = "../../paper/data/multi"
        path = f"wrn28-2_{prefix}"
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/multi_arch_same_target_{prefix}_log.png', log=True)

        log_dir = "demo_cifar10"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        prefix = "wrn28-2"
        attack_list_and_paths = [
                        (f"lira 1", f"report_lira_offline_1_ref_model"),
                        (f"lira 2", f"report_lira_online_2_ref_model"),
                        (f"relative 1", f"report_relative_offline_1_ref_model"),
                        (f"relative 2", f"report_relative_online_2_ref_model"),
                        (f"reference 1", f"report_reference_1_ref_model"),
                        ("population", f"report_population"),
                    ]
        folder = "../../paper/data/multi"
        path = f"wrn28-2_{prefix}"
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/multi_arch_same_target_{prefix}_log.png', log=True)

        log_dir = "demo_cifar10"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        prefix = "wrn28-10"
        attack_list_and_paths = [
                        (f"lira 1", f"{prefix}_report_lira_offline_1_ref_model"),
                        (f"lira 2", f"{prefix}_report_lira_online_2_ref_model"),
                        (f"relative 1", f"{prefix}_report_relative_offline_1_ref_model"),
                        (f"relative 2", f"{prefix}_report_relative_online_2_ref_model"),
                        (f"reference 1", f"{prefix}_report_reference_1_ref_model"),
                        ("population", f"{prefix}_report_population"),
                    ]
        folder = "../../paper/data/multi"
        path = f"wrn28-2_{prefix}"
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/multi_arch_same_target_{prefix}_log.png', log=True)

    elif args.figure == 203: # DP-SGD experiments
        
        multiplier = 0.0
        c = 10
        suffix=f"cifar10_cnn32_dp_noise_{multiplier}_c_{c}"
        log_dir = f"demo_{suffix}"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        attack_list_and_paths = [
                        (f"lira 2", f"report_lira_offline_2_ref_model"),
                        (f"lira 4", f"report_lira_online_4_ref_model"),
                        (f"relative 2", f"report_relative_offline_2_ref_model"),
                        (f"relative 4", f"report_relative_online_4_ref_model"),
                        (f"reference 2", f"report_reference_2_ref_model"),
                        ("population", f"report_population"),
                    ]
        folder = "../../paper/data/dp"
        path = suffix
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/dp_sgd_{suffix}_log.png', log=True)


        multiplier = 0.2
        c = 5
        suffix=f"cifar10_cnn32_dp_noise_{multiplier}_c_{c}"
        log_dir = f"demo_{suffix}"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        attack_list_and_paths = [
                        (f"lira 2", f"report_lira_offline_2_ref_model"),
                        (f"lira 4", f"report_lira_online_4_ref_model"),
                        (f"relative 2", f"report_relative_offline_2_ref_model"),
                        (f"relative 4", f"report_relative_online_4_ref_model"),
                        (f"reference 2", f"report_reference_2_ref_model"),
                        ("population", f"report_population"),
                    ]
        folder = "../../paper/data/dp"
        path = suffix
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/dp_sgd_{suffix}_log.png', log=True)


        multiplier = 0.8
        c = 1
        suffix=f"cifar10_cnn32_dp_noise_{multiplier}_c_{c}"
        log_dir = f"demo_{suffix}"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        attack_list_and_paths = [
                        (f"lira 2", f"report_lira_offline_2_ref_model"),
                        (f"lira 4", f"report_lira_online_4_ref_model"),
                        (f"relative 2", f"report_relative_offline_2_ref_model"),
                        (f"relative 4", f"report_relative_online_4_ref_model"),
                        (f"reference 2", f"report_reference_2_ref_model"),
                        ("population", f"report_population"),
                    ]
        folder = "../../paper/data/dp"
        path = suffix
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/dp_sgd_{suffix}_log.png', log=True)
    
    elif args.figure == 204: # Gradient Boosted Decision Trees
        suffix=f"purchase100_gbdt_d_3"
        log_dir = f"demo_{suffix}"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        attack_list_and_paths = [
                        (f"lira 1", f"report_lira_offline_1_ref_model"),
                        (f"lira 2", f"report_lira_online_2_ref_model"),
                        (f"relative 1", f"report_relative_offline_1_ref_model"),
                        (f"relative 2", f"report_relative_online_2_ref_model"),
                        (f"reference 1", f"report_reference_1_ref_model"),
                        ("population", f"report_population"),
                    ]
        folder = "../../paper/data/multi"
        path = suffix
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/{suffix}_log.png', log=True)



        suffix=f"purchase100_gbdt_d_5"
        log_dir = f"demo_{suffix}"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        attack_list_and_paths = [
                        (f"lira 1", f"report_lira_offline_1_ref_model"),
                        (f"lira 2", f"report_lira_online_2_ref_model"),
                        (f"relative 1", f"report_relative_offline_1_ref_model"),
                        (f"relative 2", f"report_relative_online_2_ref_model"),
                        (f"reference 1", f"report_reference_1_ref_model"),
                        ("population", f"report_population"),
                    ]
        folder = "../../paper/data/multi"
        path = suffix
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/{suffix}_log.png', log=True)



        suffix=f"purchase100_gbdt_d_7"
        log_dir = f"demo_{suffix}"
        model_idx = 0 # 12# "all" # "all" # 0 # "all"
        attacks_stats = {}
        attack_list_and_paths = [
                        (f"lira 1", f"report_lira_offline_1_ref_model"),
                        (f"lira 2", f"report_lira_online_2_ref_model"),
                        (f"relative 1", f"report_relative_offline_1_ref_model"),
                        (f"relative 2", f"report_relative_online_2_ref_model"),
                        (f"reference 1", f"report_reference_1_ref_model"),
                        ("population", f"report_population"),
                    ]
        folder = "../../paper/data/multi"
        path = suffix
        get_csv_rocs_from(model_idx, attack_list_and_paths, log_dir, path, folder)
        get_rocs_from(model_idx, attack_list_and_paths, log_dir, f'{log_dir}/{suffix}_log.png', log=True)

    else:
        raise Exception("Figure number is not implemented or incorrect")