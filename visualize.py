import numpy as np
from matplotlib import pyplot as plt
import textwrap


def plot_roc(fpr_list, tpr_list, roc_auc, path):
    """Function to get the ROC plot using FPR and TPR results

    Args:
        fpr_list (list or ndarray): List of FPR values
        tpr_list (list or ndarray): List of TPR values
        roc_auc (float or floating): Area Under the ROC Curve
        path (str): Folder for saving the ROC plot
    """
    range01 = np.linspace(0, 1)
    plt.fill_between(fpr_list, tpr_list, alpha=0.15)
    plt.plot(fpr_list, tpr_list, label="ROC curve")
    plt.plot(range01, range01, "--", label="Random guess")
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


def plot_roc_log(fpr_list, tpr_list, roc_auc, path):
    """Function to get the log-scale ROC plot using FPR and TPR results

    Args:
        fpr_list (list or ndarray): List of False Positive Rate values
        tpr_list (list or ndarray): List of True Positive Rate values
        roc_auc (float or floating): Area Under the ROC Curve
        path (str): Folder for saving the ROC plot
    """
    range01 = np.linspace(0, 1)
    plt.fill_between(fpr_list, tpr_list, alpha=0.15)
    plt.plot(fpr_list, tpr_list, label="ROC curve")
    plt.plot(range01, range01, "--", label="Random guess")
    plt.xlim([10e-6, 1])
    plt.ylim([10e-6, 1])
    plt.xscale("log")
    plt.yscale("log")
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





def plot_eps_vs_num_guesses(eps_list, correct_num_list, k_neg_list, k_pos_list, total_num, path):
    """Function to get the auditing performance versus number of guesses plot

    Args:
        eps_list (list or ndarray): List of audited eps values
        correct_num_list (list or ndarray): List of number of correct guesses
        k_neg_list (list or ndarray): List of positive guesses
        k_pos_list (list or ndarray): List of negative guesses
        total_num (int): Total number of samples
        path (str): Folder for saving the auditing performance plot
    """
    fig, ax = plt.subplots(1, 1)
    num_guesses_grid = np.array(k_neg_list) + total_num - np.array(k_pos_list)
    ax.scatter(num_guesses_grid, correct_num_list/num_guesses_grid,
        color = '#FF9999', alpha=0.6, label=r'Inference Accuracy', s = 80)
    ax.scatter(num_guesses_grid, eps_list,
        color = '#66B2FF', alpha=0.6, label=r'$EPS LB$', s = 80)
    ax.set_xlabel(r'number of guesses')
    plt.legend(fontsize=10)

    min_interval_idx = np.argmax(eps_list)
    t = f"k_neg={k_neg_list[min_interval_idx]} and k_pos={k_pos_list[min_interval_idx]} enables the highest audited EPS LB: num of guesses is {num_guesses_grid[min_interval_idx]}, EPS LB is {eps_list[min_interval_idx]}"
    tt = textwrap.fill(t, width = 70)
    plt.text(num_guesses_grid.mean(), -0.2, tt, ha='center', va='top')
    
   
    plt.savefig(path, bbox_inches = 'tight')
    
    plt.close()