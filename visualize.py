import numpy as np
from matplotlib import pyplot as plt


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
