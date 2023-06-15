"""This file contains ploting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_roc(fpr_list, tpr_list, roc_auc, path):
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


def plot_signal_histogram(in_signal, out_signal, train_data_idx, audit_data_idx, path):
    labels = np.concatenate(
        [np.ones(in_signal.shape[0]), np.zeros(out_signal.shape[0])]
    )
    sns.histplot(
        data=pd.DataFrame(
            {
                "Signal": np.concatenate([in_signal, out_signal]),
                "Membership": [
                    f"In ({train_data_idx})" if y == 1 else f"Out {train_data_idx}"
                    for y in labels
                ],
            }
        ),
        x="Signal",
        hue="Membership",
        element="step",
        kde=True,
    )
    plt.grid()
    plt.xlabel("Signal value")
    plt.ylabel("Number of Models")
    plt.title(f"Signal histogram for data point {audit_data_idx}")
    plt.savefig(path)
