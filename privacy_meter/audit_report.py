from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from privacy_meter.metric_result import MetricResult


class AuditReport(ABC):

    @staticmethod
    @abstractmethod
    def generate_report(metric_result: MetricResult):
        pass


class ROCCurveReport(AuditReport):

    @staticmethod
    def generate_report(metric_result: MetricResult,
                        show: bool = False,
                        save: bool = True,
                        filename: str = 'roc_curve.jpg'
                        ):
        range01 = np.linspace(0, 1)
        fpr, tpr, thresholds = metric_result.roc
        plt.fill_between(fpr, tpr, alpha=0.15)
        plt.plot(fpr, tpr, label=metric_result.metric_name)
        plt.plot(range01, range01, '--', label='Random guess')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid()
        plt.legend()
        plt.xlabel('False positive rate (FPR)')
        plt.ylabel('True positive rate (TPR)')
        plt.title('ROC curve')
        plt.text(
            0.7, 0.3,
            f'AUC = {metric_result.roc_auc:.03f}',
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(facecolor='white', alpha=0.5)
        )
        if save:
            plt.savefig(fname=filename, dpi=1000)
        if show:
            plt.show()
        plt.clf()


class ConfusionMatrixReport(AuditReport):

    @staticmethod
    def generate_report(metric_result: MetricResult,
                        show: bool = False,
                        save: bool = True,
                        filename: str = 'confusion_matrix.jpg'
                        ):
        cm = np.array([[metric_result.tn, metric_result.fp], [metric_result.fn, metric_result.tp]])
        cm = 100 * cm / np.sum(cm)
        index = ["Non-member", "Member"]
        df_cm = pd.DataFrame(cm, index, index)
        sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
        plt.grid()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion matrix (in %)')
        if save:
            plt.savefig(fname=filename, dpi=1000)
        if show:
            plt.show()
        plt.clf()


class SignalHistogramReport(AuditReport):

    @staticmethod
    def generate_report(metric_result: MetricResult,
                        show: bool = False,
                        save: bool = True,
                        filename: str = 'signal_histogram.jpg'
                        ):
        member_signals = metric_result.signal_values[np.array(metric_result.true_labels)]
        non_member_signals = metric_result.signal_values[1 - np.array(metric_result.true_labels)]
        plt.hist(member_signals, label='Members', alpha=0.5)
        plt.hist(non_member_signals, label='Non-members', alpha=0.5)
        plt.legend()
        plt.xlabel('Signal value')
        plt.ylabel('Number of samples')
        plt.title('Signal histogram')
        if save:
            plt.savefig(fname=filename, dpi=1000)
        if show:
            plt.show()
        plt.clf()
