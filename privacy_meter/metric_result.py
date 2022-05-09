from typing import List

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve


class MetricResult:
    """
    Contains results related to the performance of the metric.
    """

    def __init__(
            self,
            metric_id: str,
            predicted_labels: list,
            true_labels: list,
            predictions_proba: List[float] = None,
            signal_values=None,
            threshold: float = None
    ):
        """
        Constructor.
        Computes and stores the accuracy, ROC AUC score, and the confusion matrix for a metric.

        Args:
            metric_id: ID of the metric that was used (c.f. the report_files/explanations.json file)
            predicted_labels: Membership predictions of the metric
            true_labels: True membership labels used to evaluate the metric
            predictions_proba: Continuous version of the predicted_labels
            signal_values: Values of the signal used by the metric
            threshold: Threshold computed by the metric
        """
        self.metric_id = metric_id
        self.predicted_labels = predicted_labels
        self.true_labels = true_labels
        self.predictions_proba = predictions_proba
        self.signal_values = signal_values
        self.threshold = threshold

        self.accuracy = accuracy_score(y_true=true_labels, y_pred=predicted_labels)

        if self.predictions_proba is None:
            self.roc = roc_curve(y_true=true_labels, y_score=predicted_labels)
        else:
            self.roc = roc_curve(y_true=true_labels, y_score=predictions_proba)

        if self.predictions_proba is None:
            self.roc_auc = roc_auc_score(y_true=true_labels, y_score=predicted_labels)
        else:
            self.roc_auc = roc_auc_score(y_true=true_labels, y_score=predictions_proba)

        self.tn, self.fp, self.fn, self.tp = confusion_matrix(y_true=true_labels, y_pred=predicted_labels).ravel()

    def __str__(self):
        """
        Return a string describing the metric result
        """
        txt = [
            f'{" METRIC RESULT OBJECT ":=^48}',
            f'Accuracy          = {self.accuracy}',
            f'ROC AUC Score     = {self.roc_auc}',
            f'FPR               = {self.fp / (self.fp + self.tn)}',
            f'TN, FP, FN, TP    = {self.tn, self.fp, self.fn, self.tp}'
        ]
        return '\n'.join(txt)
