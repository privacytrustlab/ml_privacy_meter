from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve


class MetricResult:
    """
    Contains results related to the performance of the metric.
    """
    def __init__(self, metric_id, predicted_labels, true_labels, predictions_proba, signal_values, threshold=None):
        """
        Constructor.

        Computes and stores the accuracy, ROC AUC score, and the confusion matrix for a metric.

        Args:
            predicted_labels: Membership predictions of the metric
            true_labels: True membership labels used to evaluate the metric
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
