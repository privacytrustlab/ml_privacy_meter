from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


class MetricResult:
    """
    Contains results related to the performance of the metric.
    """
    def __init__(self, predictions, true_labels):
        """
        Constructor.

        Computes and stores the accuracy, ROC AUC score, and the confusion matrix for a metric.

        Args:
            predictions: Membership predictions of the metric
            true_labels: True membership labels used to evaluate the metric
        """
        self.predictions = predictions
        self.true_labels = true_labels

        self.accuracy = accuracy_score(y_true=true_labels, y_pred=predictions)
        self.roc_auc = roc_auc_score(y_true=true_labels, y_score=predictions)
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(y_true=true_labels, y_pred=predictions).ravel()

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
