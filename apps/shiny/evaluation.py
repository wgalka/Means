"""Binary classification evaluation of row aggregation scores."""
import re

import numpy as np
from sklearn import metrics


CRITERIA = {'youden': 'Maximize Youden J (equivalent to maximizing balanced accuracy)',
            'accuracy': 'Maximize accuracy', 'f1': 'Maximize positive class F1'}

# Documentation pages also expose a [source] link to the implementation.
METRIC_DOCS = {
    'ROC AUC': ('roc_auc_score', 'ROC AUC computed from continuous scores'),
    'Average precision (AP)': ('average_precision_score', 'Average precision computed from continuous scores'),
    'ACC': ('accuracy_score', 'Proportion of correct predictions'),
    'Balanced ACC': ('balanced_accuracy_score', 'Mean recall across both classes'),
    'Youden J': ('balanced_accuracy_score', 'Computed as TPR − FPR = 2 × balanced accuracy − 1. Equivalent to adjusted=True for two classes.'),
    'Precision (+)': ('precision_score', 'Precision for the positive class (pos_label=1)'),
    'Recall / sensitivity (+)': ('recall_score', 'Recall for the positive class (pos_label=1)'),
    'Specificity (−)': ('recall_score', 'TN/(TN+FP), equivalent to recall for the negative class (pos_label=0)'),
    'NPV (−)': ('precision_score', 'TN/(TN+FN), equivalent to precision for the negative class (pos_label=0)'),
    'F1 (+)': ('f1_score', "average='binary', pos_label=1"),
    'F1 micro': ('f1_score', "average='micro'"),
    'F1 macro': ('f1_score', "average='macro'"),
    'F1 weighted': ('f1_score', "average='weighted'"),
    'MCC': ('matthews_corrcoef', 'Matthews correlation coefficient'),
    'Cohen kappa': ('cohen_kappa_score', 'Cohen kappa'),
}


def text_decisions(text):
    return [item for item in re.split(r'[\s,;]+', text.strip()) if item]


def evaluate(decisions, scores, positive='1', direction='higher', criterion='youden', threshold=None):
    labels = np.asarray([str(value).strip() for value in decisions])
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores):
        raise ValueError(f'The number of labels ({len(labels)}) must match the number of scores ({len(scores)}), including any additional vectors.')
    if any(value == '' for value in labels) or len(set(labels)) != 2:
        raise ValueError('Labels must contain exactly two classes and no empty values.')
    if positive.strip() not in labels:
        raise ValueError('The positive class label does not appear in the ground truth labels.')
    if not np.isfinite(scores).all():
        raise ValueError('Scores must be finite numbers.')
    if direction not in ('higher', 'lower') or criterion not in CRITERIA:
        raise ValueError('Unknown score direction or threshold criterion.')
    y = (labels == positive.strip()).astype(int)
    oriented = scores if direction == 'higher' else -scores
    fpr, tpr, thresholds = metrics.roc_curve(y, oriented, drop_intermediate=False)
    positives, negatives = y.sum(), len(y) - y.sum()
    tp, fp = np.rint(tpr * positives), np.rint(fpr * negatives)
    fn, tn = positives - tp, negatives - fp
    objectives = {'youden': tp * negatives - fp * positives, 'accuracy': tp + tn,
                  'f1': 2 * tp / np.maximum(2 * tp + fp + fn, 1)}
    # ROC thresholds descend: ties prefer the largest oriented threshold,
    # i.e. fewer positive predictions. Infinity represents all-negative.
    index = int(np.argmax(objectives[criterion]))
    cutoff = thresholds[index]
    optimal_threshold = float(cutoff if direction == 'higher' else -cutoff)
    if threshold is not None:
        threshold = float(threshold)
        if not np.isfinite(threshold):
            raise ValueError('The manual threshold must be a finite number.')
        cutoff = threshold if direction == 'higher' else -threshold
    predicted = (oriented >= cutoff).astype(int)
    threshold = float(cutoff if direction == 'higher' else -cutoff)
    matrix = metrics.confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    values = {
        'ROC AUC': metrics.roc_auc_score(y, oriented),
        'Average precision (AP)': metrics.average_precision_score(y, oriented),
        'ACC': metrics.accuracy_score(y, predicted),
        'Balanced ACC': metrics.balanced_accuracy_score(y, predicted),
        'Youden J': tp / (tp + fn) - fp / (fp + tn),
        'Precision (+)': metrics.precision_score(y, predicted, zero_division=0),
        'Recall / sensitivity (+)': metrics.recall_score(y, predicted, zero_division=0),
        'Specificity (−)': tn / (tn + fp),
        'NPV (−)': tn / (tn + fn) if tn + fn else 0.,
        'F1 (+)': metrics.f1_score(y, predicted, zero_division=0),
        'F1 micro': metrics.f1_score(y, predicted, average='micro', zero_division=0),
        'F1 macro': metrics.f1_score(y, predicted, average='macro', zero_division=0),
        'F1 weighted': metrics.f1_score(y, predicted, average='weighted', zero_division=0),
        'MCC': metrics.matthews_corrcoef(y, predicted),
        'Cohen kappa': metrics.cohen_kappa_score(y, predicted),
    }
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    negative = next(label for label in labels if label != positive.strip())
    return {'y': y, 'labels': labels, 'predicted': predicted,
            'prediction_labels': np.where(predicted == 1, positive.strip(), negative),
            'threshold': threshold, 'fpr': fpr, 'tpr': tpr, 'index': index,
            'optimal_threshold': optimal_threshold,
            'point': (fp / (fp + tn), tp / (tp + fn)),
            'metrics': values, 'matrix': matrix, 'normalized': normalized,
            'rule': f"score {'≥' if direction == 'higher' else '≤'} {threshold:.12g}",
            'positive': positive.strip(), 'negative': negative}


