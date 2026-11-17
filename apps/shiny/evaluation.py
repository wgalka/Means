"""Binary and multiclass classification evaluation of row aggregation scores."""
import re

import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


CRITERIA = {'youden': 'Maximize Youden J (equivalent to maximizing balanced accuracy)',
            'accuracy': 'Maximize accuracy', 'f1': 'Maximize positive class F1'}

# Documentation pages also expose a [source] link to the implementation.
METRIC_DOCS = {
    'ROC AUC': ('roc_auc_score', 'ROC AUC computed from continuous scores'),
    'ROC AUC (OvR macro)': ('roc_auc_score', "One-vs-Rest multiclass ROC AUC (multi_class='ovr', average='macro') from 1D tree probabilities"),
    'ROC AUC (OvO macro)': ('roc_auc_score', "One-vs-One multiclass ROC AUC (multi_class='ovo', average='macro') from 1D tree probabilities"),
    'Average precision (AP)': ('average_precision_score', 'Average precision computed from continuous scores'),
    'ACC': ('accuracy_score', 'Proportion of correct predictions'),
    'Balanced ACC': ('balanced_accuracy_score', 'Mean recall across all classes'),
    'Youden J': ('balanced_accuracy_score', 'Computed as TPR − FPR = 2 × balanced accuracy − 1. Equivalent to adjusted=True for two classes.'),
    'Precision (+)': ('precision_score', 'Precision for the positive class (pos_label=1)'),
    'Recall / sensitivity (+)': ('recall_score', 'Recall for the positive class (pos_label=1)'),
    'Specificity (−)': ('recall_score', 'TN/(TN+FP), equivalent to recall for the negative class (pos_label=0)'),
    'NPV (−)': ('precision_score', 'TN/(TN+FN), equivalent to precision for the negative class (pos_label=0)'),
    'F1 (+)': ('f1_score', "average='binary', pos_label=1"),
    'Precision micro': ('precision_score', "average='micro'"),
    'Precision macro': ('precision_score', "average='macro'"),
    'Precision weighted': ('precision_score', "average='weighted'"),
    'Recall micro': ('recall_score', "average='micro'"),
    'Recall macro': ('recall_score', "average='macro'"),
    'Recall weighted': ('recall_score', "average='weighted'"),
    'F1 micro': ('f1_score', "average='micro'"),
    'F1 macro': ('f1_score', "average='macro'"),
    'F1 weighted': ('f1_score', "average='weighted'"),
    'MCC': ('matthews_corrcoef', 'Matthews correlation coefficient'),
    'Cohen kappa': ('cohen_kappa_score', 'Cohen kappa'),
}


def text_decisions(text):
    return [item for item in re.split(r'[\s,;]+', text.strip()) if item]


def _sort_class_labels(unique_labels):
    try:
        return sorted(unique_labels, key=lambda item: (0, float(item), item))
    except ValueError:
        return sorted(unique_labels, key=lambda item: (1, item))


def _evaluate_multiclass(labels, scores):
    classes = _sort_class_labels(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    y = np.array([label_to_index[label] for label in labels], dtype=int)
    tree = DecisionTreeClassifier(max_leaf_nodes=len(classes), random_state=0)
    tree.fit(scores.reshape(-1, 1), y)
    predicted = tree.predict(scores.reshape(-1, 1))
    proba = tree.predict_proba(scores.reshape(-1, 1))
    thresholds = sorted(float(t) for t in tree.tree_.threshold if t != -2.0)
    class_indices = list(range(len(classes)))
    matrix = metrics.confusion_matrix(y, predicted, labels=class_indices)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, np.where(row_sums == 0, 1, row_sums), dtype=float)
    values = {
        'ROC AUC (OvR macro)': metrics.roc_auc_score(y, proba, multi_class='ovr', average='macro'),
        'ROC AUC (OvO macro)': metrics.roc_auc_score(y, proba, multi_class='ovo', average='macro'),
        'ACC': metrics.accuracy_score(y, predicted),
        'Balanced ACC': metrics.balanced_accuracy_score(y, predicted),
        'Precision micro': metrics.precision_score(y, predicted, average='micro', zero_division=0),
        'Precision macro': metrics.precision_score(y, predicted, average='macro', zero_division=0),
        'Precision weighted': metrics.precision_score(y, predicted, average='weighted', zero_division=0),
        'Recall micro': metrics.recall_score(y, predicted, average='micro', zero_division=0),
        'Recall macro': metrics.recall_score(y, predicted, average='macro', zero_division=0),
        'Recall weighted': metrics.recall_score(y, predicted, average='weighted', zero_division=0),
        'F1 micro': metrics.f1_score(y, predicted, average='micro', zero_division=0),
        'F1 macro': metrics.f1_score(y, predicted, average='macro', zero_division=0),
        'F1 weighted': metrics.f1_score(y, predicted, average='weighted', zero_division=0),
        'MCC': metrics.matthews_corrcoef(y, predicted),
        'Cohen kappa': metrics.cohen_kappa_score(y, predicted),
    }
    prediction_labels = np.array([classes[idx] for idx in predicted])
    threshold_summary = ', '.join(f'{t:.6g}' for t in thresholds) if thresholds else 'none'
    return {
        'mode': 'multiclass',
        'classes': classes,
        'y': y,
        'labels': labels,
        'scores': scores,
        'proba': proba,
        'predicted': predicted,
        'prediction_labels': prediction_labels,
        'thresholds': thresholds,
        'threshold': thresholds[0] if len(thresholds) == 1 else float('nan'),
        'optimal_threshold': thresholds[0] if len(thresholds) == 1 else float('nan'),
        'metrics': values,
        'matrix': matrix,
        'normalized': normalized,
        'rule': f'1D decision-tree splits ({threshold_summary})',
    }


def evaluate(decisions, scores, positive='1', direction='higher', criterion='youden', threshold=None):
    labels = np.asarray([str(value).strip() for value in decisions])
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores):
        raise ValueError(f'The number of labels ({len(labels)}) must match the number of scores ({len(scores)}), including any additional vectors.')
    unique_labels = set(labels)
    if any(value == '' for value in labels) or len(unique_labels) < 2:
        raise ValueError('Labels must contain at least two classes and no empty values.')
    if not np.isfinite(scores).all():
        raise ValueError('Scores must be finite numbers.')
    if len(unique_labels) > 2:
        return _evaluate_multiclass(labels, scores)
    if positive.strip() not in labels:
        raise ValueError('The positive class label does not appear in the ground truth labels.')
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
    return {'mode': 'binary', 'y': y, 'labels': labels, 'predicted': predicted,
            'prediction_labels': np.where(predicted == 1, positive.strip(), negative),
            'threshold': threshold, 'fpr': fpr, 'tpr': tpr, 'index': index,
            'optimal_threshold': optimal_threshold,
            'point': (fp / (fp + tn), tp / (tp + fn)),
            'metrics': values, 'matrix': matrix, 'normalized': normalized,
            'rule': f"score {'≥' if direction == 'higher' else '≤'} {threshold:.12g}",
            'positive': positive.strip(), 'negative': negative}


def evaluate_proba(decisions, class_names, scores_matrix, positive='1'):
    """Evaluate per-class aggregated scores/probabilities using argmax."""
    labels = np.asarray([str(value).strip() for value in decisions])
    class_names = [str(c).strip() for c in class_names]
    scores = np.asarray(scores_matrix, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(class_names) or len(class_names) < 2:
        raise ValueError('Provide aggregated scores for at least two classes.')
    if len(labels) != scores.shape[0]:
        raise ValueError(f'The number of labels ({len(labels)}) must match the number of rows ({scores.shape[0]}).')
    if any(value == '' for value in labels):
        raise ValueError('Ground truth labels cannot contain empty values.')
    if not np.isfinite(scores).all():
        raise ValueError('Aggregated class scores must be finite numbers.')
    unknown = sorted(set(labels) - set(class_names))
    if unknown:
        raise ValueError(f"Ground truth labels contain classes not in the uploaded class list: {', '.join(unknown)}.")

    clipped = np.maximum(scores, 0.0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    proba = np.where(row_sums > 0, clipped / np.where(row_sums > 0, row_sums, 1.0), 1.0 / len(class_names))
    pred_idx = np.argmax(scores, axis=1)
    prediction_labels = np.array([class_names[i] for i in pred_idx])

    if len(class_names) == 2:
        pos = positive.strip() if positive.strip() in class_names else class_names[1]
        pos_idx = class_names.index(pos)
        neg = class_names[1 - pos_idx]
        y = (labels == pos).astype(int)
        predicted = (pred_idx == pos_idx).astype(int)
        pos_proba = proba[:, pos_idx]
        fpr, tpr, _ = metrics.roc_curve(y, pos_proba, drop_intermediate=False)
        matrix = metrics.confusion_matrix(y, predicted, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        values = {
            'ROC AUC': metrics.roc_auc_score(y, pos_proba),
            'Average precision (AP)': metrics.average_precision_score(y, pos_proba),
            'ACC': metrics.accuracy_score(y, predicted),
            'Balanced ACC': metrics.balanced_accuracy_score(y, predicted),
            'Youden J': (tp / (tp + fn) if tp + fn else 0.) - (fp / (fp + tn) if fp + tn else 0.),
            'Precision (+)': metrics.precision_score(y, predicted, zero_division=0),
            'Recall / sensitivity (+)': metrics.recall_score(y, predicted, zero_division=0),
            'Specificity (−)': tn / (tn + fp) if tn + fp else 0.,
            'NPV (−)': tn / (tn + fn) if tn + fn else 0.,
            'F1 (+)': metrics.f1_score(y, predicted, zero_division=0),
            'F1 micro': metrics.f1_score(y, predicted, average='micro', zero_division=0),
            'F1 macro': metrics.f1_score(y, predicted, average='macro', zero_division=0),
            'F1 weighted': metrics.f1_score(y, predicted, average='weighted', zero_division=0),
            'MCC': metrics.matthews_corrcoef(y, predicted),
            'Cohen kappa': metrics.cohen_kappa_score(y, predicted),
        }
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, np.where(row_totals == 0, 1, row_totals), dtype=float)
        return {
            'mode': 'binary', 'fusion': True, 'classes': class_names,
            'y': y, 'labels': labels, 'proba': proba,
            'predicted': predicted, 'prediction_labels': prediction_labels,
            'threshold': float('nan'), 'optimal_threshold': float('nan'),
            'fpr': fpr, 'tpr': tpr,
            'point': (fp / (fp + tn) if fp + tn else 0., tp / (tp + fn) if tp + fn else 0.),
            'metrics': values, 'matrix': matrix, 'normalized': normalized,
            'rule': f'argmax across classes ({neg} vs {pos})',
            'positive': pos, 'negative': neg,
        }

    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y = np.array([label_to_idx[val] for val in labels], dtype=int)
    class_indices = list(range(len(class_names)))
    matrix = metrics.confusion_matrix(y, pred_idx, labels=class_indices)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, np.where(row_totals == 0, 1, row_totals), dtype=float)
    try:
        auc_ovr = metrics.roc_auc_score(y, proba, multi_class='ovr', average='macro', labels=class_indices)
        auc_ovo = metrics.roc_auc_score(y, proba, multi_class='ovo', average='macro', labels=class_indices)
    except ValueError:
        auc_ovr, auc_ovo = float('nan'), float('nan')
    values = {
        'ROC AUC (OvR macro)': auc_ovr,
        'ROC AUC (OvO macro)': auc_ovo,
        'ACC': metrics.accuracy_score(y, pred_idx),
        'Balanced ACC': metrics.balanced_accuracy_score(y, pred_idx),
        'Precision micro': metrics.precision_score(y, pred_idx, average='micro', zero_division=0),
        'Precision macro': metrics.precision_score(y, pred_idx, average='macro', zero_division=0),
        'Precision weighted': metrics.precision_score(y, pred_idx, average='weighted', zero_division=0),
        'Recall micro': metrics.recall_score(y, pred_idx, average='micro', zero_division=0),
        'Recall macro': metrics.recall_score(y, pred_idx, average='macro', zero_division=0),
        'Recall weighted': metrics.recall_score(y, pred_idx, average='weighted', zero_division=0),
        'F1 micro': metrics.f1_score(y, pred_idx, average='micro', zero_division=0),
        'F1 macro': metrics.f1_score(y, pred_idx, average='macro', zero_division=0),
        'F1 weighted': metrics.f1_score(y, pred_idx, average='weighted', zero_division=0),
        'MCC': metrics.matthews_corrcoef(y, pred_idx),
        'Cohen kappa': metrics.cohen_kappa_score(y, pred_idx),
    }
    return {
        'mode': 'multiclass', 'fusion': True, 'classes': class_names,
        'y': y, 'labels': labels, 'scores': proba.max(axis=1), 'proba': proba,
        'predicted': pred_idx, 'prediction_labels': prediction_labels,
        'thresholds': [], 'threshold': float('nan'), 'optimal_threshold': float('nan'),
        'metrics': values, 'matrix': matrix, 'normalized': normalized,
        'rule': 'argmax across per-class aggregated scores',
    }


