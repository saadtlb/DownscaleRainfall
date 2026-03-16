import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


def compute_f1_by_threshold(y_true, y_proba, thresholds=None):
    """Compute F1 scores on a grid of decision thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.02)

    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_proba > threshold).astype(int)
        f1_scores.append(f1_score(y_true, y_pred))

    f1_scores = np.array(f1_scores)
    best_index = np.argmax(f1_scores)
    return thresholds, f1_scores, thresholds[best_index], f1_scores[best_index]


def find_best_f1_threshold(y_true, y_proba, thresholds=None):
    """Return the best threshold according to the F1 score."""
    _, _, best_threshold, best_score = compute_f1_by_threshold(y_true, y_proba, thresholds)
    return best_threshold, best_score


def compute_optimal_threshold_roc(y_true, y_proba):
    """Return the ROC curve and the Youden-optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_index = tpr - fpr
    best_index = np.argmax(youden_index)
    auc = roc_auc_score(y_true, y_proba)
    return fpr, tpr, thresholds, thresholds[best_index], auc


def apply_thresholds_per_station(results, thresholds_by_station):
    """Recompute accuracy and F1 from stored probabilities and custom thresholds."""
    updated = {}

    for station_id, values in results.items():
        threshold = thresholds_by_station.get(station_id, 0.5)
        y_true = values["y_true"]
        y_proba = values["y_proba"]
        y_pred = (y_proba > threshold).astype(int)

        updated[station_id] = {
            "accuracy": float((y_pred == y_true).mean()),
            "f1": float(f1_score(y_true, y_pred)),
            "y_true": y_true,
            "y_proba": y_proba,
            "y_pred": y_pred,
        }

    return updated
