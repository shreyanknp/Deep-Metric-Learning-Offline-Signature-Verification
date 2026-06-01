from typing import Tuple
import numpy as np
from sklearn import metrics


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    # Compute ROC and EER (threshold where FPR == 1 - TPR)
    fpr, tpr, thresh = metrics.roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    # EER is point where FPR ~= FNR
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    return eer, thresh[eer_idx]


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = None):
    auc = metrics.roc_auc_score(y_true, scores)
    eer, eer_thresh = compute_eer(y_true, scores)
    if threshold is None:
        threshold = eer_thresh
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {
        "auc": float(auc),
        "eer": float(eer),
        "eer_threshold": float(eer_thresh),
        "far": float(far),
        "frr": float(frr),
        "accuracy": float(acc),
        "used_threshold": float(threshold)
    }
