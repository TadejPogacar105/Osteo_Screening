import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

from oste_screening.config import RunConfig


@torch.no_grad()
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    m = {}

    if len(np.unique(y_true)) > 1:
        m["auprc"] = average_precision_score(y_true, y_prob)
        m["auroc"] = roc_auc_score(y_true, y_prob)
    else:
        m["auprc"] = 0.0
        m["auroc"] = 0.0

    m["accuracy"] = accuracy_score(y_true, y_pred)
    m["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m["sensitivity"] = tp / (tp + fn + 1e-8)
    m["specificity"] = tn / (tn + fp + 1e-8)

    return m


def find_best_screening_threshold(y_true, y_prob, cfg: RunConfig):
    if len(y_true) == 0:
        return cfg.default_screen_threshold, compute_metrics(y_true, y_prob, cfg.default_screen_threshold)

    if not cfg.use_threshold_search:
        t = cfg.default_screen_threshold
        return t, compute_metrics(y_true, y_prob, t)

    thresholds = np.arange(cfg.threshold_min, cfg.threshold_max + 1e-12, cfg.threshold_step)
    candidates = []
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        candidates.append((float(t), m))

    mode = str(cfg.threshold_search_mode).lower()
    if mode == "constraint_sens_max_spec":
        valid = [(t, m) for t, m in candidates if m["sensitivity"] >= cfg.target_sensitivity]
        if valid:
            best_t, best_m = max(
                valid,
                key=lambda x: (x[1]["specificity"], x[1]["balanced_acc"], x[1]["sensitivity"], -x[0]),
            )
            return best_t, best_m

    best_t, best_m = max(
        candidates,
        key=lambda x: (x[1]["balanced_acc"], x[1]["specificity"], x[1]["sensitivity"], -x[0]),
    )
    return best_t, best_m


@torch.no_grad()
def compute_tscore_metrics(y_true, y_pred):
    out = {
        "tscore_mae": 0.0,
        "tscore_rmse": 0.0,
        "tscore_pearson": 0.0,
        "tscore_count": 0.0,
    }
    if len(y_true) == 0:
        return out

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    out["tscore_count"] = float(len(y_true))
    out["tscore_mae"] = float(np.mean(np.abs(y_pred - y_true)))
    out["tscore_rmse"] = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    if len(y_true) > 1 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        out["tscore_pearson"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    return out
