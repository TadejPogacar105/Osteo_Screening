import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from oste_screening.config import RunConfig

logger = logging.getLogger(__name__)


def clinical_calibration(cfg: RunConfig, val_df, val_probs):
    """
    Optional: fit logistic regression on val (image prob + age/sex/BMI).
    Does not change main model weights.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        logger.info("sklearn not available; skipping clinical calibration.")
        return None

    cols = [cfg.COL_AGE, cfg.COL_SEX, cfg.COL_BMI]
    available = [c for c in cols if c in val_df.columns]
    if not available:
        logger.info("No clinical columns found; skipping calibration.")
        return None

    X_parts = [val_probs.reshape(-1, 1)]
    for c in available:
        vals = pd.to_numeric(val_df[c], errors="coerce").fillna(0).values
        X_parts.append(vals.reshape(-1, 1))
    X = np.hstack(X_parts)
    y = val_df["binary_label"].values

    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X, y)
    cal_probs = clf.predict_proba(X)[:, 1]
    cal_auprc = average_precision_score(y, cal_probs)
    img_auprc = average_precision_score(y, val_probs)
    logger.info(f"[Calibration] Image-only AUPRC={img_auprc:.4f} → Calibrated AUPRC={cal_auprc:.4f}")
    return clf
