import logging
import os

import numpy as np
import pandas as pd
import torch

from oste_screening.config import RunConfig

logger = logging.getLogger(__name__)


def compute_pos_weight(train_df):
    counts = train_df["binary_label"].value_counts()
    n_neg = counts.get(0, 1)
    n_pos = counts.get(1, 1)
    pw = max(1.0, n_neg / (n_pos + 1e-6))
    logger.info(f"Class balance  neg={n_neg}  pos={n_pos}  → pos_weight={pw:.2f}")
    return torch.tensor([pw], dtype=torch.float32)


def patient_level_split(df, cfg: RunConfig):
    if cfg.COL_PATIENT_ID not in df.columns:
        logger.warning("No patient_id column; falling back to random split.")
        idx = np.random.permutation(len(df))
        cut = int(len(df) * (1 - cfg.val_ratio))
        return df.iloc[idx[:cut]].copy(), df.iloc[idx[cut:]].copy()

    pids = df[cfg.COL_PATIENT_ID].unique()
    np.random.seed(cfg.random_seed)
    np.random.shuffle(pids)
    val_n = max(1, int(len(pids) * cfg.val_ratio))
    val_pids = set(pids[:val_n])
    train_mask = ~df[cfg.COL_PATIENT_ID].isin(val_pids)
    return (
        df[train_mask].reset_index(drop=True),
        df[~train_mask].reset_index(drop=True),
    )


def prepare_dataframe(cfg: RunConfig):
    if not os.path.exists(cfg.CSV_STAGE1) or not os.path.exists(cfg.CSV_STAGE2):
        raise FileNotFoundError("Stage-1 or Stage-2 splits.csv not found.")

    df1 = pd.read_csv(cfg.CSV_STAGE1)
    df2 = pd.read_csv(cfg.CSV_STAGE2)

    df1["path_lower"] = df1["path"].astype(str).str.replace("\\", "/").str.lower()
    df2["path_lower"] = df2["path"].astype(str).str.replace("\\", "/").str.lower()
    df = pd.merge(df1, df2, on="path_lower", how="outer")

    col_path_x = df.get("path_x", df.get("path"))
    df[cfg.COL_IMAGE_PATH] = col_path_x.fillna(df.get("path_y", pd.Series(dtype=object)))

    pid_x = df.get("patient_id_x", df.get("patient_id"))
    df[cfg.COL_PATIENT_ID] = pid_x.fillna(df.get("patient_id_y", pd.Series(dtype=object)))

    for col in [cfg.COL_TSCORE, cfg.COL_AGE, cfg.COL_SEX, cfg.COL_BMI]:
        cx = df.get(f"{col}_x", df.get(col, pd.Series(dtype=object)))
        cy = df.get(f"{col}_y", pd.Series(dtype=object))
        if isinstance(cx, pd.Series):
            df[col] = cx.fillna(cy)
        else:
            df[col] = cy

    missing = df[cfg.COL_PATIENT_ID].isna()
    if missing.any():
        df.loc[missing, cfg.COL_PATIENT_ID] = (
            df.loc[missing, cfg.COL_IMAGE_PATH].apply(lambda x: str(x).split("/")[-1].split("_")[0])
        )

    folder_to_label = {
        "normal": 0,
        "osteopenia": 1,
        "osteoporosis": 2,
    }

    def _label_from_parent_folder(rel_path):
        p = str(rel_path).replace("\\", "/").strip()
        parent = os.path.basename(os.path.dirname(p)).strip().lower()
        return folder_to_label.get(parent, -1)

    df[cfg.COL_LABEL] = df[cfg.COL_IMAGE_PATH].apply(_label_from_parent_folder).astype(int)
    invalid = df[cfg.COL_LABEL] < 0
    if invalid.any():
        bad_examples = df.loc[invalid, cfg.COL_IMAGE_PATH].astype(str).head(8).tolist()
        n_drop = int(invalid.sum())
        logger.warning(
            f"Dropping {n_drop} samples due to invalid parent folder labels. "
            f"Expected parent in {list(folder_to_label.keys())}. Examples: {bad_examples}"
        )
        df = df[~invalid].copy()

    df["binary_label"] = (df[cfg.COL_LABEL] >= 1).astype(int)

    split_x = df.get("split_x", df.get("split"))
    df["split"] = split_x.fillna(df.get("split_y", pd.Series(dtype=object))).fillna("train")

    if cfg.COL_TSCORE in df.columns:
        df[cfg.COL_TSCORE] = pd.to_numeric(df[cfg.COL_TSCORE], errors="coerce")

    return df
