import csv
import json
import logging
import os
from dataclasses import asdict, is_dataclass

import numpy as np
import pandas as pd
import torch

from oste_screening.config import RunConfig

logger = logging.getLogger(__name__)


def setup_dirs(cfg: RunConfig):
    os.makedirs(cfg.DIR_LOGS, exist_ok=True)
    os.makedirs(cfg.DIR_CKPTS, exist_ok=True)
    os.makedirs(cfg.DIR_SUMS, exist_ok=True)

    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
    fh = logging.FileHandler(os.path.join(cfg.DIR_LOGS, "train.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)


def append_history_row(filepath, row_dict):
    exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not exists:
            w.writeheader()
        w.writerow(row_dict)


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, best_threshold, cfg: RunConfig, is_best=False, is_latest=True, tag=""):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "best_threshold": best_threshold,
    }
    if is_latest:
        torch.save(state, os.path.join(cfg.DIR_CKPTS, "latest.pth"))
    if is_best:
        torch.save(state, os.path.join(cfg.DIR_CKPTS, "best.pth"))
    if tag:
        torch.save(state, os.path.join(cfg.DIR_CKPTS, f"{tag}.pth"))


def load_checkpoint(path, model, cfg: RunConfig, optimizer=None, scheduler=None):
    logger.info(f"Loading checkpoint: {path}")
    state = torch.load(path, map_location=cfg.device)
    model.load_state_dict(state["model"])
    if optimizer and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    epoch = state.get("epoch", -1)
    best_metric = state.get("best_metric", -float("inf"))
    best_threshold = float(state.get("best_threshold", cfg.default_screen_threshold))
    logger.info(f"  → Epoch {epoch}, best AUPRC {best_metric:.4f}, best threshold {best_threshold:.3f}")
    return epoch, best_metric, best_threshold


def save_split_summary(cfg: RunConfig, df_all, train_df, val_df, test_df):
    def _lbl_dist(df):
        if "binary_label" in df.columns:
            return df["binary_label"].value_counts().to_dict()
        return {}

    lines = [
        "=== Split Summary (Binary Screening) ===",
        f"Total: {len(df_all)}  |  Train: {len(train_df)}  |  "
        f"Val: {len(val_df)}  |  Test: {len(test_df) if test_df is not None else 0}",
        f"Train label dist: {_lbl_dist(train_df)}",
        f"Val   label dist: {_lbl_dist(val_df)}",
    ]
    with open(os.path.join(cfg.DIR_SUMS, "split_summary.txt"), "w") as f:
        f.write("\n".join(lines))
    for line in lines:
        logger.info(line)


def _config_to_jsonable(obj):
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_config_to_jsonable(x) for x in obj]
    return str(obj)


def save_config_snapshot(cfg: RunConfig):
    if is_dataclass(cfg):
        raw = asdict(cfg)
    else:
        raw = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}
    d = {}
    for k, v in raw.items():
        try:
            d[k] = _config_to_jsonable(v)
        except Exception:
            d[k] = str(v)
    path = os.path.join(cfg.DIR_SUMS, "config_snapshot.json")
    with open(path, "w") as f:
        json.dump(d, f, indent=4)


def export_final_metrics(cfg: RunConfig, metrics):
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            clean[k] = float(v)
        else:
            clean[k] = v
    path = os.path.join(cfg.DIR_SUMS, "final_test_metrics.json")
    with open(path, "w") as f:
        json.dump(clean, f, indent=4)
    logger.info(f"Final metrics saved → {path}")
