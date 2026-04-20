import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from oste_screening.config import RunConfig
from oste_screening.data.dataset import ScreeningDataset, build_loader
from oste_screening.io_utils import append_history_row, load_checkpoint, save_checkpoint
from oste_screening.metrics import compute_metrics, compute_tscore_metrics, find_best_screening_threshold

logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, criterion, tscore_criterion, epoch, cfg: RunConfig):
    model.train()
    running_loss = 0.0
    running_loss_screen = 0.0
    running_loss_severe = 0.0
    running_loss_tscore = 0.0
    running_loss_fp = 0.0

    severe_criterion = nn.BCEWithLogitsLoss(reduction="none")
    tscore_enabled = bool(cfg.use_tscore_branch) and (epoch >= int(cfg.tscore_warmup_epochs))
    fp_enabled = bool(cfg.use_fp_suppression) and (epoch >= int(cfg.fp_warmup_epochs))
    for batch in loader:
        images = batch["image"].to(cfg.device)
        labels = batch["label"].to(cfg.device).unsqueeze(1)
        label_cls = batch["label_cls"].to(cfg.device)
        baseline_info = batch["baseline_info"].to(cfg.device)
        t_score = batch["t_score"].to(cfg.device).unsqueeze(1)
        tscore_mask = batch["tscore_mask"].to(cfg.device).unsqueeze(1)

        logits_screen, logits_severe, logits_tscore = model(images, baseline_info=baseline_info)
        loss_screen = criterion(logits_screen, labels)

        loss_fp = torch.zeros((), device=cfg.device)
        if fp_enabled:
            probs_screen = torch.sigmoid(logits_screen)
            neg_mask = labels < 0.5
            if torch.any(neg_mask):
                p_neg = probs_screen[neg_mask]
                loss_fp = torch.relu(p_neg - float(cfg.fp_margin)).pow(2).mean()

        loss_severe = torch.zeros((), device=cfg.device)
        if bool(cfg.use_severe_head) and logits_severe is not None:
            severe_mask = (label_cls >= 1).float().unsqueeze(1)
            severe_target = (label_cls == 2).float().unsqueeze(1)
            severe_loss_all = severe_criterion(logits_severe, severe_target)
            mask_sum = severe_mask.sum()
            if mask_sum > 0:
                loss_severe = (severe_loss_all * severe_mask).sum() / (mask_sum + 1e-8)

        loss_tscore = torch.zeros((), device=cfg.device)
        if tscore_enabled and logits_tscore is not None:
            tscore_loss_all = tscore_criterion(logits_tscore, t_score)
            tmask_sum = tscore_mask.sum()
            if tmask_sum > 0:
                loss_tscore = (tscore_loss_all * tscore_mask).sum() / (tmask_sum + 1e-8)

        loss = loss_screen
        if bool(cfg.use_severe_head):
            loss = loss + cfg.lambda_severe * loss_severe
        if tscore_enabled:
            loss = loss + cfg.lambda_tscore * loss_tscore
        if fp_enabled:
            loss = loss + cfg.lambda_fp * loss_fp

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        running_loss += loss.item()
        running_loss_screen += loss_screen.item()
        running_loss_severe += loss_severe.item()
        running_loss_tscore += loss_tscore.item()
        running_loss_fp += loss_fp.item()
    n = len(loader) + 1e-8
    return (
        running_loss / n,
        running_loss_screen / n,
        running_loss_severe / n,
        running_loss_tscore / n,
        running_loss_fp / n,
        tscore_enabled,
        fp_enabled,
    )


@torch.no_grad()
def evaluate(model, loader, cfg: RunConfig, screen_threshold=0.5, severe_threshold=0.5):
    model.eval()
    all_labels, all_probs = [], []
    severe_labels, severe_probs = [], []
    tscore_true, tscore_pred = [], []
    for batch in loader:
        images = batch["image"].to(cfg.device)
        label_cls = batch["label_cls"].to(cfg.device)
        baseline_info = batch["baseline_info"].to(cfg.device)
        logits, logits_severe, logits_tscore = model(images, baseline_info=baseline_info)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(batch["label"].numpy())

        severe_mask = label_cls >= 1
        if logits_severe is not None and torch.any(severe_mask):
            severe_prob = torch.sigmoid(logits_severe[severe_mask]).cpu().numpy().flatten()
            severe_true = (label_cls[severe_mask] == 2).long().cpu().numpy().flatten()
            severe_probs.extend(severe_prob)
            severe_labels.extend(severe_true)

        if logits_tscore is not None:
            tmask = batch["tscore_mask"] > 0.5
            if torch.any(tmask):
                tscore_pred.extend(logits_tscore.detach().cpu().numpy().flatten()[tmask.numpy()])
                tscore_true.extend(batch["t_score"].numpy().flatten()[tmask.numpy()])

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    screen_m = compute_metrics(y_true, y_prob, threshold=screen_threshold)

    severe_y_true = np.array(severe_labels)
    severe_y_prob = np.array(severe_probs)
    if len(severe_y_true) > 0:
        severe_m_raw = compute_metrics(severe_y_true, severe_y_prob, threshold=severe_threshold)
    else:
        severe_m_raw = {
            "auprc": 0.0,
            "auroc": 0.0,
            "accuracy": 0.0,
            "balanced_acc": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
        }
    severe_m = {f"severe_{k}": v for k, v in severe_m_raw.items()}
    tscore_m = compute_tscore_metrics(tscore_true, tscore_pred)

    out = {}
    out.update(screen_m)
    out.update(severe_m)
    out.update(tscore_m)
    return out, y_true, y_prob


def run_training(model, train_df, val_df, img_dirs, pos_weight, cfg: RunConfig, resume_checkpoint=None, save_every_epoch=False):
    train_ds = ScreeningDataset(train_df, img_dirs, cfg, is_train=True)
    val_ds = ScreeningDataset(val_df, img_dirs, cfg, is_train=False)
    train_loader = build_loader(train_ds, cfg, is_train=True)
    val_loader = build_loader(val_ds, cfg, is_train=False)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(cfg.device))
    tscore_criterion = nn.SmoothL1Loss(reduction="none")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=cfg.patience)

    start_epoch = 0
    best_auprc = -1.0
    best_threshold = cfg.default_screen_threshold
    no_improve = 0

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        ep, bm, bt = load_checkpoint(resume_checkpoint, model, cfg, optimizer, scheduler)
        start_epoch = ep + 1
        best_auprc = bm
        best_threshold = bt
        logger.info(f"Resumed from epoch {start_epoch}, best AUPRC={best_auprc:.4f}")

    history_file = os.path.join(cfg.DIR_LOGS, "history.csv")

    for epoch in range(start_epoch, cfg.epochs):
        (
            train_loss,
            train_loss_screen,
            train_loss_severe,
            train_loss_tscore,
            train_loss_fp,
            tscore_enabled,
            fp_enabled,
        ) = train_one_epoch(model, train_loader, optimizer, criterion, tscore_criterion, epoch, cfg)
        val_metrics, val_y_true, val_y_prob = evaluate(
            model, val_loader, cfg, screen_threshold=cfg.default_screen_threshold, severe_threshold=0.5
        )
        val_best_threshold, val_screen_metrics = find_best_screening_threshold(val_y_true, val_y_prob, cfg)
        val_metrics.update(val_screen_metrics)
        val_metrics["best_threshold"] = val_best_threshold
        auprc = val_metrics["auprc"]

        improved = auprc > best_auprc
        if improved:
            best_auprc = auprc
            best_threshold = val_best_threshold
            no_improve = 0
        else:
            no_improve += 1

        scheduler.step(auprc)

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_metric=best_auprc,
            best_threshold=best_threshold,
            cfg=cfg,
            is_best=improved,
            is_latest=True,
            tag=str(epoch) if save_every_epoch else "",
        )

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.5f}",
            "train_loss_screen": f"{train_loss_screen:.5f}",
            "train_loss_severe": f"{train_loss_severe:.5f}",
            "tscore_enabled": int(tscore_enabled),
            "fp_enabled": int(fp_enabled),
            "train_tscore_loss": f"{train_loss_tscore:.5f}",
            "train_fp_loss": f"{train_loss_fp:.5f}",
            "val_tscore_mae": f"{val_metrics['tscore_mae']:.5f}",
            "val_tscore_rmse": f"{val_metrics['tscore_rmse']:.5f}",
            "val_best_threshold": f"{val_best_threshold:.3f}",
            "val_balanced_acc": f"{val_metrics['balanced_acc']:.4f}",
            "val_sensitivity": f"{val_metrics['sensitivity']:.4f}",
            "val_specificity": f"{val_metrics['specificity']:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        }
        for k, v in val_metrics.items():
            row[f"val_{k}"] = f"{v:.4f}"
        append_history_row(history_file, row)

        flag = "⭐ BEST" if improved else ""
        logger.info(
            f"Epoch {epoch:3d} | loss {train_loss:.4f} | "
            f"L_screen {train_loss_screen:.4f} | "
            f"L_severe {train_loss_severe:.4f} | "
            f"L_tscore {train_loss_tscore:.4f} | "
            f"L_fp {train_loss_fp:.4f} | "
            f"T-on {int(tscore_enabled)} | "
            f"FP-on {int(fp_enabled)} | "
            f"Thr {val_best_threshold:.3f} | "
            f"AUPRC {auprc:.4f} | AUROC {val_metrics['auroc']:.4f} | "
            f"Acc {val_metrics['accuracy']:.4f} | "
            f"bAcc {val_metrics['balanced_acc']:.4f} | "
            f"Sens {val_metrics['sensitivity']:.4f} | "
            f"Spec {val_metrics['specificity']:.4f} | "
            f"T-MAE {val_metrics['tscore_mae']:.4f} | "
            f"T-RMSE {val_metrics['tscore_rmse']:.4f} | "
            f"S-AUPRC {val_metrics['severe_auprc']:.4f} | "
            f"S-AUROC {val_metrics['severe_auroc']:.4f} | "
            f"S-bAcc {val_metrics['severe_balanced_acc']:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}  {flag}"
        )

        if no_improve >= cfg.early_stop_patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch} (no AUPRC improvement for {cfg.early_stop_patience} epochs)."
            )
            break

    best_path = os.path.join(cfg.DIR_CKPTS, "best.pth")
    if os.path.exists(best_path):
        _, _, best_threshold = load_checkpoint(best_path, model, cfg)
    logger.info(f"Training complete. Best validation AUPRC = {best_auprc:.4f}, best threshold = {best_threshold:.3f}")
    return model, best_threshold
