"""
Command-line entry for training / evaluation.
"""

from __future__ import annotations

import argparse
import logging
import os
import random

import numpy as np
import torch

from oste_screening.calibration import clinical_calibration
from oste_screening.config import RunConfig, cfg
from oste_screening.data.dataset import ScreeningDataset, build_loader
from oste_screening.engine import evaluate, run_training
from oste_screening.io_utils import (
    export_final_metrics,
    load_checkpoint,
    save_config_snapshot,
    save_split_summary,
    setup_dirs,
)
from oste_screening.model import BinaryScreeningModel
from oste_screening.pipeline import compute_pos_weight, patient_level_split, prepare_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def apply_runtime_overrides(args, run_cfg: RunConfig):
    overrides = {
        "OUTPUT_DIR": args.output_dir,
        "CSV_STAGE1": args.csv_stage1,
        "CSV_STAGE2": args.csv_stage2,
        "IMAGE_ROOT_STAGE1": args.image_root_stage1,
        "IMAGE_ROOT_STAGE2": args.image_root_stage2,
        "random_seed": args.random_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "early_stop_patience": args.early_stop_patience,
        "dropout": args.dropout,
        "neck_dim": args.neck_dim,
        "lambda_severe": args.lambda_severe,
        "lambda_tscore": args.lambda_tscore,
        "lambda_fp": args.lambda_fp,
        "fp_margin": args.fp_margin,
        "fp_warmup_epochs": args.fp_warmup_epochs,
        "tscore_warmup_epochs": args.tscore_warmup_epochs,
        "threshold_search_mode": args.threshold_search_mode,
        "target_sensitivity": args.target_sensitivity,
        "default_screen_threshold": args.default_screen_threshold,
        "experiment_id": args.experiment_id,
        "experiment_category": args.experiment_category,
        "device": torch.device(args.device) if args.device else None,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(run_cfg, k, v)

    for k in [
        "use_neck",
        "use_severe_head",
        "use_tscore_branch",
        "use_tarr",
        "use_threshold_search",
        "use_fp_suppression",
    ]:
        v = getattr(args, k)
        if v is not None:
            setattr(run_cfg, k, bool(v))

    if args.tarr_hidden_dim is not None:
        run_cfg.tarr_hidden_dim = int(args.tarr_hidden_dim)
    else:
        run_cfg.tarr_hidden_dim = int(run_cfg.neck_dim if bool(run_cfg.use_neck) else 2048)
    if args.tarr_adapter_dim is not None:
        run_cfg.tarr_adapter_dim = int(args.tarr_adapter_dim)

    run_cfg.refresh_output_dirs()


def build_argparser():
    p = argparse.ArgumentParser(description="Binary Screening: Normal vs Bone Loss")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--csv_stage1", type=str, default=None)
    p.add_argument("--csv_stage2", type=str, default=None)
    p.add_argument("--image_root_stage1", type=str, default=None)
    p.add_argument("--image_root_stage2", type=str, default=None)
    p.add_argument("--random_seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--neck_dim", type=int, default=None)
    p.add_argument("--lambda_severe", type=float, default=None)
    p.add_argument("--lambda_tscore", type=float, default=None)
    p.add_argument("--lambda_fp", type=float, default=None)
    p.add_argument("--fp_margin", type=float, default=None)
    p.add_argument("--fp_warmup_epochs", type=int, default=None)
    p.add_argument("--tscore_warmup_epochs", type=int, default=None)
    p.add_argument(
        "--threshold_search_mode",
        type=str,
        default=None,
        choices=["constraint_sens_max_spec", "max_balanced_acc"],
    )
    p.add_argument("--target_sensitivity", type=float, default=None)
    p.add_argument("--default_screen_threshold", type=float, default=None)
    p.add_argument("--tarr_hidden_dim", type=int, default=None)
    p.add_argument("--tarr_adapter_dim", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--experiment_id", type=str, default=None)
    p.add_argument("--experiment_category", type=str, default=None)
    p.add_argument("--use_neck", type=str2bool, default=None)
    p.add_argument("--use_severe_head", type=str2bool, default=None)
    p.add_argument("--use_tscore_branch", type=str2bool, default=None)
    p.add_argument("--use_tarr", type=str2bool, default=None)
    p.add_argument("--use_threshold_search", type=str2bool, default=None)
    p.add_argument("--use_fp_suppression", type=str2bool, default=None)
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    apply_runtime_overrides(args, cfg)

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_seed)

    setup_dirs(cfg)
    save_config_snapshot(cfg)

    df_main = prepare_dataframe(cfg)
    logger.info(f"Total samples after label cleaning: {len(df_main)}")

    train_df = df_main[df_main["split"].str.lower().isin(["train"])].copy().reset_index(drop=True)
    val_df = df_main[df_main["split"].str.lower().isin(["val", "valid"])].copy().reset_index(drop=True)
    test_df = df_main[df_main["split"].str.lower().isin(["test"])].copy().reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        logger.warning("Pre-defined split empty; falling back to patient-level split.")
        train_df, val_df = patient_level_split(df_main, cfg)

    save_split_summary(cfg, df_main, train_df, val_df, test_df)
    pos_weight = compute_pos_weight(train_df)

    img_dirs = [cfg.IMAGE_ROOT_STAGE1, cfg.IMAGE_ROOT_STAGE2]
    model = BinaryScreeningModel(cfg.to_model_config()).to(cfg.device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Experiment: id={cfg.experiment_id} | category={cfg.experiment_category}")
    logger.info(
        f"Switches: neck={int(cfg.use_neck)} severe={int(cfg.use_severe_head)} "
        f"tscore={int(cfg.use_tscore_branch)} tarr={int(cfg.use_tarr)} "
        f"th_search={int(cfg.use_threshold_search)} fp_sup={int(cfg.use_fp_suppression)}"
    )
    logger.info(
        f"TARR enabled: {int(cfg.use_tarr)} | tarr_hidden_dim={cfg.tarr_hidden_dim} | tarr_adapter_dim={cfg.tarr_adapter_dim}"
    )

    resume_path = args.resume
    if args.auto_resume and not resume_path:
        auto = os.path.join(cfg.DIR_CKPTS, "latest.pth")
        if os.path.exists(auto):
            resume_path = auto

    if args.eval_only:
        if not resume_path or not os.path.exists(resume_path):
            raise ValueError("--eval_only requires a valid checkpoint (use --resume or --auto_resume).")
        _, _, best_threshold = load_checkpoint(resume_path, model, cfg)

        for name, split_df in [("val", val_df), ("test", test_df)]:
            if split_df is None or len(split_df) == 0:
                continue
            ds = ScreeningDataset(split_df, img_dirs, cfg, is_train=False)
            loader = build_loader(ds, cfg, is_train=False)
            metrics, _, _ = evaluate(model, loader, cfg, screen_threshold=best_threshold, severe_threshold=0.5)
            logger.info(f"[EVAL] {name}: {metrics}")
        return

    model, best_threshold = run_training(
        model,
        train_df,
        val_df,
        img_dirs,
        pos_weight,
        cfg,
        resume_checkpoint=resume_path,
        save_every_epoch=args.save_every_epoch,
    )

    if args.calibrate:
        val_ds = ScreeningDataset(val_df, img_dirs, cfg, is_train=False)
        val_loader = build_loader(val_ds, cfg, is_train=False)
        _, _, val_probs = evaluate(model, val_loader, cfg, screen_threshold=best_threshold, severe_threshold=0.5)
        clinical_calibration(cfg, val_df, val_probs)

    if test_df is not None and len(test_df) > 0:
        logger.info("=== Final Test Evaluation ===")
        test_ds = ScreeningDataset(test_df, img_dirs, cfg, is_train=False)
        test_loader = build_loader(test_ds, cfg, is_train=False)
        test_metrics, _, _ = evaluate(model, test_loader, cfg, screen_threshold=best_threshold, severe_threshold=0.5)
        test_metrics["screen_threshold"] = best_threshold
        best_p = os.path.join(cfg.DIR_CKPTS, "best.pth")
        test_metrics["best_validation_metric"] = (
            float(load_checkpoint(best_p, model, cfg)[1]) if os.path.exists(best_p) else float("nan")
        )
        test_metrics["experiment_id"] = str(cfg.experiment_id)
        test_metrics["category"] = str(cfg.experiment_category)
        test_metrics["use_neck"] = int(bool(cfg.use_neck))
        test_metrics["use_severe_head"] = int(bool(cfg.use_severe_head))
        test_metrics["use_tscore_branch"] = int(bool(cfg.use_tscore_branch))
        test_metrics["use_tarr"] = int(bool(cfg.use_tarr))
        test_metrics["use_threshold_search"] = int(bool(cfg.use_threshold_search))
        test_metrics["use_fp_suppression"] = int(bool(cfg.use_fp_suppression))
        export_final_metrics(cfg, test_metrics)
        logger.info(
            f"Test AUPRC={test_metrics['auprc']:.4f}  AUROC={test_metrics['auroc']:.4f}  "
            f"Sens={test_metrics['sensitivity']:.4f}  Spec={test_metrics['specificity']:.4f}  "
            f"S-AUPRC={test_metrics['severe_auprc']:.4f}  S-AUROC={test_metrics['severe_auroc']:.4f}  "
            f"S-bAcc={test_metrics['severe_balanced_acc']:.4f}  Thr={best_threshold:.3f}"
        )
    else:
        logger.info("No test set available; skipping final evaluation.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
