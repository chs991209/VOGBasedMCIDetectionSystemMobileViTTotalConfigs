"""CLI — Swin SIFT-DBT hybrid pipeline entry point.

Example:
    # Default (no attention prior)
    python src/candidates/f_swin_sift_dbt/detection_caller/sift_dbt_caller.py

    # Warm-start attention gate with the legacy wvote scheme
    python src/candidates/f_swin_sift_dbt/detection_caller/sift_dbt_caller.py \\
        --attention-prior wvote

    # Ad-hoc manual weights (comma-separated 8 floats — one per task)
    python src/candidates/f_swin_sift_dbt/detection_caller/sift_dbt_caller.py \\
        --custom-prior "0.0,0.5,0.5,0.5,0.0,1.5,1.5,1.5"

    # Resume an interrupted run
    python src/candidates/f_swin_sift_dbt/detection_caller/sift_dbt_caller.py \\
        --resume outputs/checkpoints/run_20260703_123456_sift_dbt_swin_s1050_s2500_optA_prior_loo/state.pkl
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

_SRC_DIR = Path(__file__).resolve().parents[3]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paths import (  # noqa: E402
    CACHE_DIR, CHECKPOINTS_DIR, DATA_DIR, LOGS_DIR, ensure_output_dirs,
)
from full_experiments_using.data_processor.data_engineering import EventLockedCWTPipeline  # noqa: E402

from candidates.f_swin_sift_dbt.attention_priors import (  # noqa: E402
    ATTENTION_PRIOR_SCHEMES,
    resolve_prior,
)
from candidates.f_swin_sift_dbt.data_processor.datasets import (  # noqa: E402
    FlatWindowDataset, SubjectBundleDataset,
)
from candidates.f_swin_sift_dbt.hybrid_trainer import HybridTrainer  # noqa: E402


NUM_TASKS = 8


def _parse_tasks(s: str) -> tuple:
    ids = tuple(int(x) for x in s.split(",") if x.strip())
    if not ids or not all(0 <= i < 8 for i in ids):
        raise argparse.ArgumentTypeError(f"--tasks must be a subset of 0..7; got {ids}")
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError(f"--tasks must be unique; got {ids}")
    return ids


def _parse_custom_prior(s: str) -> list:
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) != NUM_TASKS:
        raise argparse.ArgumentTypeError(
            f"--custom-prior must have exactly {NUM_TASKS} comma-separated floats; got {len(parts)}"
        )
    try:
        return [float(x) for x in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--custom-prior contains a non-float value: {e}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Swin-Tiny + SIFT-DBT Hybrid Pipeline — Candidate F"
    )

    # ── Stage 1 knobs ──────────────────────────────────────────────────
    p.add_argument("--stage1-epochs", dest="stage1_epochs", type=int, default=50)
    p.add_argument("--stage1-patience", dest="stage1_patience", type=int, default=10)
    p.add_argument("--stage1-lr", dest="stage1_lr", type=float, default=5e-5)
    p.add_argument("--stage1-batch-size", dest="stage1_batch_size", type=int, default=32)

    # ── Stage 2 knobs ──────────────────────────────────────────────────
    p.add_argument("--stage2-epochs", dest="stage2_epochs", type=int, default=500)
    p.add_argument("--stage2-patience", dest="stage2_patience", type=int, default=30)
    p.add_argument("--stage2-lr", dest="stage2_lr", type=float, default=1e-4)
    p.add_argument("--stage2-batch-size", dest="stage2_batch_size", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.5)

    # ── CV + tasks ─────────────────────────────────────────────────────
    p.add_argument("--n-splits", dest="n_splits", type=int, default=30)
    p.add_argument("--tasks", type=_parse_tasks, default=tuple(range(8)),
                   help="Comma-separated task IDs (0-7). Default: all 8.")
    p.add_argument("--min-trials", dest="min_trials", type=int, default=1)

    # ── Attention prior (manual task weighting facilities) ─────────────
    prior_group = p.add_mutually_exclusive_group()
    prior_group.add_argument(
        "--attention-prior", dest="attention_prior",
        choices=list(ATTENTION_PRIOR_SCHEMES.keys()), default="none",
        help="Warm-start scheme for the per-task attention bias.\n"
             "  none  → no warm start (uniform init)\n"
             "  wvote → legacy wvote scheme [0.0, 0.5, 0.5, 0.5, 0.0, 1.5, 1.5, 1.5]\n"
             "  loo   → scaled per-task LOO ΔAUROC from the wvote probe",
    )
    prior_group.add_argument(
        "--custom-prior", dest="custom_prior", type=_parse_custom_prior, default=None,
        help="Ad-hoc manual weights: 8 comma-separated floats (one per task). "
             "Mutually exclusive with --attention-prior.",
    )

    # ── Resume ─────────────────────────────────────────────────────────
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to a state.pkl to resume from. "
                        "Skips completed folds; continues from the last one.")

    return p.parse_args()


def _setup_logging(log_path: Path, mode_tag: str) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(f"%(asctime)s | [{mode_tag}] | %(levelname)s | %(name)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8"); fh.setFormatter(fmt); root.addHandler(fh)
    sh = logging.StreamHandler(); sh.setFormatter(fmt); root.addHandler(sh)


def _resolve_attention_prior(args) -> tuple:
    """Return (torch.Tensor of prior or None, run-id suffix string)."""
    if args.custom_prior is not None:
        return (
            torch.tensor(args.custom_prior, dtype=torch.float32).reshape(NUM_TASKS, 1),
            "custom",
        )
    return resolve_prior(args.attention_prior, num_tasks=NUM_TASKS), args.attention_prior


def main():
    args = _parse_args()
    ensure_output_dirs()

    prior_tensor, prior_tag = _resolve_attention_prior(args)

    # Run-id stamps every meaningful configuration change
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_sift_dbt_swin_s1{args.stage1_epochs:03d}"
        + f"_s2{args.stage2_epochs:03d}"
        + "_optA"
        + f"_prior_{prior_tag}"
    )
    if tuple(args.tasks) != tuple(range(8)):
        run_id += "_t" + "".join(str(i) for i in args.tasks)

    log_path = LOGS_DIR / f"run_{run_id}.log"
    _setup_logging(log_path, mode_tag="SIFT-DBT")
    log = logging.getLogger(__name__)
    log.info(
        "Run ID: %s | tasks=%s | dropout=%.2f | prior=%s | S1(ep=%d, pat=%d, lr=%.0e) | S2(ep=%d, pat=%d, lr=%.0e)",
        run_id, args.tasks, args.dropout, prior_tag,
        args.stage1_epochs, args.stage1_patience, args.stage1_lr,
        args.stage2_epochs, args.stage2_patience, args.stage2_lr,
    )
    if prior_tensor is not None:
        log.info("Attention prior values (per task): %s", prior_tensor.squeeze().tolist())
    log.info("Log file: %s", log_path)

    # Load or rebuild the CWT cache (four_error mode, 45° artifact threshold)
    pipeline = EventLockedCWTPipeline(
        pre_stimulus_sec=0.2, post_stimulus_sec=0.8,
        min_freq=15.0, max_freq=60.0,
        freq_bins=32, target_time_bins=32, w_morlet=4.0,
        artifact_threshold=45.0,
        cache_path=CACHE_DIR / "data_store_meta_4err.pkl",
        signal_mode="four_error",
    )
    if not DATA_DIR.exists():
        log.error("Data directory not found: %s", DATA_DIR)
        return
    log.info("Loading 4-error CWT cache from: %s", pipeline.cache_path)
    pipeline.process_directory(DATA_DIR)

    # Datasets: FlatWindowDataset for Stage 1 (all 8 tasks always), SubjectBundleDataset for Stage 2
    flat_ds = FlatWindowDataset(pipeline.data_store, keep_task_ids=list(range(NUM_TASKS)))
    log.info("Stage 1 flat windows: %d (across %d unique subjects)",
             len(flat_ds), len(set(flat_ds.subject_ids)))

    bundle_ds = SubjectBundleDataset(
        pipeline.data_store,
        keep_task_ids=args.tasks,
        min_trials=args.min_trials,
    )
    log.info("Stage 2 subject bundles: %d  (shape=%s)  dropped=%d",
             len(bundle_ds), bundle_ds.shape, len(bundle_ds.dropped))
    if len(bundle_ds) == 0:
        log.error("No subjects pass Stage 2 admission floor. Aborting.")
        return

    # Run-checkpoint dir + orchestrator
    run_ckpt_dir = CHECKPOINTS_DIR / f"run_{run_id}"
    ht = HybridTrainer(
        flat_dataset=flat_ds,
        bundle_dataset=bundle_ds,
        checkpoint_dir=run_ckpt_dir,
        num_tasks=len(args.tasks),
        n_splits=args.n_splits,
        stage1_epochs=args.stage1_epochs,
        stage1_patience=args.stage1_patience,
        stage1_lr=args.stage1_lr,
        stage1_batch_size=args.stage1_batch_size,
        stage2_epochs=args.stage2_epochs,
        stage2_patience=args.stage2_patience,
        stage2_lr=args.stage2_lr,
        stage2_batch_size=args.stage2_batch_size,
        stage2_dropout=args.dropout,
        attention_prior=prior_tensor,
        resume_from=args.resume,
    )
    ht.run()


if __name__ == "__main__":
    main()
