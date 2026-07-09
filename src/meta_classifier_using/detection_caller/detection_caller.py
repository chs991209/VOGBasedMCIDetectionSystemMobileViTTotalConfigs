"""Meta-classifier entry point (Distribution-Aware Gated Fusion).

Uses the 4-error CWT pipeline (`signal_mode='four_error'`): channels = CWT
magnitudes of {LH−TH, RH−TH, LV−TV, RV−TV}. A separate cache file is used
so this never overwrites the legacy `data_store_full.pkl`.

No padding mask, no Option C sequence fusion, no Youden's J threshold.
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Bootstrap so this script can be launched from any CWD.
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paths import (  # noqa: E402
    CACHE_DIR, CHECKPOINTS_DIR, DATA_DIR, LOGS_DIR, REPORTS_DIR, ensure_output_dirs,
)
# Re-use the 8-task CWT pipeline class — meta toggles `signal_mode='four_error'`.
from full_experiments_using.data_processor.data_engineering import EventLockedCWTPipeline  # noqa: E402

from meta_classifier_using.data_processor.data_preprocessing import SubjectBundleDataset  # noqa: E402
from meta_classifier_using.evaluators.monte_carlo_evaluator import MetaMonteCarloGroupEvaluator  # noqa: E402


_DEFAULT_DROPOUT = 0.5
_DEFAULT_PATIENCE = 40


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Meta-classifier (Distribution-Aware Gated Fusion) MCI detection pipeline."
    )
    p.add_argument("--dropout", type=float, default=_DEFAULT_DROPOUT,
                   help=f"Head dropout (default {_DEFAULT_DROPOUT}). "
                        "run_id suffixed _dropNNN when != default.")
    p.add_argument("--patience", type=int, default=_DEFAULT_PATIENCE,
                   help=f"Early-stop patience on val-AUROC (default {_DEFAULT_PATIENCE}). "
                        "run_id suffixed _patNNN when != default.")
    p.add_argument("--fc-hidden", dest="fc_hidden", type=int, default=128,
                   help="FC hidden dim of the final head (default 128).")
    p.add_argument("--max-epochs", dest="max_epochs", type=int, default=500)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    p.add_argument("--n-splits", dest="n_splits", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="AdamW learning rate (default 1e-4 — HDLSS regularization).")
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-2,
                   help="AdamW weight decay (default 1e-2 — HDLSS regularization).")
    return p.parse_args()


def _setup_logging(log_path: Path, mode_tag: str) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(f"%(asctime)s | [{mode_tag}] | %(levelname)s | %(name)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8"); fh.setFormatter(fmt); root.addHandler(fh)
    sh = logging.StreamHandler(); sh.setFormatter(fmt); root.addHandler(sh)


def main():
    args = _parse_args()

    ensure_output_dirs()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_meta_dagf"
    if abs(args.dropout - _DEFAULT_DROPOUT) > 1e-6:
        run_id += f"_drop{int(round(args.dropout * 100)):03d}"
    if int(args.patience) != _DEFAULT_PATIENCE:
        run_id += f"_pat{int(args.patience):03d}"

    log_path = LOGS_DIR / f"run_{run_id}.log"
    _setup_logging(log_path, mode_tag="META-DAGF")
    log = logging.getLogger(__name__)
    log.info(
        "Run ID: %s | dropout=%.2f | patience=%d | fc_hidden=%d | lr=%.0e | wd=%.0e",
        run_id, args.dropout, args.patience, args.fc_hidden, args.lr, args.weight_decay,
    )
    log.info("Log file: %s", log_path)

    # 4-error CWT pipeline — separate cache file from the legacy pipelines.
    pipeline = EventLockedCWTPipeline(
        pre_stimulus_sec=0.2, post_stimulus_sec=0.8,
        min_freq=15.0, max_freq=60.0,
        freq_bins=32, target_time_bins=32, w_morlet=4.0,
        artifact_threshold=30.0,
        cache_path=CACHE_DIR / "data_store_meta_4err.pkl",
        signal_mode="four_error",
    )
    if not DATA_DIR.exists():
        log.error("Data directory not found: %s", DATA_DIR)
        return
    log.info("Loading 8-task VOG tensors (4-error CWT) from: %s", DATA_DIR)
    pipeline.process_directory(DATA_DIR)

    dataset = SubjectBundleDataset(pipeline.data_store)
    log.info("Subject bundles built: %d  (bundle shape = %s)", len(dataset), dataset.shape)
    log.info("Strict-parity dropped subjects: %d", len(dataset.dropped))
    if len(dataset) == 0:
        log.error("No subjects pass the strict-parity rule. Aborting.")
        return

    run_ckpt_dir = CHECKPOINTS_DIR / f"run_{run_id}"
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    mc = MetaMonteCarloGroupEvaluator(
        dataset,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        n_splits=args.n_splits,
        checkpoint_dir=run_ckpt_dir,
        early_stop_patience=args.patience,
        dropout=args.dropout,
        fc_hidden=args.fc_hidden,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    mc.run()


if __name__ == "__main__":
    main()
