"""Renewed meta-classifier entry point (Solution D: Dynamic Latent Classifier)."""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paths import CACHE_DIR, CHECKPOINTS_DIR, DATA_DIR, LOGS_DIR, ensure_output_dirs
from full_experiments_using.data_processor.data_engineering import EventLockedCWTPipeline

# [Solution D] Dataset & Evaluator Imports
from meta_classifier_renewed.data_processor.data_preprocessing import SubjectBundleDataset
from meta_classifier_renewed.evaluators.monte_carlo_evaluator import MetaMonteCarloGroupEvaluator


_DEFAULT_DROPOUT = 0.5
_DEFAULT_PATIENCE = 30
_DEFAULT_TASKS = (2, 6)   # HSacBanti, VSacBanti — Master Context §1, §5 (anti-saccade only)
_DEFAULT_MIN_TRIALS = 1 # 상한선(MAX)을 없애고, 최소 통과 기준(MIN)만 설정

_TASK_TAGS = {
    0: "HSacA", 1: "HSacB", 2: "HSacBanti", 3: "HSacR",
    4: "VSacA", 5: "VSacB", 6: "VSacBanti", 7: "VSacR",
}

def _parse_task_ids(s: str) -> tuple:
    ids = tuple(int(x) for x in s.split(",") if x.strip())
    return ids

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Renewed meta-classifier (Dynamic Latent Aggregation).")
    p.add_argument("--dropout", type=float, default=_DEFAULT_DROPOUT)
    p.add_argument("--patience", type=int, default=_DEFAULT_PATIENCE)
    p.add_argument("--tasks", dest="tasks", type=_parse_task_ids, default=_DEFAULT_TASKS)
    # [Solution D] max_trials -> min_trials 로 변경
    p.add_argument("--min-trials", dest="min_trials", type=int, default=_DEFAULT_MIN_TRIALS,
                   help="Admission Floor: minimum trials required per task (default 1).")
    p.add_argument("--max-epochs", dest="max_epochs", type=int, default=500)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    p.add_argument("--n-splits", dest="n_splits", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-2)
    return p.parse_args()

def _setup_logging(log_path: Path, mode_tag: str) -> None:
    root = logging.getLogger()
    for h in list(root.handlers): root.removeHandler(h)
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(f"%(asctime)s | [{mode_tag}] | %(levelname)s | %(name)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8"); fh.setFormatter(fmt); root.addHandler(fh)
    sh = logging.StreamHandler(); sh.setFormatter(fmt); root.addHandler(sh)

def main():
    args = _parse_args()
    ensure_output_dirs()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_dynamic_latent"
    if tuple(args.tasks) != _DEFAULT_TASKS:
        run_id += "_t" + "".join(str(i) for i in args.tasks)

    log_path = LOGS_DIR / f"run_{run_id}.log"
    _setup_logging(log_path, mode_tag="DYN-LATENT")
    log = logging.getLogger(__name__)

    task_names = ", ".join(f"{i}={_TASK_TAGS[i]}" for i in args.tasks)
    log.info("Run ID: %s | tasks=[%s] | min_trials=%d | batch=%d", run_id, task_names, args.min_trials, args.batch_size)

    pipeline = EventLockedCWTPipeline(
        pre_stimulus_sec=0.2, post_stimulus_sec=0.8,
        min_freq=15.0, max_freq=60.0, freq_bins=32, target_time_bins=32, w_morlet=4.0,
        artifact_threshold=45.0, cache_path=CACHE_DIR / "data_store_meta_4err.pkl", signal_mode="four_error",
    )

    if not DATA_DIR.exists():
        log.error("Data directory not found: %s", DATA_DIR)
        return

    pipeline.process_directory(DATA_DIR)

    # [Solution D] Dataset 초기화 시 min_trials 전달
    dataset = SubjectBundleDataset(
        pipeline.data_store,
        keep_task_ids=args.tasks,
        min_trials=args.min_trials,
    )

    log.info("Subject bundles built: %d (Shape: %s)", len(dataset), dataset.shape)
    if len(dataset) == 0:
        log.error("No subjects pass the admission floor. Aborting.")
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
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    mc.run()

if __name__ == "__main__":
    main()