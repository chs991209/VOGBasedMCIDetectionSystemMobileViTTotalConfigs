"""Full-experiment (8-task) entry point.

Mirrors the 2-experiment detection_caller in layout and conventions:
self-bootstraps sys.path so it can be launched from any working directory,
uses the shared paths.py + probe_generator, writes per-run output dirs under
outputs/, and stamps every log line with a tag ([FULL    ] or [FULL+AUG]).

Launch directly:
    python src/full_experiments_using/detection_caller/detection_caller.py [--augment]

Or dispatch via the main entry point:
    python src/detection_caller/detection_caller.py --full-experiments-using [--augment]
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Bootstrap: add src/ to sys.path so shared modules (paths, probe_generators)
# AND the full_experiments_using package both resolve cleanly.
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paths import (  # noqa: E402
    CACHE_DIR,
    CHECKPOINTS_DIR,
    DATA_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    ensure_output_dirs,
)
from probe_generators.probe_generator import TaskWiseProbeGenerator  # noqa: E402

from full_experiments_using.data_processor.data_engineering import (  # noqa: E402
    EventLockedCWTPipeline,
    TaskConditionedDataset,
)
from full_experiments_using.evaluators.monte_carlo_evaluator import (  # noqa: E402
    MonteCarloGroupEvaluator,
)


_DEFAULT_DROPOUT = 0.3
_DEFAULT_PATIENCE = 40

# Weighted soft-vote scheme derived from the task-contribution probe analysis:
# Vertical B / B-anti / R (5,6,7) up-weighted 1.5×; Horizontal B / B-anti / R
# (1,2,3) kept at 0.5×; low-information "A" tasks (0,4) excluded (weight 0).
WEIGHTED_VOTE_SCHEME = {0: 0.0, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.0, 5: 1.5, 6: 1.5, 7: 1.5}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-experiment (8-task) MCI detection pipeline.")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Wrap the training Subset in AugmentedSubset (SpecAugment-style freq+time masking).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=_DEFAULT_DROPOUT,
        help=f"Dropout probability for the classification head (default {_DEFAULT_DROPOUT}). "
             f"Run id is suffixed with _dropNNN when value differs from default, so the run "
             f"won't collide with the default plain/aug runs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=_DEFAULT_PATIENCE,
        help=f"Early-stopping patience on val-AUROC (default {_DEFAULT_PATIENCE}). "
             f"Run id is suffixed with _patNNN when value differs from default.",
    )
    parser.add_argument(
        "--weighted-vote",
        dest="weighted_vote",
        action="store_true",
        help="Use weighted subject-level soft-voting with the probe-derived scheme "
             f"{WEIGHTED_VOTE_SCHEME} (Vertical B/B-anti/R ×1.5, Horizontal B/B-anti/R ×0.5, "
             "A tasks excluded). Affects only aggregation, not training. Run id suffixed _wvote.",
    )
    return parser.parse_args()


def _setup_logging(log_path: Path, mode_tag: str) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        f"%(asctime)s | [{mode_tag}] | %(levelname)s | %(name)s | %(message)s"
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)


def main():
    args = _parse_args()

    ensure_output_dirs()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_full"
    if args.augment:
        run_id += "_aug"
    # Tag the run with the dropout value only when it differs from the default,
    # so existing default runs keep their canonical names and the new ablation
    # run gets a distinct, self-documenting id.
    if abs(args.dropout - _DEFAULT_DROPOUT) > 1e-6:
        run_id += f"_drop{int(round(args.dropout * 100)):03d}"
    if int(args.patience) != _DEFAULT_PATIENCE:
        run_id += f"_pat{int(args.patience):03d}"
    if args.weighted_vote:
        run_id += "_wvote"

    task_weights = WEIGHTED_VOTE_SCHEME if args.weighted_vote else None

    mode_tag = "FULL+AUG" if args.augment else "FULL    "

    log_path = LOGS_DIR / f"run_{run_id}.log"
    _setup_logging(log_path, mode_tag=mode_tag)
    log = logging.getLogger(__name__)
    log.info(
        "Run ID: %s | augment=%s | dropout=%.2f | patience=%d | weighted_vote=%s",
        run_id, args.augment, args.dropout, args.patience,
        (task_weights if task_weights is not None else "off"),
    )
    log.info("Log file: %s", log_path)

    pipeline = EventLockedCWTPipeline(
        pre_stimulus_sec=0.2,
        post_stimulus_sec=0.8,
        min_freq=15.0,
        max_freq=60.0,
        freq_bins=32,
        target_time_bins=32,
        w_morlet=4.0,
        artifact_threshold=30.0,
        # Dedicated cache file: 8-task tensors never collide with the 2-task cache.
        cache_path=CACHE_DIR / "data_store_full.pkl",
    )

    if not DATA_DIR.exists():
        log.error("Data directory not found: %s", DATA_DIR)
        return

    log.info("Loading full 8-task VOG saccade tensors from: %s", DATA_DIR)
    pipeline.process_directory(DATA_DIR)

    dataset = TaskConditionedDataset(pipeline.data_store)
    log.info("Total epoched samples: %d", len(dataset))

    if len(dataset) == 0:
        log.error("No data processed. Check file paths and headers.")
        return

    run_checkpoint_dir = CHECKPOINTS_DIR / f"run_{run_id}"
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"run_{run_id}_task_wise_probe.md"

    probe = TaskWiseProbeGenerator(
        num_tasks=8,
        output_path=report_path,
        report_style="full",  # 8-task layout with reflexive-vs-anti deep dive
    )
    mc = MonteCarloGroupEvaluator(
        dataset,
        max_epochs=500,
        batch_size=32,
        n_splits=30,
        probe=probe,
        checkpoint_dir=run_checkpoint_dir,
        num_tasks=8,
        augment=args.augment,
        early_stop_patience=args.patience,
        dropout=args.dropout,
        task_weights=task_weights,
    )
    mc.run()


if __name__ == "__main__":
    main()
