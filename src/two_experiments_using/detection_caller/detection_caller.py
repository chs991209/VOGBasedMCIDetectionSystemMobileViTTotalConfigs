"""2-experiment (Anti-Saccade B) entry point.

Mirrors the full-experiments-using caller in layout and conventions:
self-bootstraps sys.path so it can be launched from any working directory,
uses the shared paths.py + probe_generator, writes per-run output dirs under
outputs/, and stamps every log line with a tag ([BASE] or [AUG ]).

Launch directly:
    python src/two_experiments_using/detection_caller/detection_caller.py [--augment] [--patience N]

Or dispatch via the main entry point:
    python src/detection_caller/detection_caller.py [--augment] [--patience N]
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Bootstrap: add src/ to sys.path so shared modules (paths, probe_generators)
# AND the two_experiments_using package both resolve cleanly.
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

from two_experiments_using.data_processor.data_engineering import (  # noqa: E402
    EventLockedCWTPipeline,
    TaskConditionedDataset,
)
from two_experiments_using.evaluators.monte_carlo_evaluator import (  # noqa: E402
    MonteCarloGroupEvaluator,
)


_DEFAULT_PATIENCE = 40


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2-experiment Anti-Saccade B MCI detection pipeline."
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Wrap the training Subset in AugmentedSubset (SpecAugment-style freq+time masking).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=_DEFAULT_PATIENCE,
        help=f"Early-stopping patience on val-loss (default {_DEFAULT_PATIENCE}). "
             f"Run id is suffixed with _patNNN when value differs from default.",
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
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.augment:
        run_id += "_aug"
    if int(args.patience) != _DEFAULT_PATIENCE:
        run_id += f"_pat{int(args.patience):03d}"

    mode_tag = "AUG " if args.augment else "BASE"

    log_path = LOGS_DIR / f"run_{run_id}.log"
    _setup_logging(log_path, mode_tag=mode_tag)
    log = logging.getLogger(__name__)
    log.info(
        "Run ID: %s | augment=%s | patience=%d", run_id, args.augment, args.patience
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
        cache_path=CACHE_DIR / "data_store.pkl",
    )

    if not DATA_DIR.exists():
        log.error("Data directory not found: %s", DATA_DIR)
        return

    log.info("Filtering strictly for Horizontal/Vertical Anti-Saccade B tensors from: %s", DATA_DIR)
    pipeline.process_directory(DATA_DIR)

    dataset = TaskConditionedDataset(pipeline.data_store)
    log.info("Total epoched samples (anti-saccades only): %d", len(dataset))

    if len(dataset) == 0:
        log.error("No data processed. Check file paths and headers.")
        return

    run_checkpoint_dir = CHECKPOINTS_DIR / f"run_{run_id}"
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"run_{run_id}_anti_saccade_b_probe.md"

    probe = TaskWiseProbeGenerator(num_tasks=2, output_path=report_path)
    mc = MonteCarloGroupEvaluator(
        dataset,
        max_epochs=500,
        batch_size=32,
        n_splits=30,
        probe=probe,
        checkpoint_dir=run_checkpoint_dir,
        num_tasks=2,
        augment=args.augment,
        patience=args.patience,
    )
    mc.run()


if __name__ == "__main__":
    main()
