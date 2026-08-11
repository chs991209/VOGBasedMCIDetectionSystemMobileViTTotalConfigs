"""Full-experiment (8-task) entry point.

Mirrors the 2-experiment detection_caller in layout and conventions:
self-bootstraps sys.path so it can be launched from any working directory,
uses the shared paths.py + probe_generator, writes per-run output dirs under
outputs/, and stamps every log line with a tag ([FULL    ] or [FULL+AUG]).

Launch directly:
    python src/four_error_using/detection_caller/detection_caller.py [--augment]

Or dispatch via the main entry point:
    python src/detection_caller/detection_caller.py --full-experiments-using [--augment]
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Bootstrap: add src/ to sys.path so shared modules (paths, probe_generators)
# AND the four_error_using package both resolve cleanly.
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

from four_error_using.data_processor.data_engineer import (  # noqa: E402
    EventLockedCWTPipeline,
    TaskConditionedDataset,
)
from four_error_using.evaluators.repetitive_validator import (  # noqa: E402
    RepetitiveGroupValidator,
)


_DEFAULT_DROPOUT = 0.5
_DEFAULT_PATIENCE = 30
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_ARTIFACT_THRESHOLD = 45.0
_NUM_TASKS = 8

# Default weighted soft-vote scheme derived from the task-contribution probe
# analysis. Vertical B / B-anti / R (5,6,7) up-weighted 1.5×; Horizontal B /
# B-anti / R (1,2,3) kept at 0.5×; low-information "A" tasks (0,4) excluded
# (weight 0). Used when --weighted-vote is given without explicit --vote-weights.
WEIGHTED_VOTE_SCHEME = {0: 0.0, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.0, 5: 1.5, 6: 1.5, 7: 1.5}

# Fixed per-class TEST-subject counts for --stratified sampling, matching the
# dataset (14 HC / 23 MCI subjects): HC test=4 (train=10), MCI test=8 (train=15)
# → 25 train / 12 test each fold, with random subject membership per fold.
# Keys are class labels (HC=0, MCI=1) as assigned in TaskConditionedDataset.
STRATIFIED_TEST_COUNTS = {0: 4, 1: 8}


def _parse_vote_weights(spec: str) -> dict:
    """Parse a CLI vote-weight spec into a {task_id: weight} dict.

    Accepts the eight per-task weights separated by whitespace and/or commas,
    e.g. "0.0 0.5 1.0 1.0 0.0 1.0 2.0 2.0". Validates that exactly _NUM_TASKS
    non-negative floats are given and that at least one is > 0 (an all-zero
    scheme would make every subject fall back to the unweighted mean)."""
    tokens = [t for t in spec.replace(",", " ").split() if t]
    try:
        vals = [float(t) for t in tokens]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--vote-weights: non-numeric value in {spec!r} ({e})")
    if len(vals) != _NUM_TASKS:
        raise argparse.ArgumentTypeError(
            f"--vote-weights: expected {_NUM_TASKS} weights, got {len(vals)} in {spec!r}"
        )
    if any(v < 0 for v in vals):
        raise argparse.ArgumentTypeError(f"--vote-weights: weights must be >= 0, got {spec!r}")
    if sum(vals) <= 0:
        raise argparse.ArgumentTypeError("--vote-weights: at least one weight must be > 0")
    return {i: vals[i] for i in range(_NUM_TASKS)}


def _weights_tag(weights: dict) -> str:
    """Compact, self-documenting run_id fragment from a weight dict: each weight
    encoded as tenths and joined by '-', e.g. {0:0.0,1:0.5,...,7:2.0} -> 'w0-5-...-20'."""
    return "w" + "-".join(str(int(round(weights[i] * 10))) for i in range(_NUM_TASKS))


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
        "--batch-size",
        dest="batch_size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help=f"Training mini-batch size (default {_DEFAULT_BATCH_SIZE}, the legacy Jetson "
             f"recipe). The A6000 (48 GB) can go much higher, but larger batches change "
             f"BatchNorm stats / gradient noise vs. prior runs, so raise it only when you "
             f"don't need exact comparability. Inference batch is scaled automatically and "
             f"is numerically inert. Run id suffixed _bsNNN when value differs from default.",
    )
    parser.add_argument(
        "--weighted-vote",
        dest="weighted_vote",
        action="store_true",
        help="Use weighted subject-level soft-voting with the built-in probe-derived scheme "
             f"{WEIGHTED_VOTE_SCHEME} (Vertical B/B-anti/R ×1.5, Horizontal B/B-anti/R ×0.5, "
             "A tasks excluded). Ignored if --vote-weights is given. Affects only aggregation, "
             "not training. Run id suffixed _wvote_artifact<threshold>.",
    )
    parser.add_argument(
        "--vote-weights",
        dest="vote_weights",
        type=_parse_vote_weights,
        default=None,
        metavar='"w0 w1 ... w7"',
        help=f"Manual weighted soft-vote scheme: {_NUM_TASKS} per-task weights (>=0, not all "
             'zero) separated by spaces and/or commas, e.g. "0.0 0.5 1.0 1.0 0.0 1.0 2.0 2.0". '
             "Implies weighted voting (no separate --weighted-vote needed) and overrides the "
             "built-in scheme. Affects only aggregation, not training. Run id is tagged with a "
             "compact encoding of the weights (e.g. _w0-5-10-10-0-10-20-20) so sweep outputs "
             "stay distinguishable.",
    )
    parser.add_argument(
        "--artifact-threshold",
        dest="artifact_threshold",
        type=float,
        default=_DEFAULT_ARTIFACT_THRESHOLD,
        help=f"Max abs baseline-corrected gaze error (deg) before an epoch is rejected "
             f"(default {_DEFAULT_ARTIFACT_THRESHOLD}). Part of the CWT cache signature, so "
             f"changing it triggers a one-time reprocess of the CSVs. When weighted voting is "
             f"on, the run id carries _artifact<threshold>.",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Use stratified group sampling for the Monte-Carlo folds: a FIXED per-class "
             f"test-subject count {STRATIFIED_TEST_COUNTS} (HC=0, MCI=1 → HC test=4/train=10, "
             "MCI test=8/train=15 = 25 train / 12 test) with random subject membership each "
             "fold (still grouped by subject → no leakage). Keeps the repeated-random 30-fold "
             "design but holds the HC/MCI ratio constant across folds. Default (off) is the "
             "unstratified GroupShuffleSplit. Run id suffixed _strat.",
    )
    parser.add_argument(
        "--signal-mode",
        dest="signal_mode",
        choices=["legacy", "four_error", "full_error"],
        default="legacy",
        help="CWT channel representation. 'legacy' (default, 4ch) = [mag_L, re_L, mag_R, re_R] "
             "(task-axis, per eye). 'four_error' (4ch) = [|CWT(LH-TH)|, |CWT(RH-TH)|, "
             "|CWT(LV-TV)|, |CWT(RV-TV)|] (both axes, magnitude only). 'full_error' (8ch) = all "
             "traits: both axes × both eyes × (mag, re) = [mag_LH, re_LH, mag_RH, re_RH, mag_LV, "
             "re_LV, mag_RV, re_RV]. Each mode uses a separate cache; run id suffixed "
             "_4err / _8err. Adapter in_channels adjusts automatically.",
    )
    parser.add_argument(
        "--no-artifact-reject",
        dest="no_artifact_reject",
        action="store_true",
        help="Disable artifact rejection: keep ALL event-locked trials, including the "
             "large-gaze-error ones normally dropped by --artifact-threshold. Uses a dedicated "
             "'_allTrials' cache and run-id label so it never collides with the gated caches. "
             "Overrides --artifact-threshold.",
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
    # Single source of truth for the artifact-rejection threshold: used both in
    # the run_id suffix and the pipeline below, so the run name can never claim a
    # different value than the one actually applied.
    # --no-artifact-reject keeps every event-locked trial. Use a huge threshold
    # (matches the dedicated _allTrials cache signature) so the gate never fires.
    artifact_threshold = 1e6 if args.no_artifact_reject else args.artifact_threshold

    # Resolve the vote scheme: explicit --vote-weights wins; else --weighted-vote
    # uses the built-in scheme; else plain (unweighted) voting. --vote-weights
    # implies weighted mode on its own.
    if args.vote_weights is not None:
        task_weights = args.vote_weights
        custom_weights = True
    elif args.weighted_vote:
        task_weights = WEIGHTED_VOTE_SCHEME
        custom_weights = False
    else:
        task_weights = None
        custom_weights = False

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
    if int(args.batch_size) != _DEFAULT_BATCH_SIZE:
        run_id += f"_bs{int(args.batch_size):03d}"
    if task_weights is not None:
        thr_label = "allTrials" if args.no_artifact_reject else f"artifact{int(round(artifact_threshold))}"
        run_id += f"_wvote_{thr_label}"
        # Custom schemes additionally carry a compact weight encoding so multiple
        # sweeps at the same threshold don't collide on identical run ids.
        if custom_weights:
            run_id += f"_{_weights_tag(task_weights)}"
    if args.stratified:
        run_id += "_strat"
    if args.signal_mode == "four_error":
        run_id += "_4err"
    elif args.signal_mode == "full_error":
        run_id += "_8err"

    # Adapter input channels follow the representation: full_error stacks mag+re
    # on both axes/eyes → 8; the others are 4-channel.
    in_channels = 8 if args.signal_mode == "full_error" else 4

    mode_tag = "FULL+AUG" if args.augment else "FULL    "

    log_path = LOGS_DIR / f"run_{run_id}.log"
    _setup_logging(log_path, mode_tag=mode_tag)
    log = logging.getLogger(__name__)
    log.info(
        "Run ID: %s | augment=%s | dropout=%.2f | patience=%d | batch_size=%d | artifact=%s | signal_mode=%s (%dch) | stratified=%s | weighted_vote=%s",
        run_id, args.augment, args.dropout, args.patience, args.batch_size,
        ("OFF (all trials)" if args.no_artifact_reject else f"thr={artifact_threshold:.1f}"),
        args.signal_mode, in_channels,
        (STRATIFIED_TEST_COUNTS if args.stratified else False),
        (task_weights if task_weights is not None else "off"),
    )
    log.info("Log file: %s", log_path)

    cache_name = {
        "four_error": "data_store_full_4err.pkl",
        "full_error": "data_store_full_8err.pkl",
    }.get(args.signal_mode, "data_store_full.pkl")
    if args.no_artifact_reject:
        cache_name = cache_name.replace(".pkl", "_allTrials.pkl")

    pipeline = EventLockedCWTPipeline(
        pre_stimulus_sec=0.2,
        post_stimulus_sec=0.8,
        min_freq=15.0,
        max_freq=60.0,
        freq_bins=32,
        target_time_bins=32,
        w_morlet=4.0,
        artifact_threshold=artifact_threshold,
        signal_mode=args.signal_mode,
        # Separate cache per signal mode (and per rejection setting) so tensors of
        # different shape / trial-set never collide.
        cache_path=CACHE_DIR / cache_name,
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
    mc = RepetitiveGroupValidator(
        dataset,
        max_epochs=500,
        batch_size=args.batch_size,
        n_splits=30,
        probe=probe,
        checkpoint_dir=run_checkpoint_dir,
        num_tasks=8,
        augment=args.augment,
        early_stop_patience=args.patience,
        dropout=args.dropout,
        task_weights=task_weights,
        in_channels=in_channels,
        stratified=args.stratified,
        strat_test_counts=STRATIFIED_TEST_COUNTS if args.stratified else None,
    )
    mc.run()


if __name__ == "__main__":
    main()
