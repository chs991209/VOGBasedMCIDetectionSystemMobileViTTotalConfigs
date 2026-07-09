"""Top-level dispatcher.

Selects between the 2-experiment (Anti-Saccade B) and full-experiment (8-task)
pipelines based on the --full-experiments-using flag, and subprocess-launches
the appropriate sub-package's entry point. Keeps each pipeline isolated in its
own Python interpreter — separate sys.modules cache, separate logging root, no
risk of cross-contamination between concurrent runs.

Examples:
    # 2-exp Anti-Saccade B (default)
    python src/detection_caller/detection_caller.py
    python src/detection_caller/detection_caller.py --augment
    python src/detection_caller/detection_caller.py --patience 30

    # full-experiments-using (8-task)
    python src/detection_caller/detection_caller.py --full-experiments-using
    python src/detection_caller/detection_caller.py --full-experiments-using --augment --dropout 0.5
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MCI detection pipeline dispatcher. "
                    "Default → 2-experiment Anti-Saccade B; "
                    "--full-experiments-using → full 8-task pipeline.",
    )
    parser.add_argument(
        "--full-experiments-using",
        dest="full_experiments_using",
        action="store_true",
        help="Dispatch to the full 8-task pipeline "
             "(src/full_experiments_using/detection_caller/detection_caller.py). "
             "Default is the 2-experiment pipeline "
             "(src/two_experiments_using/detection_caller/detection_caller.py).",
    )
    parser.add_argument(
        "--meta",
        action="store_true",
        help="Dispatch to the meta-classifier pipeline "
             "(src/meta_classifier_using/detection_caller/detection_caller.py). "
             "Unified 8-task flatten-and-concat topology — no soft-voting.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="SpecAugment-style train-only augmentation. Forwarded to whichever sub-pipeline.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override the head dropout. Only meaningful for --full-experiments-using "
             "(2-exp has fixed dropout=0.5 by design); silently ignored in 2-exp mode.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stopping patience (default 40 in both pipelines). "
             "Run id is suffixed with _patNNN when value differs from default.",
    )
    parser.add_argument(
        "--weighted-vote",
        dest="weighted_vote",
        action="store_true",
        help="Weighted subject-level soft-vote (probe-derived task weights). "
             "Only meaningful for --full-experiments-using; ignored in 2-exp mode.",
    )
    return parser.parse_args()


def _dispatch(target_caller: Path, augment: bool, dropout, patience, weighted_vote, label: str) -> int:
    if not target_caller.exists():
        print(f"[!] caller not found: {target_caller}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(target_caller)]
    if augment:
        cmd.append("--augment")
    if dropout is not None:
        cmd.extend(["--dropout", str(dropout)])
    if patience is not None:
        cmd.extend(["--patience", str(int(patience))])
    if weighted_vote:
        cmd.append("--weighted-vote")

    print(f"[*] Dispatching to {label}: {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=os.getcwd())
    return completed.returncode


def main() -> None:
    args = _parse_args()

    if args.meta:
        target = _SRC_DIR / "meta_classifier_using" / "detection_caller" / "detection_caller.py"
        label = "meta-classifier-using"
        # Meta caller accepts --dropout and --patience but NOT --weighted-vote / --augment
        if args.augment:
            print("[!] --augment has no effect in meta mode (no per-window aggregation); ignored.",
                  file=sys.stderr)
        if args.weighted_vote:
            print("[!] --weighted-vote has no effect in meta mode (no soft-vote); ignored.",
                  file=sys.stderr)
        cmd = [sys.executable, str(target)]
        if args.dropout is not None:
            cmd.extend(["--dropout", str(args.dropout)])
        if args.patience is not None:
            cmd.extend(["--patience", str(int(args.patience))])
        print(f"[*] Dispatching to {label}: {' '.join(cmd)}", flush=True)
        import subprocess as _sp
        sys.exit(_sp.run(cmd, cwd=os.getcwd()).returncode)

    if args.full_experiments_using:
        target = _SRC_DIR / "full_experiments_using" / "detection_caller" / "detection_caller.py"
        label = "full-experiments-using"
        dropout_to_forward = args.dropout
        weighted_to_forward = args.weighted_vote
    else:
        target = _SRC_DIR / "two_experiments_using" / "detection_caller" / "detection_caller.py"
        label = "two-experiments-using"
        # 2-exp caller doesn't accept --dropout / --weighted-vote; drop them if passed.
        if args.dropout is not None:
            print(
                "[!] --dropout is only used with --full-experiments-using; ignored in 2-exp mode.",
                file=sys.stderr,
            )
        if args.weighted_vote:
            print(
                "[!] --weighted-vote is only used with --full-experiments-using; ignored in 2-exp mode.",
                file=sys.stderr,
            )
        dropout_to_forward = None
        weighted_to_forward = False

    sys.exit(_dispatch(
        target_caller=target,
        augment=args.augment,
        dropout=dropout_to_forward,
        patience=args.patience,
        weighted_vote=weighted_to_forward,
        label=label,
    ))


if __name__ == "__main__":
    main()
