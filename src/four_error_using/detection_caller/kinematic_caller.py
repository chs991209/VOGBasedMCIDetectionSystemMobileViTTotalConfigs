"""Kinematic-feature experiment: velocity (high/low), latency, velocity-variance.

Standalone system (no CWT / MobileViT). Extracts 3 feature groups per event-locked
trial, aggregates task-weighted per subject, and runs stratified 30-fold
subject-grouped CV with a logistic model. Same cohort / artifact rule as four_error.

Run:
    python src/four_error_using/detection_caller/kinematic_caller.py --artifact-threshold 30
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paths import CACHE_DIR, DATA_DIR, LOGS_DIR, REPORTS_DIR  # noqa: E402
from four_error_using.data_processor.kinematic_features import (  # noqa: E402
    KinematicFeaturePipeline, FEATURE_NAMES,
)
from four_error_using.evaluators.kinematic_validator import run_cv, DEFAULT_WEIGHTS  # noqa: E402


def _args():
    p = argparse.ArgumentParser(description="Kinematic (velocity/latency/variance) HC-vs-MCI experiment.")
    p.add_argument("--artifact-threshold", type=float, default=30.0, dest="thr",
                   help="Gaze-error rejection threshold (deg). Default 30 — same as the adopted CWT run.")
    p.add_argument("--n-splits", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _save_confusion(conf, out_png, title):
    TP, FN, FP, TN = conf
    mat = np.array([[TN, FP], [FN, TP]])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred HC", "pred MCI"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true HC", "true MCI"])
    mx = mat.max() or 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    fontsize=15, color="white" if mat[i, j] > mx / 2 else "black")
    acc = (TP + TN) / max(TP + TN + FP + FN, 1)
    se = TP / max(TP + FN, 1); sp = TN / max(TN + FP, 1)
    ax.set_title(f"{title}\nAcc {acc:.3f} · Sens {se:.3f} · Spec {sp:.3f}", fontsize=8)
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    a = _args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_kinematic_artifact{int(round(a.thr))}"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(LOGS_DIR / f"run_{run_id}.log"), logging.StreamHandler()])
    log = logging.getLogger("kinematic")
    log.info("Run ID: %s | artifact thr=%.1f | features=%s | task-weighted aggregation=%s",
             run_id, a.thr, FEATURE_NAMES, DEFAULT_WEIGHTS)

    pipe = KinematicFeaturePipeline(artifact_threshold=a.thr,
                                    cache_path=CACHE_DIR / f"kinematic_features_thr{int(round(a.thr))}.pkl")
    pipe.process_directory(DATA_DIR)
    if not pipe.data_store:
        log.error("No kinematic features extracted."); return

    res = run_cv(pipe.data_store, FEATURE_NAMES, weights=DEFAULT_WEIGHTS,
                 n_splits=a.n_splits, seed=a.seed)

    def f(v): return f"{v[0]:.3f} ± {v[1]:.3f}"
    log.info("=== MC-CV Results (kinematic, %d subjects) ===", res["n_hc"] + res["n_mci"])
    log.info("Accuracy    : %s", f(res["acc"]))
    log.info("Sensitivity : %s", f(res["sens"]))
    log.info("Specificity : %s", f(res["spec"]))
    log.info("AUROC       : %s", f(res["auroc"]))
    log.info("Confusion (TP,FN,FP,TN): %s", res["conf"])
    log.info("Mean logistic coef (standardized): %s", res["coef_mean"])

    out_dir = REPORTS_DIR / "kinematic_using"
    _save_confusion(res["conf"], out_dir / "confusion_matrix.png",
                    f"kinematic · stratified · thr{int(round(a.thr))}")
    (out_dir / "metrics.txt").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.txt").write_text(
        f"Kinematic-feature experiment (velocity hi/lo, latency, velocity-variance)\n"
        f"Run: {run_id}\nSubjects: HC={res['n_hc']} MCI={res['n_mci']}  "
        f"(stratified 30-fold, HC=4/MCI=8 test, artifact thr {a.thr:.0f})\n"
        f"Aggregation: task-weighted per subject (A=0, H=0.5, V=1.5)\n\n"
        f"Accuracy    : {f(res['acc'])}\n"
        f"Sensitivity : {f(res['sens'])}\n"
        f"Specificity : {f(res['spec'])}\n"
        f"AUROC       : {f(res['auroc'])}\n\n"
        f"Confusion (pooled): TN={res['conf'][3]} FP={res['conf'][2]} "
        f"FN={res['conf'][1]} TP={res['conf'][0]}\n"
        f"Mean logistic coefficients (standardized): {res['coef_mean']}\n")
    log.info("Wrote %s + confusion_matrix.png", out_dir / "metrics.txt")


if __name__ == "__main__":
    main()
