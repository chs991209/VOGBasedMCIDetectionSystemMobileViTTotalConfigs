"""Render the cached CWT tensors as 2×2 PNG spectrograms.

Output layout:
    imgs/wavelet_transformed_spectrograms/
        <GROUP>/<subject_id>/task<id>_<task_name>/
            trial<n>.png

Each PNG is a 2×2 grid of the 4 CWT channels. Frequencies and time axes are
derived from the cache's config signature (no magic numbers).

CLI examples:
    # Render up to 3 trials per (subject, task) from the meta 4-error cache:
    python src/scalograms_generators/generator.py --trials 3

    # Render from the legacy cache instead:
    python src/scalograms_generators/generator.py \\
        --cache outputs/cache/data_store_full.pkl

    # Only HC subjects, only tasks 0 and 4, all trials:
    python src/scalograms_generators/generator.py \\
        --groups HC --tasks 0,4 --trials -1
"""
import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paths import CACHE_DIR, WAVELET_IMGS_DIR  # noqa: E402

logger = logging.getLogger("scalograms_generators")


_TASK_NAMES = {
    0: "HSacA",
    1: "HSacB",
    2: "HSacB_anti",
    3: "HSacR",
    4: "VSacA",
    5: "VSacB",
    6: "VSacB_anti",
    7: "VSacR",
}

_FOUR_ERR_LABELS = ["|CWT(LH-TH)|", "|CWT(RH-TH)|", "|CWT(LV-TV)|", "|CWT(RV-TV)|"]
_LEGACY_LABELS   = ["|CWT(L_err)|",  "Re CWT(L_err)", "|CWT(R_err)|",  "Re CWT(R_err)"]


def _parse_csv_ints(s: str) -> list:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_csv_strs(s: str) -> list:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CWT spectrogram visualizer.")
    p.add_argument("--cache", type=Path,
                   default=CACHE_DIR / "data_store_meta_4err.pkl",
                   help="Cache file built by EventLockedCWTPipeline. "
                        "Default = the meta-pipeline 4-error cache.")
    p.add_argument("--out", type=Path, default=WAVELET_IMGS_DIR,
                   help="Output directory root (default imgs/wavelet_transformed_spectrograms).")
    p.add_argument("--trials", type=int, default=3,
                   help="Max trials per (subject, task). -1 = render all. Default 3.")
    p.add_argument("--groups", type=_parse_csv_strs, default=["HC", "MCI"],
                   help="Comma-separated groups to include (HC,MCI).")
    p.add_argument("--tasks", type=_parse_csv_ints, default=list(range(8)),
                   help="Comma-separated task IDs to include (0..7).")
    p.add_argument("--subjects", type=_parse_csv_strs, default=None,
                   help="Comma-separated subject_id substrings to include. "
                        "Default: all subjects.")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--cmap", default="viridis")
    return p.parse_args()


def _load_cache(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    cfg = payload.get("config", {})
    ds = payload.get("data_store", {})
    return cfg, ds


def _freq_axis(cfg: dict) -> np.ndarray:
    fmin = float(cfg.get("min_freq", 15.0))
    fmax = float(cfg.get("max_freq", 60.0))
    nfreq = int(cfg.get("freq_bins", 32))
    return np.logspace(np.log10(fmin), np.log10(fmax), nfreq)


def _time_axis(cfg: dict, n_time: int) -> np.ndarray:
    pre = float(cfg.get("pre_sec", 0.2))
    post = float(cfg.get("post_sec", 0.8))
    return np.linspace(-pre, post, n_time)


def _render_one(tensor: np.ndarray, labels: list, freqs: np.ndarray,
                times: np.ndarray, title: str, out_path: Path, cmap: str, dpi: int):
    """tensor: [4, F, T]"""
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0))
    for i, ax in enumerate(axes.flat):
        if i >= tensor.shape[0]:
            ax.axis("off"); continue
        ch = tensor[i]
        im = ax.imshow(
            ch,
            aspect="auto", origin="lower", cmap=cmap,
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
        )
        ax.set_yscale("log")
        ax.set_title(labels[i], fontsize=10)
        ax.set_xlabel("time (s, 0 = stimulus)")
        ax.set_ylabel("freq (Hz)")
        ax.axvline(0.0, color="white", lw=0.6, alpha=0.6)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

    cfg, ds = _load_cache(args.cache)
    signal_mode = cfg.get("signal_mode", "legacy")
    labels = _FOUR_ERR_LABELS if signal_mode == "four_error" else _LEGACY_LABELS

    logger.info("Loaded cache: %s  (signal_mode=%s)", args.cache, signal_mode)
    logger.info("Output root : %s", args.out)
    logger.info("Filters     : groups=%s tasks=%s trials=%s subjects=%s",
                args.groups, args.tasks,
                "ALL" if args.trials == -1 else args.trials,
                args.subjects or "ALL")

    freqs = _freq_axis(cfg)
    n_imgs = 0
    n_subjects = 0
    n_skipped_empty = 0

    for group, subjects in ds.items():
        if group not in args.groups:
            continue
        for sid, epochs in subjects.items():
            if args.subjects and not any(s in sid for s in args.subjects):
                continue
            # Bucket epochs by task_id
            per_task = {t: [] for t in args.tasks}
            for tensor, tid, *_ in epochs:  # four_error epochs carry a 3rd item (ratio); ignore it
                if tid in per_task:
                    per_task[tid].append(tensor)
            if not any(per_task.values()):
                n_skipped_empty += 1
                continue
            n_subjects += 1
            for tid, trials in per_task.items():
                if not trials:
                    continue
                k = len(trials) if args.trials == -1 else min(args.trials, len(trials))
                for i in range(k):
                    arr = np.asarray(trials[i])
                    if arr.ndim != 3 or arr.shape[0] != 4:
                        logger.warning("Skip %s/%s task=%d trial=%d unexpected shape %s",
                                       group, sid, tid, i, arr.shape)
                        continue
                    times = _time_axis(cfg, arr.shape[-1])
                    task_name = _TASK_NAMES.get(tid, f"task{tid}")
                    title = f"{group} | {sid}  |  Task {tid}: {task_name}  |  trial {i+1}/{len(trials)}"
                    out_path = (args.out / group / sid /
                                f"task{tid}_{task_name}" /
                                f"trial{i+1:02d}.png")
                    _render_one(arr, labels, freqs, times, title,
                                out_path, args.cmap, args.dpi)
                    n_imgs += 1

    logger.info("Done. Wrote %d PNGs across %d subjects (skipped %d empty).",
                n_imgs, n_subjects, n_skipped_empty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
