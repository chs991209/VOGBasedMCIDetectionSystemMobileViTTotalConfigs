"""Legacy-style consolidated single-file version of the CWT visualizer package.

This file holds the SAME LOGIC as the divided files under
`spectrograms_generators/wavelet_transform_generators/` (data_tensor_processor.py,
mean_spectrograms_visualizer.py, variance_maps_visualizer.py,
squared_difference_maps_visualizer.py, generator.py) but inlined into one
module so you can copy-paste the whole pipeline into an external AI-review
tool (Gemini, ChatGPT, etc.) for analysis without juggling imports.

The divided package remains the authoritative production code. This file is
generated for review/portability only — keep them in sync if you change the
logic in either place.

Architecture (matches the divided package, top-to-bottom):
  1. WaveletSpectrogramCore   — loads the EventLockedCWTPipeline cache
                                (`outputs/cache/data_store_full.pkl`),
                                aggregates per-(subject, task) trials into
                                a per-eye mean magnitude spectrogram per
                                subject, and exposes a STFT-compatible
                                data_store[group][task_name][eye] layout.
  2. WaveletMeanVisualizer    — 4×8 cohort dashboard of CWT magnitudes
                                (rows: HC-H / HC-V / MCI-H / MCI-V,
                                 cols: Saccade A/B/B-anti/R × Left/Right).
  3. WaveletVarianceVisualizer — 4×8 dashboard of pixel-wise inter-subject
                                CWT variance (intra-group instability).
  4. WaveletDiffVisualizer    — 2×8 dashboard of (MCI_mean_dB − HC_mean_dB)²
                                (between-group separability heatmap).
  5. main()                   — single orchestrator: build Core once, run
                                all three visualizers.

Cache-channel mapping (legacy 4-channel CWT scheme from
EventLockedCWTPipeline in `full_experiments_using/data_processor/`):
  ch0 = |CWT(target_axis − L)|   →  'Left'  eye
  ch1 = Re(CWT(target_axis − L))  →  unused for dashboards
  ch2 = |CWT(target_axis − R)|   →  'Right' eye
  ch3 = Re(CWT(target_axis − R))  →  unused for dashboards

Output PNGs (when run as a script):
  imgs/wavelet_transformed_spectrograms/mean_spectrograms/Combined_32_Panels_CWT_Means.png
  imgs/wavelet_transformed_spectrograms/variance_maps/Combined_CWT_Variance_Maps.png
  imgs/wavelet_transformed_spectrograms/squared_difference_maps/Combined_CWT_Difference_Maps.png
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import argparse
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Inline paths (matches src/paths.py constants)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = _PROJECT_ROOT / "outputs" / "cache"
IMGS_DIR = _PROJECT_ROOT / "imgs"
WAVELET_IMGS_DIR = IMGS_DIR / "wavelet_transformed_spectrograms"

logger = logging.getLogger("wavelet_transform_generators.legacy_consolidated")


# Channel indices in the legacy 4-channel CWT cache
_CH_LEFT = 0    # mag_L : CWT magnitude of task-axis target-to-L error
_CH_RIGHT = 2   # mag_R : CWT magnitude of task-axis target-to-R error


# ===========================================================================
# 1. DATA TENSOR PROCESSOR
# ---------------------------------------------------------------------------
# Loads the project's CWT cache and aggregates event-locked per-trial tensors
# into a per-(subject, task, eye) mean magnitude spectrogram. Exposes a nested
# dict layout `data_store[group][task_name][eye] -> list[np.ndarray]` that the
# downstream visualizers consume.
# ===========================================================================

class WaveletSpectrogramCore:
    """Aggregates the cached event-locked CWT tensors into a per-(group, task,
    eye) list of *per-subject* mean magnitude spectrograms.

    For each subject who has ≥ 1 trial of a given task, the subject's CWT
    magnitude for that task is collapsed to a single [F, T] tensor by averaging
    over the subject's trials. The resulting tensor is appended to
    `data_store[group][task_name][eye]`. Subjects with zero trials of a task
    are silently skipped for *that* task only.
    """

    def __init__(self, cwt_cache_path):
        self.cwt_cache_path = Path(cwt_cache_path)
        self.task_map = {
            0: "Horizontal Saccade A",
            1: "Horizontal Saccade B",
            2: "Horizontal Saccade B (anti)",
            3: "Horizontal Saccade R",
            4: "Vertical Saccade A",
            5: "Vertical Saccade B",
            6: "Vertical Saccade B (anti)",
            7: "Vertical Saccade R",
        }
        # data_store[group][task_name][eye] -> list[np.ndarray of shape [F, T]]
        self.data_store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # Populated by load()
        self.min_freq = None
        self.max_freq = None
        self.freq_bins = None
        self.time_bins = None

    def load(self):
        if not self.cwt_cache_path.exists():
            raise FileNotFoundError(f"CWT cache not found: {self.cwt_cache_path}")
        with open(self.cwt_cache_path, "rb") as f:
            payload = pickle.load(f)
        cfg = payload.get("config", {})
        self.min_freq = float(cfg.get("min_freq", 15.0))
        self.max_freq = float(cfg.get("max_freq", 60.0))
        self.freq_bins = int(cfg.get("freq_bins", 32))
        self.time_bins = int(cfg.get("time_bins", 32))

        cwt_data = payload.get("data_store", {})
        n_subj = n_task_cells = 0
        for group, subjects in cwt_data.items():
            for sid, epochs in subjects.items():
                per_task = defaultdict(list)
                for tensor, task_id in epochs:
                    per_task[task_id].append(tensor)

                contributed = False
                for task_id, trials in per_task.items():
                    if task_id not in self.task_map or not trials:
                        continue
                    task_name = self.task_map[task_id]
                    mean_trial = np.mean(trials, axis=0)        # [4, F, T]
                    self.data_store[group][task_name]["Left"].append(mean_trial[_CH_LEFT])
                    self.data_store[group][task_name]["Right"].append(mean_trial[_CH_RIGHT])
                    n_task_cells += 1
                    contributed = True
                if contributed:
                    n_subj += 1

        logger.info(
            "Loaded CWT cache: %s  (subjects=%d, per-(subject,task) cells aggregated=%d)",
            self.cwt_cache_path, n_subj, n_task_cells,
        )


# ===========================================================================
# 2. MEAN SPECTROGRAM VISUALIZER
# ---------------------------------------------------------------------------
# 4×8 dashboard: rows = (HC-H, HC-V, MCI-H, MCI-V); cols = Saccade A/B/B-anti/R
# × Left/Right eyes. Each cell = pixel-wise linear-space mean of the subject
# CWT magnitudes contributing to that cell, then 10·log10 mapped for dB scale.
# ===========================================================================

class WaveletMeanVisualizer:
    """[평균 맵 시각화 전담 모듈 — CWT 버전]
    Core 로부터 CWT 매그니튜드 텐서를 주입받아 그룹(HC/MCI)의 집단 평균
    스펙트로그램 시각화만을 전담합니다.
    """

    def __init__(self, core_processor: WaveletSpectrogramCore):
        self.core = core_processor

    def plot_and_save(self, save_dir=None):
        if save_dir is None:
            save_dir = WAVELET_IMGS_DIR / "mean_spectrograms"
        save_path_obj = Path(save_dir)
        save_path_obj.mkdir(parents=True, exist_ok=True)

        base_tasks = ["Saccade A", "Saccade B", "Saccade B (anti)", "Saccade R"]
        columns_mapping = [(bt, eye) for bt in base_tasks for eye in ["Left", "Right"]]
        row_configs = [("HC", "Horizontal"), ("HC", "Vertical"),
                       ("MCI", "Horizontal"), ("MCI", "Vertical")]

        fig, axes = plt.subplots(nrows=len(row_configs), ncols=len(columns_mapping),
                                 figsize=(26, 12))
        fig.suptitle(
            f"Comprehensive CWT Mean Spectrograms ({self.core.min_freq:.0f}–{self.core.max_freq:.0f} Hz)",
            fontsize=22, fontweight="bold", y=1.02,
        )

        for row_idx, (group, axis_type) in enumerate(row_configs):
            for col_idx, (base_task, eye) in enumerate(columns_mapping):
                ax = axes[row_idx, col_idx]
                tensors = self.core.data_store[group].get(f"{axis_type} {base_task}", {}).get(eye, [])
                if not tensors:
                    ax.axis("off")
                    continue

                # Linear-space averaging of CWT magnitudes, then Log(dB) mapping
                mean_tensor_linear = np.mean(tensors, axis=0)
                mean_tensor_db = 10 * np.log10(np.clip(mean_tensor_linear, a_min=1e-10, a_max=None))

                ax.imshow(mean_tensor_db, aspect="auto", origin="lower", cmap="viridis")
                if row_idx == 0:
                    ax.set_title(f"{base_task}\n({eye})", fontsize=11, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{group} ({axis_type[0]})\n({self.core.min_freq:.0f}–{self.core.max_freq:.0f}Hz)",
                        fontsize=14, fontweight="bold",
                    )
                else:
                    ax.set_yticks([])
                if row_idx == len(row_configs) - 1:
                    ax.set_xlabel("Time Bins")
                else:
                    ax.set_xticks([])

        plt.tight_layout()
        output_file = save_path_obj / "Combined_32_Panels_CWT_Means.png"
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[+] CWT Mean 대시보드 시각화 완료: {output_file}")


# ===========================================================================
# 3. VARIANCE MAP VISUALIZER
# ---------------------------------------------------------------------------
# Same 4×8 layout, but each cell is pixel-wise variance across the subjects
# contributing to that cell (intra-group inconsistency). Cells with < 2
# subjects are blanked (variance undefined). Linear-space variance, then
# 10·log10 dB scaling for visual range compression. magma colormap to make
# high-instability regions pop.
# ===========================================================================

class WaveletVarianceVisualizer:
    """[분산 맵 시각화 전담 모듈 — CWT 버전]
    Core 로부터 CWT 매그니튜드 텐서를 주입받아 환자 집단(HC/MCI) 내부의
    고주파 에러 불안정성(Variance) 시각화만을 독립적으로 전담합니다.
    """

    def __init__(self, core_processor: WaveletSpectrogramCore):
        self.core = core_processor

    def plot_and_save(self, save_dir=None):
        if save_dir is None:
            save_dir = WAVELET_IMGS_DIR / "variance_maps"
        print("\n[*] CWT Variance Map 연산 및 시각화를 시작합니다...")
        save_path_obj = Path(save_dir)
        save_path_obj.mkdir(parents=True, exist_ok=True)

        base_tasks = ["Saccade A", "Saccade B", "Saccade B (anti)", "Saccade R"]
        columns_mapping = [(bt, eye) for bt in base_tasks for eye in ["Left", "Right"]]
        row_configs = [("HC", "Horizontal"), ("HC", "Vertical"),
                       ("MCI", "Horizontal"), ("MCI", "Vertical")]

        fig, axes = plt.subplots(nrows=len(row_configs), ncols=len(columns_mapping),
                                 figsize=(26, 12))
        fig.suptitle(
            f"CWT Variance Maps: Intra-Group Inconsistency ({self.core.min_freq:.0f}–{self.core.max_freq:.0f} Hz)",
            fontsize=22, fontweight="bold", y=1.02,
        )

        for row_idx, (group, axis_type) in enumerate(row_configs):
            for col_idx, (base_task, eye) in enumerate(columns_mapping):
                ax = axes[row_idx, col_idx]
                full_task_name = f"{axis_type} {base_task}"
                tensors = self.core.data_store[group].get(full_task_name, {}).get(eye, [])

                if len(tensors) < 2:
                    ax.axis("off")
                    continue

                # Pixel-wise variance across subjects (linear-space), then dB mapping
                variance_tensor_linear = np.var(tensors, axis=0)
                variance_tensor_db = 10 * np.log10(np.clip(variance_tensor_linear, a_min=1e-10, a_max=None))

                ax.imshow(variance_tensor_db, aspect="auto", origin="lower", cmap="magma")
                if row_idx == 0:
                    ax.set_title(f"{base_task}\n({eye})", fontsize=11, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{group} Var ({axis_type[0]})\n({self.core.min_freq:.0f}–{self.core.max_freq:.0f}Hz)",
                        fontsize=14, fontweight="bold",
                    )
                else:
                    ax.set_yticks([])
                if row_idx == len(row_configs) - 1:
                    ax.set_xlabel("Time Bins")
                else:
                    ax.set_xticks([])

        plt.tight_layout()
        output_file = save_path_obj / "Combined_CWT_Variance_Maps.png"
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[+] CWT Variance Map 시각화 완료: {output_file}")


# ===========================================================================
# 4. SQUARED-DIFFERENCE MAP VISUALIZER
# ---------------------------------------------------------------------------
# 2×8 layout: rows = Horizontal / Vertical; cols = Saccade A/B/B-anti/R ×
# Left/Right eyes. Each cell = (mean_MCI_dB − mean_HC_dB)² — the magnitude of
# the dB-domain group gap, per pixel. inferno colormap with per-axis
# colorbars; cells where either group has no subjects for that task are
# blanked.
# ===========================================================================

class WaveletDiffVisualizer:
    """[디퍼런스 맵 시각화 전담 모듈 — CWT 버전]
    Core 로부터 CWT 매그니튜드 텐서를 주입받아 (MCI − HC)² 형태의
    집단 간 고주파 에러 격차 시각화만을 독립적으로 전담합니다.
    """

    def __init__(self, core_processor: WaveletSpectrogramCore):
        self.core = core_processor

    def plot_and_save(self, save_dir=None):
        if save_dir is None:
            save_dir = WAVELET_IMGS_DIR / "squared_difference_maps"
        save_path_obj = Path(save_dir)
        save_path_obj.mkdir(parents=True, exist_ok=True)

        base_tasks = ["Saccade A", "Saccade B", "Saccade B (anti)", "Saccade R"]
        columns_mapping = [(bt, eye) for bt in base_tasks for eye in ["Left", "Right"]]
        row_configs = ["Horizontal", "Vertical"]

        fig, axes = plt.subplots(nrows=len(row_configs), ncols=len(columns_mapping),
                                 figsize=(26, 7))
        fig.suptitle(
            f"CWT Difference Maps: $(MCI - HC)^2$ ({self.core.min_freq:.0f}–{self.core.max_freq:.0f} Hz)",
            fontsize=22, fontweight="bold", y=1.05,
        )

        for row_idx, axis_type in enumerate(row_configs):
            for col_idx, (base_task, eye) in enumerate(columns_mapping):
                ax = axes[row_idx, col_idx]
                full_task_name = f"{axis_type} {base_task}"

                hc_tensors = self.core.data_store["HC"].get(full_task_name, {}).get(eye, [])
                mci_tensors = self.core.data_store["MCI"].get(full_task_name, {}).get(eye, [])

                if not hc_tensors or not mci_tensors:
                    ax.axis("off")
                    continue

                hc_mean_db = 10 * np.log10(np.clip(np.mean(hc_tensors, axis=0), a_min=1e-10, a_max=None))
                mci_mean_db = 10 * np.log10(np.clip(np.mean(mci_tensors, axis=0), a_min=1e-10, a_max=None))
                diff_squared = np.square(mci_mean_db - hc_mean_db)

                im = ax.imshow(diff_squared, aspect="auto", origin="lower", cmap="inferno")
                if row_idx == 0:
                    ax.set_title(f"{base_task}\n({eye})", fontsize=11, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{axis_type} Diff\n({self.core.min_freq:.0f}–{self.core.max_freq:.0f}Hz)",
                        fontsize=14, fontweight="bold",
                    )
                else:
                    ax.set_yticks([])
                if row_idx == len(row_configs) - 1:
                    ax.set_xlabel("Time Bins")
                else:
                    ax.set_xticks([])

                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        output_file = save_path_obj / "Combined_CWT_Difference_Maps.png"
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[+] CWT Difference Map 시각화 완료: {output_file}")


# ===========================================================================
# 5. ORCHESTRATOR  (== generator.py)
# ---------------------------------------------------------------------------
# Single entry point: build Core once, run all three visualizers in sequence.
# Run directly as a script, or import `main` and call it from another module.
# ===========================================================================

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CWT population dashboards (legacy consolidated).")
    p.add_argument("--cache", type=Path,
                   default=CACHE_DIR / "data_store_full.pkl",
                   help="CWT cache produced by EventLockedCWTPipeline (legacy 4-channel mode).")
    p.add_argument("--skip-mean", action="store_true")
    p.add_argument("--skip-variance", action="store_true")
    p.add_argument("--skip-diff", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args(argv)

    logger.info("CWT Core build — cache=%s", args.cache)
    core = WaveletSpectrogramCore(cwt_cache_path=args.cache)
    core.load()

    n_cells = sum(len(v) for g in core.data_store.values()
                          for t in g.values()
                          for v in t.values())
    logger.info("Core ready — %d (group, task, eye)-aggregated subject tensors total.", n_cells)
    logger.info("Output root: %s", WAVELET_IMGS_DIR)

    if not args.skip_mean:
        WaveletMeanVisualizer(core).plot_and_save()
    if not args.skip_variance:
        WaveletVarianceVisualizer(core).plot_and_save()
    if not args.skip_diff:
        WaveletDiffVisualizer(core).plot_and_save()

    return 0


if __name__ == "__main__":
    sys.exit(main())
