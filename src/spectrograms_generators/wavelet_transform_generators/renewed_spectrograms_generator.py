"""Legacy-style consolidated single-file version of the CWT visualizer package.

This file holds the SAME LOGIC as the divided files under
`spectrograms_generators/wavelet_transform_generators/` but inlined into one
module so you can copy-paste the whole pipeline into an external AI-review tool.

Architecture:
  1. WaveletSpectrogramCore    - Loads cache & aggregates per-(subject, task) tensors
  2. WaveletMeanVisualizer     - 4x8 dashboard of Mean CWT magnitudes (dB)
  3. WaveletVarianceVisualizer - 4x8 dashboard of CWT Variance (Inconsistency)
  4. WaveletDiffVisualizer     - 2x8 dashboard of Raw Differences (MCI - HC)
  5. main()                    - Orchestrator
"""

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
# Inline paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = _PROJECT_ROOT / "outputs" / "cache"
IMGS_DIR = _PROJECT_ROOT / "imgs"
WAVELET_IMGS_DIR = IMGS_DIR / "wavelet_transformed_spectrograms"

logger = logging.getLogger("wavelet_transform_generators.legacy_consolidated")

# Channel indices in the legacy 4-channel CWT cache
_CH_LEFT = 0  # mag_L : CWT magnitude of task-axis target-to-L error
_CH_RIGHT = 2  # mag_R : CWT magnitude of task-axis target-to-R error


# ===========================================================================
# 1. DATA TENSOR PROCESSOR
# ===========================================================================
class WaveletSpectrogramCore:
    def __init__(self, cwt_cache_path):
        self.cwt_cache_path = Path(cwt_cache_path)
        self.task_map = {
            0: "Horizontal Saccade A", 1: "Horizontal Saccade B",
            2: "Horizontal Saccade B (anti)", 3: "Horizontal Saccade R",
            4: "Vertical Saccade A", 5: "Vertical Saccade B",
            6: "Vertical Saccade B (anti)", 7: "Vertical Saccade R",
        }
        self.data_store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
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
                    if task_id not in self.task_map or not trials: continue
                    task_name = self.task_map[task_id]
                    mean_trial = np.mean(trials, axis=0)  # [4, F, T]
                    self.data_store[group][task_name]["Left"].append(mean_trial[_CH_LEFT])
                    self.data_store[group][task_name]["Right"].append(mean_trial[_CH_RIGHT])
                    n_task_cells += 1
                    contributed = True
                if contributed: n_subj += 1

        logger.info(f"Loaded CWT cache: {self.cwt_cache_path} (subjects={n_subj}, cells={n_task_cells})")


# ===========================================================================
# 2. MEAN SPECTROGRAM VISUALIZER
# ===========================================================================
class WaveletMeanVisualizer:
    def __init__(self, core_processor: WaveletSpectrogramCore):
        self.core = core_processor

    def plot_and_save(self, save_dir=None):
        if save_dir is None: save_dir = WAVELET_IMGS_DIR / "mean_spectrograms"
        save_path_obj = Path(save_dir)
        save_path_obj.mkdir(parents=True, exist_ok=True)

        base_tasks = ["Saccade A", "Saccade B", "Saccade B (anti)", "Saccade R"]
        columns_mapping = [(bt, eye) for bt in base_tasks for eye in ["Left", "Right"]]
        row_configs = [("HC", "Horizontal"), ("HC", "Vertical"), ("MCI", "Horizontal"), ("MCI", "Vertical")]

        fig, axes = plt.subplots(nrows=len(row_configs), ncols=len(columns_mapping), figsize=(26, 12))
        fig.suptitle(f"Comprehensive CWT Mean Spectrograms ({self.core.min_freq:.0f}-{self.core.max_freq:.0f} Hz)",
                     fontsize=22, fontweight="bold", y=1.02)

        for row_idx, (group, axis_type) in enumerate(row_configs):
            for col_idx, (base_task, eye) in enumerate(columns_mapping):
                ax = axes[row_idx, col_idx]
                tensors = self.core.data_store[group].get(f"{axis_type} {base_task}", {}).get(eye, [])
                if not tensors:
                    ax.axis("off");
                    continue

                mean_tensor_linear = np.mean(tensors, axis=0)
                mean_tensor_db = 10 * np.log10(np.clip(mean_tensor_linear, a_min=1e-10, a_max=None))

                ax.imshow(mean_tensor_db, aspect="auto", origin="lower", cmap="viridis")

                # Event-Lock Line (t=0)
                time_bins = mean_tensor_db.shape[1]
                t_zero_idx = int(time_bins * 0.2)  # Assuming 0.2s pre-stimulus out of 1.0s total
                ax.axvline(x=t_zero_idx, color='white', linestyle='--', alpha=0.8, linewidth=1.5)

                if row_idx == 0: ax.set_title(f"{base_task}\n({eye})", fontsize=11, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(f"{group} ({axis_type[0]})\n({self.core.min_freq:.0f}-{self.core.max_freq:.0f}Hz)",
                                  fontsize=14, fontweight="bold")
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
# ===========================================================================
class WaveletVarianceVisualizer:
    def __init__(self, core_processor: WaveletSpectrogramCore):
        self.core = core_processor

    def plot_and_save(self, save_dir=None):
        if save_dir is None: save_dir = WAVELET_IMGS_DIR / "variance_maps"
        save_path_obj = Path(save_dir)
        save_path_obj.mkdir(parents=True, exist_ok=True)
        print("\n[*] CWT Variance Map 연산 및 시각화를 시작합니다...")

        base_tasks = ["Saccade A", "Saccade B", "Saccade B (anti)", "Saccade R"]
        columns_mapping = [(bt, eye) for bt in base_tasks for eye in ["Left", "Right"]]
        row_configs = [("HC", "Horizontal"), ("HC", "Vertical"), ("MCI", "Horizontal"), ("MCI", "Vertical")]

        fig, axes = plt.subplots(nrows=len(row_configs), ncols=len(columns_mapping), figsize=(26, 12))
        fig.suptitle(
            f"CWT Variance Maps: Intra-Group Inconsistency ({self.core.min_freq:.0f}-{self.core.max_freq:.0f} Hz)",
            fontsize=22, fontweight="bold", y=1.02)

        for row_idx, (group, axis_type) in enumerate(row_configs):
            for col_idx, (base_task, eye) in enumerate(columns_mapping):
                ax = axes[row_idx, col_idx]
                tensors = self.core.data_store[group].get(f"{axis_type} {base_task}", {}).get(eye, [])

                if len(tensors) < 2:
                    ax.axis("off");
                    continue

                variance_tensor_linear = np.var(tensors, axis=0)
                variance_tensor_db = 10 * np.log10(np.clip(variance_tensor_linear, a_min=1e-10, a_max=None))

                ax.imshow(variance_tensor_db, aspect="auto", origin="lower", cmap="magma")

                # Event-Lock Line (t=0)
                time_bins = variance_tensor_db.shape[1]
                t_zero_idx = int(time_bins * 0.2)
                ax.axvline(x=t_zero_idx, color='white', linestyle='--', alpha=0.8, linewidth=1.5)

                if row_idx == 0: ax.set_title(f"{base_task}\n({eye})", fontsize=11, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{group} Var ({axis_type[0]})\n({self.core.min_freq:.0f}-{self.core.max_freq:.0f}Hz)",
                        fontsize=14, fontweight="bold")
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
# 4. RAW DIFFERENCE MAP VISUALIZER (MCI - HC)
# ===========================================================================
class WaveletDiffVisualizer:
    """[디퍼런스 맵 시각화 전담 모듈 - Raw Difference 버전]
    에러 에너지의 방향성(누가 더 에러 에너지가 높은가)을 보존하기 위해
    제곱을 배제하고 양방향 발산형 컬러맵(coolwarm)을 사용합니다.
    """

    def __init__(self, core_processor: WaveletSpectrogramCore):
        self.core = core_processor

    def plot_and_save(self, save_dir=None):
        if save_dir is None: save_dir = WAVELET_IMGS_DIR / "difference_maps"
        save_path_obj = Path(save_dir)
        save_path_obj.mkdir(parents=True, exist_ok=True)

        base_tasks = ["Saccade A", "Saccade B", "Saccade B (anti)", "Saccade R"]
        columns_mapping = [(bt, eye) for bt in base_tasks for eye in ["Left", "Right"]]
        row_configs = ["Horizontal", "Vertical"]

        fig, axes = plt.subplots(nrows=len(row_configs), ncols=len(columns_mapping), figsize=(26, 7))
        fig.suptitle(
            f"CWT Difference Maps: (MCI Error dB - HC Error dB) ({self.core.min_freq:.0f}-{self.core.max_freq:.0f} Hz)",
            fontsize=22, fontweight="bold", y=1.05)

        for row_idx, axis_type in enumerate(row_configs):
            for col_idx, (base_task, eye) in enumerate(columns_mapping):
                ax = axes[row_idx, col_idx]
                full_task_name = f"{axis_type} {base_task}"

                hc_tensors = self.core.data_store["HC"].get(full_task_name, {}).get(eye, [])
                mci_tensors = self.core.data_store["MCI"].get(full_task_name, {}).get(eye, [])

                if not hc_tensors or not mci_tensors:
                    ax.axis("off");
                    continue

                hc_mean_db = 10 * np.log10(np.clip(np.mean(hc_tensors, axis=0), a_min=1e-10, a_max=None))
                mci_mean_db = 10 * np.log10(np.clip(np.mean(mci_tensors, axis=0), a_min=1e-10, a_max=None))

                # Raw Difference (Preserving directionality)
                diff_raw = mci_mean_db - hc_mean_db

                # Coolwarm with fixed range to emphasize directional divergence
                im = ax.imshow(diff_raw, aspect="auto", origin="lower", cmap="coolwarm", vmin=-10.0, vmax=10.0)

                # Event-Lock Line (t=0)
                time_bins = diff_raw.shape[1]
                t_zero_idx = int(time_bins * 0.2)
                ax.axvline(x=t_zero_idx, color='black', linestyle='--', alpha=0.6, linewidth=1.5)

                if row_idx == 0: ax.set_title(f"{base_task}\n({eye})", fontsize=11, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(f"{axis_type} Diff\n({self.core.min_freq:.0f}-{self.core.max_freq:.0f}Hz)",
                                  fontsize=14, fontweight="bold")
                else:
                    ax.set_yticks([])
                if row_idx == len(row_configs) - 1:
                    ax.set_xlabel("Time Bins")
                else:
                    ax.set_xticks([])

                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        output_file = save_path_obj / "Combined_CWT_Raw_Difference_Maps.png"
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[+] CWT Raw Difference Map 시각화 완료: {output_file}")


# ===========================================================================
# 5. ORCHESTRATOR
# ===========================================================================
def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CWT population dashboards (legacy consolidated).")
    # Default: RAW-magnitude cache (Option B). If absent, we'll build it from the CSVs.
    p.add_argument("--cache", type=Path, default=CACHE_DIR / "data_store_raw_cwt.pkl")
    p.add_argument("--skip-mean", action="store_true")
    p.add_argument("--skip-variance", action="store_true")
    p.add_argument("--skip-diff", action="store_true")
    return p.parse_args(argv)


def _ensure_raw_cache(cache_path: Path) -> None:
    """If the raw-magnitude cache doesn't exist yet, build it via
    `EventLockedCWTPipeline(signal_mode='raw_magnitude')`.
    """
    if cache_path.exists():
        return
    # Lazy import — only used when the cache needs building.
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from full_experiments_using.data_processor.data_engineering import EventLockedCWTPipeline
    from paths import DATA_DIR

    logger.info("Raw-magnitude cache not found, building it (this takes ~85s)…")
    pipeline = EventLockedCWTPipeline(
        pre_stimulus_sec=0.2, post_stimulus_sec=0.8,
        min_freq=15.0, max_freq=60.0,
        freq_bins=32, target_time_bins=32, w_morlet=4.0,
        artifact_threshold=30.0,
        cache_path=cache_path,
        signal_mode="raw_magnitude",
    )
    pipeline.process_directory(DATA_DIR)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args(argv)

    _ensure_raw_cache(args.cache)

    logger.info(f"CWT Core build — cache={args.cache}")
    core = WaveletSpectrogramCore(cwt_cache_path=args.cache)
    core.load()

    n_cells = sum(len(v) for g in core.data_store.values() for t in g.values() for v in t.values())
    logger.info(f"Core ready — {n_cells} (group, task, eye)-aggregated subject tensors total.")

    if not args.skip_mean:
        WaveletMeanVisualizer(core).plot_and_save()
    if not args.skip_variance:
        WaveletVarianceVisualizer(core).plot_and_save()
    if not args.skip_diff:
        WaveletDiffVisualizer(core).plot_and_save()

    return 0


if __name__ == "__main__":
    sys.exit(main())