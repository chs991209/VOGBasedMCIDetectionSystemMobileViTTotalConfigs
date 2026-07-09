import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paths import WAVELET_IMGS_DIR  # noqa: E402
from spectrograms_generators.wavelet_transform_generators.data_tensor_processor import (  # noqa: E402
    WaveletSpectrogramCore,
)


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


if __name__ == "__main__":
    import logging
    from paths import CACHE_DIR

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("CWT 캐시에서 텐서 적재를 시작합니다...")
    core = WaveletSpectrogramCore(cwt_cache_path=CACHE_DIR / "data_store_full.pkl")
    core.load()

    logging.info("CWT 디퍼런스 맵 렌더링을 시작합니다...")
    visualizer = WaveletDiffVisualizer(core)
    visualizer.plot_and_save()
