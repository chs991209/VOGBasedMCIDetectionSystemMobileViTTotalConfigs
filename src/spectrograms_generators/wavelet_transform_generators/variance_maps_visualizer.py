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


if __name__ == "__main__":
    import logging
    from paths import CACHE_DIR

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("CWT 캐시에서 텐서 적재를 시작합니다...")
    core = WaveletSpectrogramCore(cwt_cache_path=CACHE_DIR / "data_store_full.pkl")
    core.load()

    logging.info("CWT 분산 맵(Variance Map) 렌더링을 시작합니다...")
    visualizer = WaveletVarianceVisualizer(core)
    visualizer.plot_and_save()
