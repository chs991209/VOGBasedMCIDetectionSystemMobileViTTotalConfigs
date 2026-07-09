"""Single orchestrator for the three STFT population dashboards.

Builds `HighFreqSpectrogramCore` once (with on-disk cache), then runs the
mean / variance / squared-difference visualizers. Three PNGs are written to:

    imgs/short_term_fourier_transformed_spectrograms/
        mean_spectrograms/Combined_32_Panels_HighFreq_Means.png
        variance_maps/Combined_HighFreq_Variance_Maps.png
        squared_difference_maps/Combined_HighFreq_Difference_Maps.png

CLI:
    python src/spectrograms_generators/short_term_fourier_transform_generators/generator.py
"""
import argparse
import logging
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paths import CACHE_DIR, DATA_DIR, STFT_IMGS_DIR  # noqa: E402
from spectrograms_generators.short_term_fourier_transform_generators.data_tensor_processor import (  # noqa: E402
    HighFreqSpectrogramCore,
)
from spectrograms_generators.short_term_fourier_transform_generators.mean_spectrograms_visualizer import (  # noqa: E402
    HighFreqMeanVisualizer,
)
from spectrograms_generators.short_term_fourier_transform_generators.variance_maps_visualizer import (  # noqa: E402
    HighFreqVarianceVisualizer,
)
from spectrograms_generators.short_term_fourier_transform_generators.squared_difference_maps_visualizer import (  # noqa: E402
    HighFreqDiffVisualizer,
)

logger = logging.getLogger("stft_generator")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STFT population dashboards.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--cache", type=Path, default=CACHE_DIR / "stft_data_store.pkl")
    p.add_argument("--target-fs", type=float, default=120.0)
    p.add_argument("--max-freq", type=float, default=60.0)
    p.add_argument("--nperseg", type=int, default=64)
    p.add_argument("--noverlap", type=int, default=24)
    p.add_argument("--skip-mean", action="store_true")
    p.add_argument("--skip-variance", action="store_true")
    p.add_argument("--skip-diff", action="store_true")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

    logger.info("STFT Core build — data=%s  cache=%s", args.data_dir, args.cache)
    core = HighFreqSpectrogramCore(
        target_fs=args.target_fs, max_freq=args.max_freq,
        nperseg=args.nperseg, noverlap=args.noverlap,
        cache_path=args.cache,
    )
    core.process_directory(args.data_dir)

    n_tensors = sum(len(v) for g in core.data_store.values()
                          for t in g.values()
                          for v in t.values())
    logger.info("Core ready — %d STFT tensors in data_store.", n_tensors)
    logger.info("Output root: %s", STFT_IMGS_DIR)

    if not args.skip_mean:
        HighFreqMeanVisualizer(core).plot_and_save()
    if not args.skip_variance:
        HighFreqVarianceVisualizer(core).plot_and_save()
    if not args.skip_diff:
        HighFreqDiffVisualizer(core).plot_and_save()

    return 0


if __name__ == "__main__":
    sys.exit(main())
