"""Single orchestrator for the three CWT population dashboards.

Loads `HighFreqSpectrogramCore`'s wavelet-equivalent — the legacy 4-channel CWT
cache built by `EventLockedCWTPipeline` — once, and runs the mean / variance /
squared-difference visualizers in sequence. Three PNGs are written to:

    imgs/wavelet_transformed_spectrograms/
        mean_spectrograms/Combined_32_Panels_CWT_Means.png
        variance_maps/Combined_CWT_Variance_Maps.png
        squared_difference_maps/Combined_CWT_Difference_Maps.png

These coexist with the existing per-trial subfolders (`HC/<sid>/task…/trialNN.png`)
under the same root — the dashboard PNGs live in distinct map_type sub-folders.

CLI:
    python src/spectrograms_generators/wavelet_transform_generators/generator.py
"""
import argparse
import logging
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paths import CACHE_DIR, WAVELET_IMGS_DIR  # noqa: E402
from spectrograms_generators.wavelet_transform_generators.data_tensor_processor import (  # noqa: E402
    WaveletSpectrogramCore,
)
from spectrograms_generators.wavelet_transform_generators.mean_spectrograms_visualizer import (  # noqa: E402
    WaveletMeanVisualizer,
)
from spectrograms_generators.wavelet_transform_generators.variance_maps_visualizer import (  # noqa: E402
    WaveletVarianceVisualizer,
)
from spectrograms_generators.wavelet_transform_generators.squared_difference_maps_visualizer import (  # noqa: E402
    WaveletDiffVisualizer,
)

logger = logging.getLogger("cwt_generator")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CWT population dashboards.")
    p.add_argument("--cache", type=Path,
                   default=CACHE_DIR / "data_store_full.pkl",
                   help="CWT cache produced by EventLockedCWTPipeline (legacy 4-channel mode). "
                        "Default = outputs/cache/data_store_full.pkl")
    p.add_argument("--skip-mean", action="store_true")
    p.add_argument("--skip-variance", action="store_true")
    p.add_argument("--skip-diff", action="store_true")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

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
