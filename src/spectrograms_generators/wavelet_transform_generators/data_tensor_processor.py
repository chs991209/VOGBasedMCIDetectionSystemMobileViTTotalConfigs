"""WaveletSpectrogramCore — consumes the project's CWT cache for cohort dashboards.

Unlike `HighFreqSpectrogramCore` (STFT), this Core does NOT process CSV files
itself. The Continuous-Wavelet-Transform tensors are already cached by
`full_experiments_using.data_processor.data_engineering.EventLockedCWTPipeline`
in `outputs/cache/data_store_full.pkl` (legacy 4-channel mode). We just load
that pickle, aggregate per-trial tensors per (subject, task) to a single mean
spectrogram per eye, and expose a `data_store[group][task][eye]` layout
identical to the STFT core so the same-shaped visualizers can consume it.
"""
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

logger = logging.getLogger(__name__)


# Channel indices in the legacy 4-channel CWT cache
_CH_LEFT  = 0   # mag_L : CWT magnitude of task-axis target-to-L error
_CH_RIGHT = 2   # mag_R : CWT magnitude of task-axis target-to-R error


class WaveletSpectrogramCore:
    """Aggregates the cached event-locked CWT tensors into a per-(group, task,
    eye) list of *per-subject* mean magnitude spectrograms.

    For each subject who has ≥ 1 trial of a given task, the subject's CWT
    magnitude for that task is collapsed to a single [F, T] tensor by averaging
    over the subject's trials. The resulting tensor is appended to
    `data_store[group][task_name][eye]`. Subjects with zero trials of a task
    are silently skipped for *that* task only (they may still contribute to
    other tasks).

    Tensor shape on disk: [4, freq_bins, time_bins] per trial. Channels 0 and
    2 carry the L/R magnitude maps; channels 1 and 3 hold the real part and
    are not used by the dashboard.
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
        self.pre_sec = None
        self.post_sec = None

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
        self.pre_sec = float(cfg.get("pre_sec", 0.2))
        self.post_sec = float(cfg.get("post_sec", 0.8))

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
