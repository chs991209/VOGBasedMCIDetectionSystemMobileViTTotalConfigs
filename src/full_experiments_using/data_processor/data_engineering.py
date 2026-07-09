"""8-experiment CWT pipeline + dataset.

Same physics as the 2-experiment pipeline; preserves the original 8-task
filter and `is_anti` conditional target inversion. Adds the project's
cache mechanism (separate file from the 2-experiment cache so neither
overwrites the other) and an AugmentedSubset for train-only SpecAugment.
"""

import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pywt
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)


class EventLockedCWTPipeline:
    def __init__(
        self,
        pre_stimulus_sec: float = 0.2,
        post_stimulus_sec: float = 0.8,
        min_freq: float = 15.0,
        max_freq: float = 60.0,
        freq_bins: int = 32,
        target_time_bins: int = 32,
        w_morlet: float = 4.0,
        artifact_threshold: float = 45.0,
        default_fs: float = 120.0,
        cache_path: Optional[Union[str, Path]] = None,
        signal_mode: str = "legacy",
    ):
        # signal_mode:
        #   "legacy"        — task-axis only; channels = {mag_L, re_L, mag_R, re_R}
        #                     with sparsify (85th-pctile floor) + z-score per trial.
        #   "four_error"    — both axes from same CSV; channels = CWT magnitudes of
        #                     {LH-TH, RH-TH, LV-TV, RV-TV}. Used by the meta pipeline.
        #   "raw_magnitude" — task-axis only; channels = {mag_L_raw, re_L, mag_R_raw,
        #                     re_R}. Raw |CWT|, NO sparsify, NO z-score. Visualization-
        #                     only mode so dashboards see the true linear magnitude.
        if signal_mode not in ("legacy", "four_error", "raw_magnitude"):
            raise ValueError(
                f"signal_mode must be 'legacy', 'four_error', or 'raw_magnitude', got {signal_mode!r}"
            )
        self.signal_mode = signal_mode
        self.pre_sec = pre_stimulus_sec
        self.post_sec = post_stimulus_sec
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.freq_bins = freq_bins
        self.time_bins = target_time_bins
        self.w = w_morlet
        self.artifact_threshold = artifact_threshold
        self.default_fs = default_fs

        self.frequencies = np.logspace(np.log10(self.min_freq), np.log10(self.max_freq), self.freq_bins)
        self.wavelet_name = f"cmor{self.w}-1.0"
        self._wavelet_central_freq = pywt.central_frequency(self.wavelet_name)

        # 8-task mapping (per README task ID table).
        self.task_map = {
            "Horizontal Saccade A": 0,
            "Horizontal Saccade B": 1,
            "Horizontal Saccade B (anti)": 2,
            "Horizontal Saccade R": 3,
            "Vertical Saccade A": 4,
            "Vertical Saccade B": 5,
            "Vertical Saccade B (anti)": 6,
            "Vertical Saccade R": 7,
        }

        self.data_store = defaultdict(lambda: defaultdict(list))
        self.cache_path = Path(cache_path) if cache_path is not None else None

    def _config_signature(self) -> dict:
        return {
            "pre_sec": self.pre_sec,
            "post_sec": self.post_sec,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "freq_bins": self.freq_bins,
            "time_bins": self.time_bins,
            "w": self.w,
            "artifact_threshold": self.artifact_threshold,
            "task_map": dict(self.task_map),
            "signal_mode": self.signal_mode,
            "schema_version": 2 if self.signal_mode == "four_error" else 1,
        }

    def _load_csv_safely(self, file_path: Path):
        try:
            df = pd.read_csv(file_path, skipinitialspace=True)
            df.columns = [str(c).strip().lower() for c in df.columns]
            if any('lh' in c for c in df.columns):
                return df.apply(pd.to_numeric, errors='coerce').dropna(how='all').reset_index(drop=True)
        except Exception:
            pass
        for enc in ['utf-16', 'utf-16le', 'utf-8-sig', 'cp949']:
            try:
                with open(file_path, 'r', encoding=enc, errors='replace') as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    if 'lh' in line.lower() and 'rh' in line.lower():
                        cols = [c.replace('\x00', '').strip().lower() for c in line.split(',')]
                        parsed = [
                            [v.strip() for v in l.replace('\x00', '').split(',')]
                            for l in lines[i + 1:] if l.strip()
                        ]
                        df = pd.DataFrame(parsed, columns=cols)
                        return df.apply(pd.to_numeric, errors='coerce').dropna(how='all').reset_index(drop=True)
            except Exception:
                continue
        raise ValueError(f"Failed to load {file_path.name}")

    def _get_cwt_tensor(self, error_sig, fs):
        if len(error_sig) == 0:
            return None
        sampling_period = 1.0 / fs
        scales = self._wavelet_central_freq / (self.frequencies * sampling_period)

        cwtm, _ = pywt.cwt(error_sig, scales, self.wavelet_name, sampling_period=sampling_period)

        time_zoom = self.time_bins / cwtm.shape[1]
        # Nearest-neighbor (order=0) was the original 8-exp choice.
        cwt_real = zoom(np.real(cwtm), (1.0, time_zoom), mode='nearest', order=0)
        cwt_imag = zoom(np.imag(cwtm), (1.0, time_zoom), mode='nearest', order=0)
        return cwt_real, cwt_imag

    def _sparsify_and_compress(self, cwt_real, cwt_imag):
        magnitude = np.sqrt(cwt_real ** 2 + cwt_imag ** 2)
        threshold = np.percentile(magnitude, 85)
        magnitude[magnitude < threshold] = 1e-3
        magnitude[magnitude == 0] = 1e-3
        mag_db = 10 * np.log10(magnitude)
        mag_db = (mag_db - np.mean(mag_db)) / (np.std(mag_db) + 1e-8)
        return mag_db

    def _try_load_cache(self) -> bool:
        if self.cache_path is None or not self.cache_path.exists():
            return False
        try:
            with open(self.cache_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning("Cache read failed (%s); reprocessing CSVs.", e)
            return False

        if payload.get("config") != self._config_signature():
            logger.info("Cache config mismatch; reprocessing CSVs.")
            return False

        plain = payload.get("data_store", {})
        self.data_store = defaultdict(lambda: defaultdict(list))
        for group, subjects in plain.items():
            for subject_id, epochs in subjects.items():
                self.data_store[group][subject_id] = list(epochs)
        n = sum(len(eps) for s in self.data_store.values() for eps in s.values())
        logger.info("Loaded preprocessed data_store from cache: %s (%d epochs)", self.cache_path, n)
        return True

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        plain = {
            group: {sid: list(epochs) for sid, epochs in subjects.items()}
            for group, subjects in self.data_store.items()
        }
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump({"config": self._config_signature(), "data_store": plain}, f)
            tmp.replace(self.cache_path)
            logger.info("Cached preprocessed data_store to: %s", self.cache_path)
        except Exception as e:
            logger.warning("Cache write failed: %s", e)

    def _process_file_legacy(self, df, axis_char, is_anti, group, subject_id, task_id) -> int:
        target_col = next(
            (c for c in df.columns if f'target{axis_char}' in c or f'target_{axis_char}' in c),
            None,
        )
        l_col = next((c for c in df.columns if c == f'l{axis_char}'), None)
        r_col = next((c for c in df.columns if c == f'r{axis_char}'), None)

        if not target_col or not l_col or not r_col:
            return -1   # signal columns missing
        if is_anti:
            df[target_col] = df[target_col] * -1

        time_col = next((c for c in df.columns if 'time' in c or c == 't'), df.columns[0])
        time_val = df[time_col].dropna().values
        fs = 1.0 / np.mean(np.diff(time_val)) if len(time_val) > 1 else self.default_fs

        target_val = df[target_col].fillna(0).values
        l_val = df[l_col].fillna(0).values
        r_val = df[r_col].fillna(0).values

        event_indices = np.where(np.diff(target_val, prepend=0) != 0)[0]
        samples_pre = int(self.pre_sec * fs)
        samples_post = int(self.post_sec * fs)

        kept_here = 0
        for idx in event_indices:
            s, e = idx - samples_pre, idx + samples_post
            if s < 0 or e > len(df):
                continue

            err_L = target_val[s:e] - l_val[s:e]
            err_R = target_val[s:e] - r_val[s:e]
            err_L -= np.mean(err_L[:samples_pre])
            err_R -= np.mean(err_R[:samples_pre])

            if (np.max(np.abs(err_L)) > self.artifact_threshold
                    or np.max(np.abs(err_R)) > self.artifact_threshold):
                continue

            re_L, im_L = self._get_cwt_tensor(err_L, fs)
            re_R, im_R = self._get_cwt_tensor(err_R, fs)

            mag_L = self._sparsify_and_compress(re_L, im_L)
            mag_R = self._sparsify_and_compress(re_R, im_R)

            tensor = np.stack([mag_L, re_L, mag_R, re_R], axis=0)
            self.data_store[group][subject_id].append((tensor, task_id))
            kept_here += 1
        return kept_here

    def _process_file_legacy_raw(self, df, axis_char, is_anti, group, subject_id, task_id) -> int:
        """Same task-axis L/R event extraction as `_process_file_legacy`, but
        stores RAW CWT magnitude (|sqrt(re² + im²)|) — no sparsify, no z-score.

        Channels = {mag_L_raw, re_L, mag_R_raw, re_R}. Intended for downstream
        visualization where the second 10·log10 actually has dB semantics.
        """
        target_col = next(
            (c for c in df.columns if f'target{axis_char}' in c or f'target_{axis_char}' in c),
            None,
        )
        l_col = next((c for c in df.columns if c == f'l{axis_char}'), None)
        r_col = next((c for c in df.columns if c == f'r{axis_char}'), None)

        if not target_col or not l_col or not r_col:
            return -1
        if is_anti:
            df[target_col] = df[target_col] * -1

        time_col = next((c for c in df.columns if 'time' in c or c == 't'), df.columns[0])
        time_val = df[time_col].dropna().values
        fs = 1.0 / np.mean(np.diff(time_val)) if len(time_val) > 1 else self.default_fs

        target_val = df[target_col].fillna(0).values
        l_val = df[l_col].fillna(0).values
        r_val = df[r_col].fillna(0).values

        event_indices = np.where(np.diff(target_val, prepend=0) != 0)[0]
        samples_pre = int(self.pre_sec * fs)
        samples_post = int(self.post_sec * fs)

        kept_here = 0
        for idx in event_indices:
            s, e = idx - samples_pre, idx + samples_post
            if s < 0 or e > len(df):
                continue

            err_L = target_val[s:e] - l_val[s:e]
            err_R = target_val[s:e] - r_val[s:e]
            err_L -= np.mean(err_L[:samples_pre])
            err_R -= np.mean(err_R[:samples_pre])

            if (np.max(np.abs(err_L)) > self.artifact_threshold
                    or np.max(np.abs(err_R)) > self.artifact_threshold):
                continue

            re_L, im_L = self._get_cwt_tensor(err_L, fs)
            re_R, im_R = self._get_cwt_tensor(err_R, fs)

            # RAW magnitudes — no sparsify, no z-score
            mag_L_raw = np.sqrt(re_L ** 2 + im_L ** 2)
            mag_R_raw = np.sqrt(re_R ** 2 + im_R ** 2)

            tensor = np.stack([mag_L_raw, re_L, mag_R_raw, re_R], axis=0)
            self.data_store[group][subject_id].append((tensor, task_id))
            kept_here += 1
        return kept_here

    def _process_file_four_error(self, df, axis_char, is_anti, group, subject_id, task_id):
        """4-channel CWT magnitudes of {LH-TH, RH-TH, LV-TV, RV-TV} per event.

        Requires BOTH H and V axis columns to be present in the same CSV. Files
        with either axis missing are skipped at the file level (returns (0, True)).
        is_anti only flips the *task's* target axis. Event detection runs on the
        task's target. Artifact rejection uses the task axis's L/R errors (same
        threshold as legacy).
        """
        # Task-axis columns (used for event detection + artifact rule)
        target_tax_col = next(
            (c for c in df.columns if f'target{axis_char}' in c or f'target_{axis_char}' in c),
            None,
        )
        l_tax_col = next((c for c in df.columns if c == f'l{axis_char}'), None)
        r_tax_col = next((c for c in df.columns if c == f'r{axis_char}'), None)

        # Other-axis columns
        other_axis = 'v' if axis_char == 'h' else 'h'
        target_oax_col = next(
            (c for c in df.columns if f'target{other_axis}' in c or f'target_{other_axis}' in c),
            None,
        )
        l_oax_col = next((c for c in df.columns if c == f'l{other_axis}'), None)
        r_oax_col = next((c for c in df.columns if c == f'r{other_axis}'), None)

        if not all([target_tax_col, l_tax_col, r_tax_col,
                    target_oax_col, l_oax_col, r_oax_col]):
            return 0, True   # missing_cols → file-level skip

        if is_anti:
            df[target_tax_col] = df[target_tax_col] * -1

        time_col = next((c for c in df.columns if 'time' in c or c == 't'), df.columns[0])
        time_val = df[time_col].dropna().values
        fs = 1.0 / np.mean(np.diff(time_val)) if len(time_val) > 1 else self.default_fs

        target_tax = df[target_tax_col].fillna(0).values
        l_tax = df[l_tax_col].fillna(0).values
        r_tax = df[r_tax_col].fillna(0).values
        target_oax = df[target_oax_col].fillna(0).values
        l_oax = df[l_oax_col].fillna(0).values
        r_oax = df[r_oax_col].fillna(0).values

        # Map task/other → H/V signed by directive convention (eye - target)
        if axis_char == 'h':
            lh, rh, th = l_tax, r_tax, target_tax
            lv, rv, tv = l_oax, r_oax, target_oax
        else:
            lh, rh, th = l_oax, r_oax, target_oax
            lv, rv, tv = l_tax, r_tax, target_tax

        event_indices = np.where(np.diff(target_tax, prepend=0) != 0)[0]
        samples_pre = int(self.pre_sec * fs)
        samples_post = int(self.post_sec * fs)

        kept_here = 0
        for idx in event_indices:
            s, e = idx - samples_pre, idx + samples_post
            if s < 0 or e > len(df):
                continue

            err_LH = (lh[s:e] - th[s:e]).astype(np.float64)
            err_RH = (rh[s:e] - th[s:e]).astype(np.float64)
            err_LV = (lv[s:e] - tv[s:e]).astype(np.float64)
            err_RV = (rv[s:e] - tv[s:e]).astype(np.float64)

            for arr in (err_LH, err_RH, err_LV, err_RV):
                arr -= np.mean(arr[:samples_pre])

            # Artifact rejection on the task-axis L/R errors only (parity rule)
            if axis_char == 'h':
                task_l_err, task_r_err = err_LH, err_RH
            else:
                task_l_err, task_r_err = err_LV, err_RV
            if (np.max(np.abs(task_l_err)) > self.artifact_threshold
                    or np.max(np.abs(task_r_err)) > self.artifact_threshold):
                continue

            # Compute raw |CWT| magnitudes ONCE, so the cross-axis ratio is
            # measured BEFORE per-channel z-scoring (which erases cross-channel
            # magnitude comparisons).
            raw_mags = []
            channels = []
            for err in (err_LH, err_RH, err_LV, err_RV):
                re_c, im_c = self._get_cwt_tensor(err, fs)
                raw_mag = np.sqrt(re_c ** 2 + im_c ** 2)
                raw_mags.append(raw_mag)
                channels.append(self._sparsify_and_compress(re_c, im_c))
            tensor = np.stack(channels, axis=0)            # [4, F, T] z-scored

            # Cross-axis energy ratio: mean magnitude in OFF-axis channels divided
            # by mean magnitude in TASK-AXIS channels (raw, pre-normalization).
            # H tasks (id 0-3): task axis = H (ch0+ch1 raw), cross = V (ch2+ch3 raw).
            # V tasks (id 4-7): swap.
            if axis_char == 'h':
                active_e = (raw_mags[0].mean() + raw_mags[1].mean()) * 0.5
                cross_e  = (raw_mags[2].mean() + raw_mags[3].mean()) * 0.5
            else:
                active_e = (raw_mags[2].mean() + raw_mags[3].mean()) * 0.5
                cross_e  = (raw_mags[0].mean() + raw_mags[1].mean()) * 0.5
            ratio = float(cross_e / (active_e + 1e-8))     # scalar

            self.data_store[group][subject_id].append((tensor, task_id, ratio))
            kept_here += 1
        return kept_here, False

    def process_directory(self, base_dir: Path):
        if self._try_load_cache():
            return

        csv_files = list(base_dir.rglob('*.csv'))
        logger.info("Processing %d CSV files under %s", len(csv_files), base_dir)
        n_kept = 0
        n_skipped_task = 0
        n_skipped_group = 0
        n_skipped_cols = 0
        n_load_errors = 0

        for filepath in csv_files:
            clean_task = filepath.stem.replace("PD VOG -_", "").replace("PD VOG -", "").strip()
            axis_char = 'h' if 'Horizontal' in clean_task else 'v' if 'Vertical' in clean_task else None
            if not axis_char or clean_task not in self.task_map:
                n_skipped_task += 1
                continue

            task_id = self.task_map[clean_task]
            group = "HC" if "HC" in str(filepath).upper() else "MCI" if "MCI" in str(filepath).upper() else None
            if not group:
                n_skipped_group += 1
                continue

            subject_id = filepath.parent.name

            try:
                df = self._load_csv_safely(filepath)
                is_anti = "anti" in clean_task.lower()

                if self.signal_mode == "legacy":
                    kept_here = self._process_file_legacy(
                        df, axis_char, is_anti, group, subject_id, task_id,
                    )
                elif self.signal_mode == "raw_magnitude":
                    kept_here = self._process_file_legacy_raw(
                        df, axis_char, is_anti, group, subject_id, task_id,
                    )
                else:  # four_error
                    kept_here, missing_cols = self._process_file_four_error(
                        df, axis_char, is_anti, group, subject_id, task_id,
                    )
                    if missing_cols:
                        n_skipped_cols += 1
                        continue

                if kept_here < 0:  # signal: columns missing in legacy branch
                    n_skipped_cols += 1
                    continue
                n_kept += kept_here
            except Exception as e:
                n_load_errors += 1
                logger.debug("Skipping %s: %s", filepath.name, e)

        logger.info(
            "Pipeline summary | kept_epochs=%d | files_skipped(task=%d, group=%d, missing_cols=%d, load_errors=%d)",
            n_kept, n_skipped_task, n_skipped_group, n_skipped_cols, n_load_errors,
        )
        self._save_cache()


class TaskConditionedDataset(Dataset):
    def __init__(self, data_store):
        self.X = []
        self.T = []
        self.y = []
        self.subject_ids = []

        for group, subjects in data_store.items():
            label = 0 if group == "HC" else 1
            for subject_id, epochs in subjects.items():
                for tensor, task_id in epochs:
                    self.X.append(tensor)
                    self.T.append(task_id)
                    self.y.append(label)
                    self.subject_ids.append(subject_id)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.T = torch.tensor(self.T, dtype=torch.long)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.y[idx]


class AugmentedSubset(Subset):
    """SpecAugment-style wrapper for the training Subset (8-exp variant)."""

    def __init__(
        self,
        dataset,
        indices,
        freq_mask_max: int = 8,
        time_mask_max: int = 8,
        freq_mask_p: float = 0.5,
        time_mask_p: float = 0.5,
    ):
        super().__init__(dataset, indices)
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.freq_mask_p = freq_mask_p
        self.time_mask_p = time_mask_p

        if hasattr(dataset, "y"):
            self.y = dataset.y[torch.as_tensor(indices, dtype=torch.long)]

    def _freq_mask(self, x: torch.Tensor) -> torch.Tensor:
        F = x.shape[1]
        f = int(torch.randint(1, self.freq_mask_max + 1, (1,)).item())
        f = min(f, F)
        f0 = int(torch.randint(0, F - f + 1, (1,)).item())
        x = x.clone()
        x[:, f0:f0 + f, :] = 0.0
        return x

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[2]
        t = int(torch.randint(1, self.time_mask_max + 1, (1,)).item())
        t = min(t, T)
        t0 = int(torch.randint(0, T - t + 1, (1,)).item())
        x = x.clone()
        x[:, :, t0:t0 + t] = 0.0
        return x

    def __getitem__(self, idx):
        x, task_id, label = super().__getitem__(idx)
        if torch.rand(1).item() < self.freq_mask_p:
            x = self._freq_mask(x)
        if torch.rand(1).item() < self.time_mask_p:
            x = self._time_mask(x)
        return x, task_id, label

    def __getitems__(self, indices):
        return [self.__getitem__(i) for i in indices]
