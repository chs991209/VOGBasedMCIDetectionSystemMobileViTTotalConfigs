"""Per-trial saccade kinematic features (10 indicators — see FEATURE_NAMES).

For the SAME event-locked, artifact-passing windows the four_error system uses
(threshold gate, [-0.2s, +0.8s], 37-subject cohort with New_data excluded), this
extracts 10 saccade metrics per trial. Most are computed from the task-axis eye
position (mean of L, R), baseline-subtracted over the pre-event samples.

`latency` follows the research-admin's 4-step recipe (see FEATURE_NAMES / _lowpass):
target onset (TargetH/V step) → differentiate position to velocity → per-eye 2D
total speed sqrt(vH^2 + vV^2) → first crossing of VEL_THRESHOLD after onset, with a
low-pass filter smoothing the position first so sensor jitter can't fake the onset.
"""
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from four_error_using.data_processor.data_engineer import (
    EventLockedCWTPipeline, _EXCLUDED_DIR_NAMES,
)
from four_error_using.data_processor.saccade_metrics import (
    FEATURE_NAMES, FEATURE_VERSION, VEL_THRESHOLD, lowpass, total_speed_2d, window_metrics,
)

logger = logging.getLogger(__name__)

# Full saccade metric set per trial (task-axis eye position within the window):
#   latency          — reaction time, research-admin's 4-step method (120 Hz position data):
#                        1. target onset = the sample where TargetH/V steps (0 -> 15)
#                        2. position -> velocity by differentiation (dt = 1/fs)
#                        3. per eye, combine H & V velocity via Pythagoras -> 2D total speed
#                        4. eye onset = first sample AFTER target onset where the smoothed
#                           2D total speed crosses VEL_THRESHOLD; latency = eye - target onset
#                        (a low-pass filter smooths position first — see _lowpass)
#   peak_velocity    — max |dP/dt| on the smoothed task-axis position (deg/s)
#   peak_accel       — max d(speed)/dt = fastest speeding-up (deg/s^2, direction-robust)
#   peak_decel       — min d(speed)/dt = fastest slowing-down (deg/s^2, direction-robust)
#   amplitude        — |final eye displacement| (deg)
#   duration         — time |velocity| stays above threshold (s)
#   time_to_peak_vel — onset → peak velocity (s)
#   time_to_target   — onset → eye first reaches 90% of its final displacement (s)
#   gain             — final eye displacement / target step (accuracy; 1 = on target)
#   vigor            — peak_velocity / amplitude (vigour proxy; NOT the population main sequence)
# All velocity/acceleration features differentiate a LOW-PASS-SMOOTHED position.
# FEATURE_NAMES / VEL_THRESHOLD / FEATURE_VERSION and the metric math live in
# saccade_metrics.py (shared with data_engineer's kinematics-in-tensor path).


class KinematicFeaturePipeline:
    def __init__(self, artifact_threshold: float = 30.0, cache_path: Path | None = None):
        self.thr = float(artifact_threshold)
        self.cache_path = Path(cache_path) if cache_path else None
        # borrow the CWT pipeline only for its robust CSV loader, task_map, and
        # window config, so trial selection is identical to the four_error system.
        self._io = EventLockedCWTPipeline(signal_mode="four_error",
                                          artifact_threshold=self.thr, cache_path=None)
        self.task_map = self._io.task_map
        self.pre_sec, self.post_sec = self._io.pre_sec, self._io.post_sec
        self.default_fs = self._io.default_fs
        self.data_store = defaultdict(lambda: defaultdict(list))

    # ------------------------------------------------------------------ #
    def _features_for_file(self, df, axis_char, is_anti, group, subject_id, task_id):
        """Extract the 10 saccade numbers per trial from one CSV (same trials as the CWT system)."""
        tcol = next((c for c in df.columns if f'target{axis_char}' in c or f'target_{axis_char}' in c), None)
        lcol = next((c for c in df.columns if c == f'l{axis_char}'), None)
        rcol = next((c for c in df.columns if c == f'r{axis_char}'), None)
        if not all([tcol, lcol, rcol]):
            return 0
        if is_anti:
            df[tcol] = df[tcol] * -1
        time_col = next((c for c in df.columns if 'time' in c or c == 't'), df.columns[0])
        tv = df[time_col].dropna().values
        fs = 1.0 / np.mean(np.diff(tv)) if len(tv) > 1 else self.default_fs

        target = df[tcol].fillna(0).values.astype(np.float64)
        l = df[lcol].fillna(0).values.astype(np.float64)
        r = df[rcol].fillna(0).values.astype(np.float64)
        spre, spost = int(self.pre_sec * fs), int(self.post_sec * fs)

        # --- 2D total eye speed for the latency (research-admin's method) ---
        # Both axes are present in every CSV, so build one smoothed 2D total-speed
        # signal (both eyes) used to detect the saccade onset for `latency`.
        def _col(name):
            return next((c for c in df.columns if c == name), None)
        lh_c, rh_c, lv_c, rv_c = _col('lh'), _col('rh'), _col('lv'), _col('rv')
        v_total_full = None
        if all([lh_c, rh_c, lv_c, rv_c]):
            v_total_full = total_speed_2d(
                df[lh_c].fillna(0).values.astype(np.float64),
                df[rh_c].fillna(0).values.astype(np.float64),
                df[lv_c].fillna(0).values.astype(np.float64),
                df[rv_c].fillna(0).values.astype(np.float64), fs)

        # Low-pass the task-axis position ONCE over the whole recording (then slice
        # per window), so every velocity/acceleration feature differentiates a
        # smoothed signal — no per-window edge artifacts, no jitter spikes.
        p_tax_smooth = lowpass(0.5 * (l + r), fs)

        kept = 0
        for idx in np.where(np.diff(target, prepend=0) != 0)[0]:
            s, e = idx - spre, idx + spost
            if s < 0 or e > len(df):
                continue
            # artifact rejection — identical rule to four_error (task-axis L/R error)
            eL = l[s:e] - target[s:e]; eR = r[s:e] - target[s:e]
            eL -= eL[:spre].mean(); eR -= eR[:spre].mean()
            if max(np.max(np.abs(eL)), np.max(np.abs(eR))) > self.thr:
                continue
            # smoothed task-axis eye position, baseline-subtracted to pre-event
            p = p_tax_smooth[s:e] - p_tax_smooth[s:s + spre].mean()
            v_win = v_total_full[s:e] if v_total_full is not None else None
            feats = window_metrics(p, target[s:e], v_win, spre, fs)
            self.data_store[group][subject_id].append((feats, task_id))
            kept += 1
        return kept

    def process_directory(self, base_dir: Path):
        """Walk every subject CSV (New_data excluded) → kinematic vectors; cache the result."""
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                payload = pickle.load(f)
            if (payload.get("thr") == self.thr and payload.get("features") == FEATURE_NAMES
                    and payload.get("version") == FEATURE_VERSION):
                self.data_store = payload["data_store"]
                logger.info("Loaded kinematic cache: %s", self.cache_path)
                return
        csv_files = [p for p in base_dir.rglob('*.csv')
                     if not (_EXCLUDED_DIR_NAMES & set(p.parts))]
        logger.info("Kinematics: processing %d CSV files under %s (excluded %s)",
                    len(csv_files), base_dir, sorted(_EXCLUDED_DIR_NAMES))
        n = 0
        for fp in csv_files:
            clean = fp.stem.replace("PD VOG -_", "").replace("PD VOG -", "").strip()
            axis = 'h' if 'Horizontal' in clean else 'v' if 'Vertical' in clean else None
            if not axis or clean not in self.task_map:
                continue
            group = "HC" if "HC" in str(fp).upper() else "MCI" if "MCI" in str(fp).upper() else None
            if not group:
                continue
            try:
                df = self._io._load_csv_safely(fp)
                n += self._features_for_file(df, axis, "anti" in clean.lower(),
                                             group, fp.parent.name, self.task_map[clean])
            except Exception as ex:
                logger.debug("skip %s: %s", fp.name, ex)
        logger.info("Kinematics: %d trials across %d subjects",
                    n, sum(len(v) for v in self.data_store.values()))
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump({"thr": self.thr, "features": FEATURE_NAMES,
                             "version": FEATURE_VERSION,
                             "data_store": dict((g, dict(s)) for g, s in self.data_store.items())}, f)
