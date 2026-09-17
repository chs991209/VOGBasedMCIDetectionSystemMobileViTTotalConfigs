"""Shared per-trial saccade metric computation.

One place for the 10 kinematic numbers so every caller computes them identically:
 - kinematic_features.py       (standalone kinematics experiment, logistic gate)
 - data_engineer.py            (join the numbers INTO the image tensor, before the head)

`latency` uses the research-admin's 4-step method: target onset (t=0 at sample `spre`)
-> first crossing of VEL_THRESHOLD by the smoothed 2D total eye speed. The 2D speed is
built by the caller (both eyes, H&V via Pythagoras) and passed in as `v_total_window`.
"""
import numpy as np
from scipy.signal import butter, filtfilt

# 10 saccade metrics per trial, computed from the LOW-PASS-SMOOTHED task-axis eye
# position (except `latency`, which uses the 2D total eye speed). Order is fixed.
#   peak_accel / peak_decel are the max / min of d(speed)/dt — the peak rate the eye
#     SPEEDS UP / SLOWS DOWN. Defined on speed (|v|), so they are direction-robust
#     (a leftward saccade no longer swaps accel and decel).
#   vigor = peak_velocity / amplitude (deg/s per deg). A per-trial vigour proxy — NOT
#     the population "main sequence" relationship, so named honestly.
FEATURE_NAMES = ["latency", "peak_velocity", "peak_accel", "peak_decel", "amplitude",
                 "duration", "time_to_peak_vel", "time_to_target", "gain", "vigor"]
VEL_THRESHOLD = 30.0  # deg/s, standard saccade-onset velocity threshold
FEATURE_VERSION = 3   # v2: latency=2D onset; v3: +lowpass all vel/accel, dir-robust accel/decel, vigor


def lowpass(sig, fs, cutoff=40.0, order=2):
    """Zero-phase Butterworth low-pass, applied to position BEFORE differentiation.

    Differentiating raw position amplifies sensor jitter / eye micro-tremor into huge
    velocity spikes that can be mistaken for a saccade onset. Smoothing first removes
    them (research-admin requirement). Falls back to a short moving average when the
    signal is too short for the filter."""
    sig = np.asarray(sig, dtype=np.float64)
    nyq = 0.5 * fs
    wn = cutoff / nyq
    if len(sig) <= 12 or not (0.0 < wn < 1.0):
        k = min(5, len(sig))
        return sig if k < 2 else np.convolve(sig, np.ones(k) / k, mode="same")
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, sig)


def total_speed_2d(lh, rh, lv, rv, fs):
    """Smoothed 2D total eye speed for a whole recording: per eye sqrt(vH^2 + vV^2),
    then the mean of both eyes. Used to detect the saccade onset for `latency`."""
    vlh = np.gradient(lowpass(lh, fs)) * fs
    vrh = np.gradient(lowpass(rh, fs)) * fs
    vlv = np.gradient(lowpass(lv, fs)) * fs
    vrv = np.gradient(lowpass(rv, fs)) * fs
    return 0.5 * (np.hypot(vlh, vlv) + np.hypot(vrh, vrv))


def window_metrics(p, tgt, v_total_window, spre, fs):
    """The 10 saccade numbers for one event window.

    p              : LOW-PASS-SMOOTHED task-axis eye position over the window,
                     baseline-subtracted (len n). The caller smooths the whole
                     recording before slicing, so differentiation here is noise-safe.
    tgt            : task-axis target over the window (len n)
    v_total_window : smoothed 2D total eye speed over the SAME window, or None to fall
                     back to the task-axis speed for the latency onset
    spre           : index of the target onset within the window (t=0)
    Returns a float32 array in FEATURE_NAMES order.
    """
    p = np.asarray(p, dtype=np.float64)
    v = np.gradient(p) * fs                          # task-axis velocity, deg/s
    speed = np.abs(v)                                 # unsigned speed, deg/s
    accel_of_speed = np.gradient(speed) * fs          # d(speed)/dt, deg/s^2
    n = len(p)

    peak_velocity = float(np.max(speed))
    time_to_peak_vel = float((int(np.argmax(speed)) - spre) / fs)
    # peak_accel / peak_decel on the SPEED (not signed velocity), so a leftward or
    # downward saccade can't swap the accelerating and decelerating phases.
    peak_accel = float(np.max(accel_of_speed))        # fastest speeding-up
    peak_decel = float(np.min(accel_of_speed))        # fastest slowing-down (negative)

    tail = max(1, int(0.1 * fs))                      # settled position (last 100 ms)
    eye_disp = float(np.mean(p[-tail:]))
    amplitude = abs(eye_disp)
    tgt = np.asarray(tgt, dtype=np.float64)
    tgt_disp = float(tgt[-1] - tgt[:spre].mean())
    gain = float(eye_disp / tgt_disp) if abs(tgt_disp) > 1e-3 else 0.0
    vigor = float(peak_velocity / amplitude) if amplitude > 1e-3 else 0.0  # vigour proxy

    # duration — time the task-axis speed stays above threshold
    post = speed[spre:]
    above = np.where(post > VEL_THRESHOLD)[0]
    if len(above):
        pk = int(np.argmax(post))
        below_after = np.where(post[pk:] < VEL_THRESHOLD)[0]
        end_rel = (pk + below_after[0]) if len(below_after) else (len(post) - 1)
        duration = float(max(end_rel - above[0], 0) / fs)
    else:
        duration = 0.0

    # latency — first crossing of the smoothed 2D total speed after the target onset
    if v_total_window is not None:
        cross = np.where(np.asarray(v_total_window)[spre:] > VEL_THRESHOLD)[0]
    else:
        cross = above
    latency = float(cross[0] / fs) if len(cross) else float((n - spre) / fs)

    # time to reach target — onset → eye first within 90% of its final displacement
    if abs(eye_disp) > 1e-3:
        reached = np.where(np.abs(p[spre:]) >= 0.9 * abs(eye_disp))[0]
        time_to_target = float(reached[0] / fs) if len(reached) else float((n - spre) / fs)
    else:
        time_to_target = float((n - spre) / fs)

    return np.array([latency, peak_velocity, peak_accel, peak_decel, amplitude,
                     duration, time_to_peak_vel, time_to_target, gain, vigor],
                    dtype=np.float32)
