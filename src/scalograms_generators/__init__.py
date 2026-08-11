"""Scalogram (CWT) generators.

All consume a cached CWT data_store built by `EventLockedCWTPipeline`; outputs
land under `imgs/wavelet_transformed_spectrograms/`.

  - per_trial_scalograms_generator.py — each trial's 4-channel CWT tensor as a
      2×2 PNG grid. Auto-detects the cache channel scheme:
        signal_mode='four_error' →
          ch0=|CWT(LH−TH)| ch1=|CWT(RH−TH)| ch2=|CWT(LV−TV)| ch3=|CWT(RV−TV)|
        signal_mode='legacy' →
          ch0=|CWT(L_err)| ch1=Re(CWT(L_err)) ch2=|CWT(R_err)| ch3=Re(CWT(R_err))

  Cohort-level aggregate maps (across trials & subjects, HC vs MCI):
  - mean_scalograms_generator.py                  — per-group mean CWT magnitudes
  - variance_maps_of_scalograms_generator.py         — intra-group CWT variance (instability)
  - squared_difference_maps_of_scalograms_generator.py — (MCI−HC)² CWT separability
  - data_tensor_processor.py                         — `WaveletSpectrogramCore` (shared cache loader)
"""
