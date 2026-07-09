"""Wavelet-spectrogram visualizer.

Reads a cached CWT data_store (built by `EventLockedCWTPipeline`) and renders
each trial's 4-channel CWT tensor as a 2×2 PNG grid under
`imgs/wavelet_transformed_spectrograms/`.

Auto-detects the cache's channel scheme:
  - signal_mode='four_error' (default meta cache) →
        ch0=|CWT(LH−TH)| ch1=|CWT(RH−TH)| ch2=|CWT(LV−TV)| ch3=|CWT(RV−TV)|
  - signal_mode='legacy' (full-experiments cache) →
        ch0=|CWT(L_err)| ch1=Re(CWT(L_err)) ch2=|CWT(R_err)| ch3=Re(CWT(R_err))
"""
