"""Spectrogram visualizer packages.

Sub-packages:
  - short_term_fourier_transform_generators
        - data_tensor_processor          — `HighFreqSpectrogramCore` (STFT loader)
        - mean_spectrograms_visualizer   — per-group mean spectrograms
        - variance_maps_visualizer       — intra-group variance (instability)
        - squared_difference_maps_visualizer — (MCI-HC)^2 separability dashboard
        - generator                      — single orchestrator (run all 3)
"""
