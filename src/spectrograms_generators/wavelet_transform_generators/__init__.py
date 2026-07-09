"""Continuous-Wavelet-Transform population spectrogram visualizers.

Mirrors the structure of `spectrograms_generators/short_term_fourier_transform_generators/`
but consumes the project's existing CWT cache built by `EventLockedCWTPipeline`
— each (subject, task) cell carries N event-locked [4, 32, 32] CWT tensors, and
this package aggregates them across trials and subjects for cohort comparison.

Files:
  - data_tensor_processor.py — `WaveletSpectrogramCore` (loads CWT cache)
  - mean_spectrograms_visualizer.py — per-group mean CWT magnitudes
  - variance_maps_visualizer.py — intra-group CWT variance (instability)
  - squared_difference_maps_visualizer.py — (MCI−HC)² CWT separability
  - generator.py — single orchestrator (runs all 3 visualizers from one cache load)

Cache channel mapping (legacy 4-channel CWT scheme):
  ch0 = |CWT(target_axis − L)|   →  'Left'  eye
  ch1 = Re(CWT(target_axis − L))  →  unused for dashboards
  ch2 = |CWT(target_axis − R)|   →  'Right' eye
  ch3 = Re(CWT(target_axis − R))  →  unused for dashboards
"""
