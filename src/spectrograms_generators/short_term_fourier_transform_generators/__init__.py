"""Short-Time Fourier Transform spectrogram visualizers.

Population-dashboard renderings of per-(subject, task, eye) STFT spectrograms.
Each visualizer is independently runnable; `generator.py` runs all three with
a shared Core build to avoid re-processing the 300 CSVs three times.
"""
