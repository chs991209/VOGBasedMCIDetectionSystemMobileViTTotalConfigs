"""Time-series VOG visualizer.

Renders each raw VOG CSV recording as a 3-tier diagnostic figure:
  1. Raw waveform (target stimulus + L/R eye positions + anti-saccade expected target overlay)
  2. Tracking error (eye − expected-target, with anti-saccade inversion handled)
  3. Orthogonal cross-axis monitoring (off-axis L/R noise)

Output structure:
    imgs/time_series_visualized/<GROUP>/<subject_id>/<task_name>.png

Use `generator.py` (Python module + CLI) for batch rendering.
The original notebook (`visualization.ipynb`) is kept for interactive reference.
"""
