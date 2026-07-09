# Anti-Saccade B Task-Wise Probe — Folds 01–03

> **Source:** `probe_generator.py`
> **Protocol:** Post-validation subject-level aggregation isolated by Horizontal vs Vertical.
> **Status:** 3 folds evaluated.

## 1. Macro Mean ± Std by Task

| Task ID | Experiment Name | Mean Accuracy | Accuracy Std | Mean AUROC | AUROC Std |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | Horizontal Saccade B (anti) | 0.478 | ± 0.084 | **0.444** | ± 0.109 |
| **1** | Vertical Saccade B (anti) | 0.464 | ± 0.038 | **0.459** | ± 0.057 |

## 2. Analytical Conclusion

This probe isolates the exact performance of Horizontal vs. Vertical Anti-Saccade B tasks. 
Use these AUROC variances to determine if Vertical or Horizontal components require differing attention weights in future architectures.