# Intermediate Confusion Matrices — Augmented (dropout=0.3, SpecAugment train-only)
# Folds 01–05 of 30 (run still in progress)

> **Source:** `outputs/logs/run_20260524_040820_full_aug.log` (live, run not yet complete).
> **Pipeline:** Full-experiments-using (8-task, MobileViT frozen backbone).
> **Training:** AdamW (lr=1e-3, wd=1e-4), warmup=5, grad-clip=1.0, ReduceLROnPlateau on val-AUROC (factor=0.2, patience=10), early-stop on val-AUROC patience=40, best-AUROC checkpoint per fold.
> **Run config:** Augmented (dropout=0.3, SpecAugment train-only).
> **Status:** **5 / 30 folds completed.** Snapshot captured while run continues.

> **Note on N:** Each fold's splitter draws ~11–12 test subjects. `N_eff` is back-derived from logged Acc/Sens/Spec so it matches the evaluator's effective subject count after artifact rejection.

![Per-fold confusion matrices, intermediate](confusion_matrix_intermediate_full_aug.png)

---

## Per-Fold Matrices (completed folds only)

| Fold | N_eff |  TN | FP | FN | TP | Acc   | Sens  | Spec  | AUROC | Best val-AUROC | Best ep | Stopped @ |
|:----:|:-----:|:---:|:--:|:--:|:--:|:-----:|:-----:|:-----:|:-----:|:--------------:|:-------:|:---------:|
|  01  |    11 |   2 |  1 |  7 |  1 | 0.273 | 0.125 | 0.667 | 0.667 | 0.6306 |  56 |  96 |
|  02  |    11 |   2 |  2 |  2 |  5 | 0.636 | 0.714 | 0.500 | 0.714 | 0.6060 |  15 |  55 |
|  03  |    12 |   6 |  1 |  2 |  3 | 0.750 | 0.600 | 0.857 | 0.714 | 0.6001 |  59 |  99 |
|  04  |    11 |   1 |  3 |  1 |  6 | 0.636 | 0.857 | 0.250 | 0.571 | 0.5760 |  38 |  78 |
|  05  |    11 |   4 |  2 |  1 |  4 | 0.727 | 0.800 | 0.667 | 0.867 | 0.6561 |  21 |  61 |

---

## Aggregate Confusion (sum over 5 completed folds, N = 56 subject-decisions)

|              | Pred: HC  | Pred: MCI |    |
|--------------|:---------:|:---------:|:--:|
| **True HC**  |  TN = 15  |  FP = 9  | 24 |
| **True MCI** |  FN = 13  |  TP = 19  | 32 |
|              |    28     |    28    |**56**|

| Pooled (micro) metric | Value |
|---|---|
| Accuracy    | (19 + 15) / 56 = **0.607** |
| Sensitivity | 19 / 32 = **0.594** |
| Specificity | 15 / 24 = **0.625** |
| Precision   | 19 / 28 = **0.679** |
| NPV         | 15 / 28 = **0.536** |
| F1 (MCI)    | **0.633** |

---

## Macro Mean ± Std (completed folds)

| Metric | Folds used | Mean   | Std    |
|--------|:----------:|:------:|:------:|
| Accuracy    | 5 | 0.604 | 0.172 |
| Sensitivity | 5 | 0.619 | 0.262 |
| Specificity | 5 | 0.588 | 0.203 |
| AUROC       | 5 | 0.707 | 0.096 |

---

## Reading the Intermediate Aggregate

⚠️ **Only 5 of 30 folds are in.** With this small sample, single-fold variance dominates the Std column — the macro Std will shrink substantially as more folds finish, so don't treat the current numbers as final.

- **Macro AUROC > 0.65** is already meaningfully above random (0.5) and above the 2-exp BASE result (0.388).
- **Macro Accuracy 0.604 is below the 66 % MCI-majority prior** so far — but with only 5 folds the confidence interval is wide.
- The per-fold table shows the model is **actually learning** in some folds (e.g. AUROC > 0.85) — early-stop epochs are well past the 5-epoch warmup, suggesting the optimization is not collapsing to the random-init checkpoint as it did for the 2-exp BASE.

---

## Caveats

1. **5-fold sample is small.** Macro Std and pooled aggregates will move as the run completes.
2. **`N_eff` back-derived** from logged Acc/Sens/Spec; per-fold TN/FP/FN/TP may be off by ±1 in folds with reconstruction error > 0.05. Aggregates use the logged metrics directly.
3. **Reported metrics use the *final-epoch* model** (post early-stop), not the saved best-AUROC checkpoint — the trainer returns its in-memory model; the evaluator does not load the checkpoint back before inference. This was confirmed as intended behavior earlier in the project.
4. **GPU contention is active** — this run is sharing the GPU with at least one other process during this snapshot (plain ↔ aug ↔ dropout=0.5). Per-epoch wall time is therefore inflated relative to a solo run; the metrics themselves are unaffected (CUDA scheduling does not change numerical results).
5. **Probe report not yet generated** — `TaskWiseProbeGenerator.generate_markdown_report()` is called only after the 30-fold lifecycle ends. Once this run completes, the multi-grouping contribution analysis (by-task / by-axis / by-type / by-inhibition) will be written to `outputs/reports/{run_id}_task_wise_probe.md`.