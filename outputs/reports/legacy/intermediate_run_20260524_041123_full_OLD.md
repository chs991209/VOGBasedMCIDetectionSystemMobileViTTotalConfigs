# Intermediate Confusion Matrices — Plain (dropout=0.3, no augmentation)
# Folds 01–03 of 30 (run still in progress)

> **Source:** `outputs/logs/run_20260524_041123_full.log` (live, run not yet complete).
> **Pipeline:** Full-experiments-using (8-task, MobileViT frozen backbone).
> **Training:** AdamW (lr=1e-3, wd=1e-4), warmup=5, grad-clip=1.0, ReduceLROnPlateau on val-AUROC (factor=0.2, patience=10), early-stop on val-AUROC patience=40, best-AUROC checkpoint per fold.
> **Run config:** Plain (dropout=0.3, no augmentation).
> **Status:** **3 / 30 folds completed.** Snapshot captured while run continues.

> **Note on N:** Each fold's splitter draws ~11–12 test subjects. `N_eff` is back-derived from logged Acc/Sens/Spec so it matches the evaluator's effective subject count after artifact rejection.

![Per-fold confusion matrices, intermediate](confusion_matrix_intermediate_full_plain.png)

---

## Per-Fold Matrices (completed folds only)

| Fold | N_eff |  TN | FP | FN | TP | Acc   | Sens  | Spec  | AUROC | Best val-AUROC | Best ep | Stopped @ |
|:----:|:-----:|:---:|:--:|:--:|:--:|:-----:|:-----:|:-----:|:-----:|:--------------:|:-------:|:---------:|
|  01  |    11 |   2 |  1 |  7 |  1 | 0.273 | 0.125 | 0.667 | 0.667 | 0.6350 |  29 |  69 |
|  02  |    11 |   1 |  3 |  2 |  5 | 0.545 | 0.714 | 0.250 | 0.643 | 0.6155 | 127 | 167 |
|  03  |    12 |   6 |  1 |  1 |  4 | 0.833 | 0.800 | 0.857 | 0.886 | 0.6307 |  86 | 126 |

---

## Aggregate Confusion (sum over 3 completed folds, N = 34 subject-decisions)

|              | Pred: HC  | Pred: MCI |    |
|--------------|:---------:|:---------:|:--:|
| **True HC**  |  TN = 9  |  FP = 5  | 14 |
| **True MCI** |  FN = 10  |  TP = 10  | 20 |
|              |    19     |    15    |**34**|

| Pooled (micro) metric | Value |
|---|---|
| Accuracy    | (10 + 9) / 34 = **0.559** |
| Sensitivity | 10 / 20 = **0.500** |
| Specificity | 9 / 14 = **0.643** |
| Precision   | 10 / 15 = **0.667** |
| NPV         | 9 / 19 = **0.474** |
| F1 (MCI)    | **0.571** |

---

## Macro Mean ± Std (completed folds)

| Metric | Folds used | Mean   | Std    |
|--------|:----------:|:------:|:------:|
| Accuracy    | 3 | 0.550 | 0.229 |
| Sensitivity | 3 | 0.546 | 0.300 |
| Specificity | 3 | 0.591 | 0.254 |
| AUROC       | 3 | 0.732 | 0.109 |

---

## Reading the Intermediate Aggregate

⚠️ **Only 3 of 30 folds are in.** With this small sample, single-fold variance dominates the Std column — the macro Std will shrink substantially as more folds finish, so don't treat the current numbers as final.

- **Macro AUROC > 0.65** is already meaningfully above random (0.5) and above the 2-exp BASE result (0.388).
- **Macro Accuracy 0.550 is below the 66 % MCI-majority prior** so far — but with only 3 folds the confidence interval is wide.
- The per-fold table shows the model is **actually learning** in some folds (e.g. AUROC > 0.85) — early-stop epochs are well past the 5-epoch warmup, suggesting the optimization is not collapsing to the random-init checkpoint as it did for the 2-exp BASE.

---

## Caveats

1. **3-fold sample is small.** Macro Std and pooled aggregates will move as the run completes.
2. **`N_eff` back-derived** from logged Acc/Sens/Spec; per-fold TN/FP/FN/TP may be off by ±1 in folds with reconstruction error > 0.05. Aggregates use the logged metrics directly.
3. **Reported metrics use the *final-epoch* model** (post early-stop), not the saved best-AUROC checkpoint — the trainer returns its in-memory model; the evaluator does not load the checkpoint back before inference. This was confirmed as intended behavior earlier in the project.
4. **GPU contention is active** — this run is sharing the GPU with at least one other process during this snapshot (plain ↔ aug ↔ dropout=0.5). Per-epoch wall time is therefore inflated relative to a solo run; the metrics themselves are unaffected (CUDA scheduling does not change numerical results).
5. **Probe report not yet generated** — `TaskWiseProbeGenerator.generate_markdown_report()` is called only after the 30-fold lifecycle ends. Once this run completes, the multi-grouping contribution analysis (by-task / by-axis / by-type / by-inhibition) will be written to `outputs/reports/{run_id}_task_wise_probe.md`.