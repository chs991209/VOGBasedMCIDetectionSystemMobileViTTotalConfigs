# Combined Subject-Level Confusion Matrices — 2-Experiment Anti-Saccade B Pipeline
# Folds 01–30 (complete run)

> **Source:** `outputs/logs/run_20260524_010543.log`, clean single-process run that completed all 30 folds normally.
> **Pipeline:** 2-task Anti-Saccade B isolation (Horizontal anti = task 0, Vertical anti = task 1).
> **Model:** Task-Conditioned MobileViT (frozen `apple/mobilevit-small`; trainable head only — adapter + `Embedding(2, 16)` + `CosineLinear(656, 2)`).
> **Training:** AdamW (lr=1e-3, wd=1e-4), CrossEntropyLoss with inverse-frequency class weights, **dropout=0.5**, CosineAnnealingLR (T_max=500), **early-stop on val-loss patience=40**, no warmup, no grad clipping.
> **Protocol:** 30-fold Stratified MC Group CV, 70/30 subject-level split, soft-voting per subject.
> **Status:** **30 / 30 folds complete.** Run wall time: 1 h 0 min 41 s (during 2-way GPU sharing with the parallel AUG run).

> **Note on N:** Each fold's splitter draws 11 test subjects. The *evaluated* count differs when some test subjects had no eligible epochs after artifact rejection. The `N_eff` column below is the back-derived effective subject count consistent with the logged Acc/Sens/Spec — for several folds it is 9–10 rather than 11.

![All 30 Confusion Matrices](./confusion_matrix_30folds_combined.png)

---

## Per-Fold Matrices

| Fold | N_eff |  TN | FP | FN | TP | Acc   | Sens  | Spec  | AUROC | Best val-loss | Best ep | Stopped @ |
|:----:|:-----:|:---:|:--:|:--:|:--:|:-----:|:-----:|:-----:|:-----:|:-------------:|:-------:|:---------:|
|  01  |    10 |   3 |  0 |  7 |  0 | 0.308 | 0.000 | 1.000 | 0.333 | 0.7396 |   6 |  46 |
|  02  |    10 |   1 |  3 |  2 |  4 | 0.500 | 0.625 | 0.250 | 0.531 | 0.6836 |   6 |  46 |
|  03  |     9 |   3 |  1 |  3 |  2 | 0.583 | 0.400 | 0.714 | 0.486 | 0.6764 |  65 | 105 |
|  04  |    10 |   2 |  1 |  5 |  2 | 0.400 | 0.286 | 0.667 | 0.524 | 0.7859 |  37 |  77 |
|  05  |    10 |   3 |  1 |  4 |  2 | 0.500 | 0.375 | 0.750 | 0.562 | 0.6850 |   9 |  49 |
|  06  |     9 |   0 |  2 |  2 |  5 | 0.556 | 0.714 | 0.000 | 0.143 | 0.7032 |   8 |  48 |
|  07  |    10 |   2 |  2 |  2 |  4 | 0.600 | 0.667 | 0.500 | 0.667 | 0.6731 |   3 |  43 |
|  08  |    10 |   0 |  2 |  4 |  4 | 0.400 | 0.500 | 0.000 | 0.188 | 0.6671 |   2 |  42 |
|  09  |    11 |   1 |  2 |  8 |  0 | 0.083 | 0.000 | 0.333 | 0.148 | 0.7121 |   2 |  42 |
|  10  |     9 |   0 |  4 |  1 |  4 | 0.444 | 0.800 | 0.000 | 0.100 | 0.6732 |   2 |  42 |
|  11  |     9 |   2 |  1 |  5 |  1 | 0.333 | 0.222 | 0.667 | 0.593 | 0.7179 |   1 |  41 |
|  12  |    10 |   1 |  4 |  2 |  3 | 0.417 | 0.571 | 0.200 | 0.200 | 0.7148 |  40 |  80 |
|  13  |    11 |   3 |  1 |  5 |  2 | 0.417 | 0.250 | 0.750 | 0.281 | 0.6891 |  31 |  71 |
|  14  |     9 |   2 |  1 |  5 |  1 | 0.333 | 0.222 | 0.667 | 0.222 | 0.7252 |  13 |  53 |
|  15  |     9 |   6 |  2 |  1 |  0 | 0.667 | 0.000 | 0.750 | 0.750 | 0.4702 |   1 |  41 |
|  16  |    10 |   2 |  5 |  1 |  2 | 0.400 | 0.667 | 0.286 | 0.571 | 0.7546 |   7 |  47 |
|  17  |    10 |   3 |  0 |  5 |  2 | 0.500 | 0.286 | 1.000 | 0.762 | 0.6254 |  87 | 127 |
|  18  |     9 |   6 |  1 |  2 |  0 | 0.667 | 0.000 | 0.857 | 0.357 | 0.4943 |   9 |  49 |
|  19  |     9 |   1 |  2 |  2 |  4 | 0.583 | 0.667 | 0.333 | 0.370 | 0.6711 |   4 |  44 |
|  20  |    10 |   1 |  3 |  2 |  4 | 0.500 | 0.667 | 0.250 | 0.292 | 0.7653 |   2 |  42 |
|  21  |     9 |   1 |  2 |  3 |  3 | 0.444 | 0.500 | 0.333 | 0.389 | 0.7522 |   1 |  41 |
|  22  |     7 |   1 |  1 |  4 |  1 | 0.286 | 0.200 | 0.500 | 0.200 | 0.6916 | 110 | 150 |
|  23  |    10 |   0 |  3 |  1 |  6 | 0.600 | 0.857 | 0.000 | 0.619 | 0.6616 |  54 |  94 |
|  24  |    10 |   0 |  1 |  3 |  6 | 0.600 | 0.667 | 0.000 | 0.000 | 0.6813 |   1 |  41 |
|  25  |    10 |   3 |  1 |  3 |  3 | 0.600 | 0.500 | 0.750 | 0.625 | 0.6676 |  42 |  82 |
|  26  |     8 |   2 |  1 |  4 |  1 | 0.375 | 0.200 | 0.667 | 0.400 | 0.6419 |   2 |  42 |
|  27  |    10 |   0 |  1 |  6 |  3 | 0.300 | 0.333 | 0.000 | 0.111 | 0.6878 |   3 |  43 |
|  28  |    10 |   1 |  3 |  3 |  3 | 0.417 | 0.500 | 0.250 | 0.312 | 0.6848 |   9 |  49 |
|  29  |    10 |   3 |  1 |  3 |  3 | 0.583 | 0.500 | 0.750 | 0.531 | 0.6847 |  17 |  57 |
|  30  |    10 |   0 |  3 |  1 |  6 | 0.600 | 0.857 | 0.000 | 0.381 | 0.6894 |  21 |  61 |


<details>
<summary><b>📊 Click to expand: per-fold 2×2 confusion matrices (inline markdown)</b></summary>

**Fold 01** — N=10, Acc=0.308, AUROC=0.333

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    3    |    0     |
| **True MCI** |    7    |    0     |

**Fold 02** — N=10, Acc=0.500, AUROC=0.531

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    1    |    3     |
| **True MCI** |    2    |    4     |

**Fold 03** — N=9, Acc=0.583, AUROC=0.486

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    3    |    1     |
| **True MCI** |    3    |    2     |

**Fold 04** — N=10, Acc=0.400, AUROC=0.524

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    2    |    1     |
| **True MCI** |    5    |    2     |

**Fold 05** — N=10, Acc=0.500, AUROC=0.562

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    3    |    1     |
| **True MCI** |    4    |    2     |

**Fold 06** — N=9, Acc=0.556, AUROC=0.143

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    0    |    2     |
| **True MCI** |    2    |    5     |

**Fold 07** — N=10, Acc=0.600, AUROC=0.667

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    2    |    2     |
| **True MCI** |    2    |    4     |

**Fold 08** — N=10, Acc=0.400, AUROC=0.188

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    0    |    2     |
| **True MCI** |    4    |    4     |

**Fold 09** — N=11, Acc=0.083, AUROC=0.148

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    1    |    2     |
| **True MCI** |    8    |    0     |

**Fold 10** — N=9, Acc=0.444, AUROC=0.100

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    0    |    4     |
| **True MCI** |    1    |    4     |

**Fold 11** — N=9, Acc=0.333, AUROC=0.593

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    2    |    1     |
| **True MCI** |    5    |    1     |

**Fold 12** — N=10, Acc=0.417, AUROC=0.200

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    1    |    4     |
| **True MCI** |    2    |    3     |

**Fold 13** — N=11, Acc=0.417, AUROC=0.281

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    3    |    1     |
| **True MCI** |    5    |    2     |

**Fold 14** — N=9, Acc=0.333, AUROC=0.222

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    2    |    1     |
| **True MCI** |    5    |    1     |

**Fold 15** — N=9, Acc=0.667, AUROC=0.750

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    6    |    2     |
| **True MCI** |    1    |    0     |

**Fold 16** — N=10, Acc=0.400, AUROC=0.571

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    2    |    5     |
| **True MCI** |    1    |    2     |

**Fold 17** — N=10, Acc=0.500, AUROC=0.762

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    3    |    0     |
| **True MCI** |    5    |    2     |

**Fold 18** — N=9, Acc=0.667, AUROC=0.357

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    6    |    1     |
| **True MCI** |    2    |    0     |

**Fold 19** — N=9, Acc=0.583, AUROC=0.370

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    1    |    2     |
| **True MCI** |    2    |    4     |

**Fold 20** — N=10, Acc=0.500, AUROC=0.292

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    1    |    3     |
| **True MCI** |    2    |    4     |

**Fold 21** — N=9, Acc=0.444, AUROC=0.389

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    1    |    2     |
| **True MCI** |    3    |    3     |

**Fold 22** — N=7, Acc=0.286, AUROC=0.200

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    1    |    1     |
| **True MCI** |    4    |    1     |

**Fold 23** — N=10, Acc=0.600, AUROC=0.619

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    0    |    3     |
| **True MCI** |    1    |    6     |

**Fold 24** — N=10, Acc=0.600, AUROC=0.000

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    0    |    1     |
| **True MCI** |    3    |    6     |

**Fold 25** — N=10, Acc=0.600, AUROC=0.625

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    3    |    1     |
| **True MCI** |    3    |    3     |

**Fold 26** — N=8, Acc=0.375, AUROC=0.400

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    2    |    1     |
| **True MCI** |    4    |    1     |

**Fold 27** — N=10, Acc=0.300, AUROC=0.111

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    0    |    1     |
| **True MCI** |    6    |    3     |

**Fold 28** — N=10, Acc=0.417, AUROC=0.312

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    1    |    3     |
| **True MCI** |    3    |    3     |

**Fold 29** — N=10, Acc=0.583, AUROC=0.531

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    3    |    1     |
| **True MCI** |    3    |    3     |

**Fold 30** — N=10, Acc=0.600, AUROC=0.381

|              | Pred HC | Pred MCI |
|--------------|:-------:|:--------:|
| **True HC**  |    0    |    3     |
| **True MCI** |    1    |    6     |
</details>

---

## Aggregate Confusion (sum over 30 folds, N = 288 subject-decisions)

|              | Pred: HC  | Pred: MCI |    |
|--------------|:---------:|:---------:|:--:|
| **True HC**  |  TN = 53  |  FP = 55  | 108 |
| **True MCI** |  FN = 99  |  TP = 81  | 180 |
|              |    152     |    136    |**288**|

| Pooled (micro) metric | Value |
|---|---|
| Accuracy    | (81 + 53) / 288 = **0.465** |
| Sensitivity | 81 / 180 = **0.450** |
| Specificity | 53 / 108 = **0.491** |
| Precision   | 81 / 136 = **0.596** |
| NPV         | 53 / 152 = **0.349** |
| F1 (MCI)    | **0.513** |

---

## Macro Mean ± Std

| Metric | Folds used | Mean   | Std    |
|--------|:----------:|:------:|:------:|
| Accuracy    | 30 | 0.467 | 0.132 |
| Sensitivity | 30 | 0.434 | 0.256 |
| Specificity | 30 | 0.441 | 0.324 |
| AUROC       | 30 | 0.388 | 0.202 |

---

## Reading the Aggregate

- **Mean accuracy 0.467 ± 0.132 is well below the 66 % MCI-majority prior** (0.660) — the 2-experiment isolation underperforms the trivial *always-predict-MCI* classifier by ~20 pp on average.
- **Macro AUROC 0.388 is *below random* (0.500).** 19 / 30 folds yielded AUROC < 0.5, meaning the model is anti-discriminative on a majority of held-out subjects. AUROC = 0.5 (random) lives outside the macro ± 1 std interval.
- **Sensitivity ≈ Specificity ≈ ~0.45 (pooled).** Unlike the 8-experiment run where the model leaned MCI (sens > spec), this 2-experiment model has no consistent class preference — its errors are roughly symmetric. **4 folds collapsed to never predicting MCI** (Sens = 0); **7 folds collapsed to never predicting HC** (Spec = 0). Combined: 11 / 30 folds (~37 %) produced a degenerate single-class predictor.
- **The training loop almost never escaped its warm-up.** In 19 / 30 folds the best val-loss was achieved by **epoch ≤ 10**, and 12 / 30 folds peaked at **epoch ≤ 5**. With patience = 40, these folds early-stopped near epoch 40–50 — the saved checkpoint is essentially a near-random-init model, not a trained one.
- **Standout folds where training *did* stabilise:** Fold 15 (Acc 0.667, AUROC 0.750), Fold 17 (Acc 0.500, AUROC 0.762), Fold 18 (Acc 0.667, AUROC 0.357), Fold 25 (Acc 0.600, AUROC 0.625). Even these have the dropout-induced symptom of low sensitivity (Sens ≤ 0.5) — the model classifies HC well but misses many MCI subjects.
- **Comparison vs the parallel 8-experiment run (22-fold partial):**

  | Pipeline | Folds | Macro Acc | Macro AUROC | Pooled Sens | Pooled Spec |
  |---|:---:|:---:|:---:|:---:|:---:|
  | 8-exp (no aug) | 22 | **0.640** | **0.649** | 0.705 | 0.556 |
  | 2-exp (no aug) | 30 | 0.467 | 0.388 | 0.450 | 0.491 |

  Restricting the input to Anti-Saccade B only — the paradigm doctors flagged as most discriminative — **hurts** detection by ~17 pp accuracy and ~26 pp AUROC under this training recipe. The clinical-hypothesis isolation does not survive the small-data regime that results from filtering out 6 of 8 task types.

---

## Why This Likely Underperformed (Root Cause Hypotheses)

1. **Data starvation.** Filtering to 2 anti-saccade tasks leaves ~110–160 train windows per fold (vs ~250–560 in the 8-task run). The frozen MobileViT backbone has nothing to anchor a useful representation against with so few examples; the trainable head (≈ 1.7 k params) over-regularises with `Dropout(0.5)`.
2. **`patience = 40` on val-loss interacts pathologically with a tiny val set.** Each fold's val set is ~70 windows ≈ 2–3 batches. Val loss bounces 0.01–0.05 epoch-to-epoch from minibatch noise. The lowest val-loss is frequently logged in the first 1–10 epochs — before the head has trained at all — and patience then triggers ~40 epochs later. The saved `fold_NN_best.pth` is therefore frequently the near-random-init model, and (per design) the evaluator uses the *final-epoch* model anyway, which is ~40 epochs further past that bad minimum.
3. **Dropout = 0.5 may be too aggressive for the 2-task head.** The 8-exp recipe used 0.3 with a 32-dim task embedding and trained the same MobileViT-small backbone successfully; we doubled dropout *and* shrunk the embedding to 16 dims, removing about half the regularising capacity that the original recipe used to compensate. The combined effect produced a head that cannot reliably overfit even the train set in many folds.
4. **No warm-up; no grad clipping.** The 8-exp trainer ramps LR over 5 epochs and clips grad-norm at 1.0. The 2-exp trainer uses CosineAnnealingLR straight from peak with no clip. With AMP + tiny batches + small trainable head, gradient spikes early in training can shove the head into a degenerate region that's then locked in by early-stop. The 19 folds where best-epoch ≤ 10 are consistent with this.
5. **`GroupShuffleSplit` does not stratify by class.** Test folds drew 2–9 MCI vs 9–2 HC subjects (per back-derivation). Several folds drew ≤ 3 of either class — a single subject flip moves Sens or Spec by 30 + pp, and the macro-Std reflects this (Sens std 0.256, Spec std 0.324).

---

## Caveats

1. **Subject-level variance is severe.** Most folds evaluate only 9–11 subjects after artifact rejection. One subject misclassification ≈ 9–11 pp swing in accuracy.
2. **N_eff column was back-derived** from logged Acc/Sens/Spec — the evaluator does not log the per-fold raw confusion matrix or HC/MCI split directly. For ~5 folds the reconstruction error is 0.05–0.07, meaning the TN/FP/FN/TP columns may be off by ±1; the macro and pooled aggregates use the *logged* metrics directly and are exact.
3. **`best_val_loss` checkpoint vs reported metrics.** The trainer saves `fold_NN_best.pth` at every val-loss improvement, but the MC evaluator infers from the trainer's *final-epoch* model (the in-memory object). The numbers in this report reflect the final-epoch model, not the saved checkpoint. They will diverge significantly given how early the best-epoch fell in most folds.
4. **No augmentation.** This is the BASE run. A parallel `--augment` run (`two_experiments_using/run_20260524_011532_aug`) was launched but was killed by the host mid-fold-28 — no completed AUG counterpart for direct ablation comparison yet.
5. **Probe report from this run uses the *old* window-level probe** (the process imported the pre-upgrade probe class at startup). The new fold-by-fold contribution probe is in place for any future runs.