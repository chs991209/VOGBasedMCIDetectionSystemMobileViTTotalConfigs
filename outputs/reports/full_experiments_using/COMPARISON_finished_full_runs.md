# Finished Full-Experiments-Using Runs — Comparison

Side-by-side of the **completed** 8-task runs. All share the same architecture, splitter (`random_state=42` → identical folds), and 30-fold protocol; they differ in augmentation, dropout (0.3 vs 0.5), early-stop patience (40 vs 30), and subject-level aggregation (plain mean vs probe-derived weighted vote).

## Pooled (micro) metrics

| Run | Aug | Acc | Sens | Spec | PPV | NPV |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `full_experiments_using/run_20260524_041123_full` | — | 0.638 | 0.658 | 0.605 | 0.731 | 0.521 |
| `full_experiments_using/run_20260524_040820_full_aug` | ✅ | 0.670 | 0.685 | 0.645 | 0.757 | 0.559 |
| `full_experiments_using/run_20260524_112316_full_drop050` | — | 0.657 | 0.665 | 0.645 | 0.751 | 0.544 |
| `full_experiments_using/run_20260524_120827_full_drop050_pat030` | — | 0.681 | 0.712 | 0.631 | 0.758 | 0.575 |
| `full_experiments_using/run_20260527_205622_full_drop050_pat030_wvote` | — | 0.717 | 0.777 | 0.628 | 0.758 | 0.653 |

## Macro Mean ± Std

| Run | Aug | Acc | Sens | Spec | AUROC |
|---|:---:|:---:|:---:|:---:|:---:|
| `full_experiments_using/run_20260524_041123_full` | — | 0.637 ± 0.156 | 0.670 ± 0.233 | 0.616 ± 0.269 | 0.701 ± 0.151 |
| `full_experiments_using/run_20260524_040820_full_aug` | ✅ | 0.668 ± 0.146 | 0.686 ± 0.221 | 0.649 ± 0.269 | 0.707 ± 0.158 |
| `full_experiments_using/run_20260524_112316_full_drop050` | — | 0.655 ± 0.128 | 0.666 ± 0.186 | 0.642 ± 0.261 | 0.730 ± 0.155 |
| `full_experiments_using/run_20260524_120827_full_drop050_pat030` | — | 0.679 ± 0.137 | 0.716 ± 0.213 | 0.623 ± 0.257 | 0.731 ± 0.155 |
| `full_experiments_using/run_20260527_205622_full_drop050_pat030_wvote` | — | 0.717 ± 0.144 | 0.788 ± 0.197 | 0.620 ± 0.253 | 0.791 ± 0.146 |

## Aggregate Confusion Matrices

### Full-exp Plain (dropout=0.3, patience=40, no aug)
![aggregate](full_experiments_using/run_20260524_041123_full/aggregate_confusion.png)

### Full-exp Augmented (dropout=0.3, patience=40, +aug)
![aggregate](full_experiments_using/run_20260524_040820_full_aug/aggregate_confusion.png)

### Full-exp Dropout 0.5 (patience=40, no aug)
![aggregate](full_experiments_using/run_20260524_112316_full_drop050/aggregate_confusion.png)

### Full-exp Dropout 0.5 + Patience 30 (no aug)
![aggregate](full_experiments_using/run_20260524_120827_full_drop050_pat030/aggregate_confusion.png)

### Full-exp Dropout 0.5 + Patience 30 + WEIGHTED VOTE (no aug)
![aggregate](full_experiments_using/run_20260527_205622_full_drop050_pat030_wvote/aggregate_confusion.png)

---

Per-run detail lives inside each run folder, `outputs/reports/<run_id>/`:
- `analysis.md` — per-fold confusion matrices + aggregate + macro stats
- `confusion_matrix.png` — 30-fold grid · `aggregate_confusion.png` — pooled 2×2
- `task_contribution_probe.md` — multi-grouping task-contribution analysis (by-task / by-axis / by-type / by-inhibition + cross-grouping ranking)