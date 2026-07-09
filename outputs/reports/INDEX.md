# Reports Index

Per-run folders under `outputs/reports/`. Each folder holds the run's
analysis markdown + a confusion-matrix PNG. Probe reports (when generated
at end of lifecycle) land at the top level under `outputs/reports/`.

## 2-experiment Anti-Saccade B pipeline

| Run ID | Config | Status | Folds |
|---|---|---|:---:|
| [`two_experiments_using/run_20260524_010543/`](two_experiments_using/run_20260524_010543/analysis.md) | dropout=0.5, patience=40, no aug | ✅ **complete** | 30/30 |
| [`two_experiments_using/run_20260524_011532_aug/`](two_experiments_using/run_20260524_011532_aug/intermediate_analysis.md) | dropout=0.5, patience=40, **+aug** | ⛔ **stopped** (process died May 24 02:11, mid fold 28) | 27/30 |

## Full-experiments-using (8-task) pipeline

**Comparison of finished runs:** [`COMPARISON_finished_full_runs.md`](full_experiments_using/COMPARISON_finished_full_runs.md)

| Run ID | Config | Status | Folds | Probe |
|---|---|---|:---:|:---:|
| [`full_experiments_using/run_20260524_041123_full/`](full_experiments_using/run_20260524_041123_full/analysis.md) | dropout=0.3, patience=40, no aug | ✅ **complete** | 30/30 | [probe](full_experiments_using/run_20260524_041123_full/task_contribution_probe.md) |
| [`full_experiments_using/run_20260524_040820_full_aug/`](full_experiments_using/run_20260524_040820_full_aug/analysis.md) | dropout=0.3, patience=40, **+aug** | ✅ **complete** | 30/30 | [probe](full_experiments_using/run_20260524_040820_full_aug/task_contribution_probe.md) |
| [`full_experiments_using/run_20260524_112316_full_drop050/`](full_experiments_using/run_20260524_112316_full_drop050/analysis.md) | **dropout=0.5**, patience=40, no aug | ✅ **complete** | 30/30 | [probe](full_experiments_using/run_20260524_112316_full_drop050/task_contribution_probe.md) |
| [`full_experiments_using/run_20260524_120827_full_drop050_pat030/`](full_experiments_using/run_20260524_120827_full_drop050_pat030/analysis.md) | **dropout=0.5, patience=30**, no aug | ✅ **complete** | 30/30 | [probe](full_experiments_using/run_20260524_120827_full_drop050_pat030/task_contribution_probe.md) |
| [`full_experiments_using/run_20260527_205622_full_drop050_pat030_wvote/`](full_experiments_using/run_20260527_205622_full_drop050_pat030_wvote/analysis.md) | dropout=0.5, patience=30, **weighted-vote** | ✅ **complete** | 30/30 | [probe](full_experiments_using/run_20260527_205622_full_drop050_pat030_wvote/task_contribution_probe.md) · [arch](../../WVOTE_SYSTEM_ARCHITECTURE.md) |

## Meta-classifier (Option C — sequence-level trial fusion)

Subject-level direct classification: no soft-voting, MAX_TRIALS=20 trials per task fused at MobileViT's deepest transformer block. HDLSS regularization (lr=1e-4, wd=1e-2) + Youden's J dynamic threshold replacing the static 0.5.

| Run ID | Config | Status | Folds | Probe |
|---|---|---|:---:|:---:|
| [`meta_classifier_using/run_20260531_002841_meta/`](meta_classifier_using/run_20260531_002841_meta/analysis.md) | dropout=0.5, patience=40, per_task_proj=16, fc_hidden=128, **lr=1e-4 / wd=1e-2 / Youden's J** | ✅ **complete** | 30/30 | — |

## Meta-classifier (Renewed — LeakFreeMetaClassifier, 4-task strict parity)

`meta_classifier_renewed/`. Distribution-Aware Gated Fusion adapted from the user-supplied `main_modified.py`: Conv2d adapter → frozen MobileViT-small (D=640) → μ‖σ² over 10 trials → softmax-gated weighted sum → classifier head. KEEP_TASK_IDS = (HSacA, HSacR, VSacA, VSacR) per **PROBLEM.md** / **PROBLEM.ko.md** — strict-parity rule keeps 32 of 37 subjects (HC=14, MCI=18). No padding mask. No Youden's J. Strict 0.5 threshold. **Best-AUROC checkpoint restored at inference** (drift bug discovered + fixed across two consecutive runs).

| Run ID | Config | Status | Folds | Probe |
|---|---|---|:---:|:---:|
| [`meta_classifier_renewed/run_20260629_143830_meta_renewed/`](meta_classifier_renewed/run_20260629_143830_meta_renewed/analysis.md) | dropout=0.5, **patience=30**, lr=1e-4, wd=1e-2, **best-ckpt restored** | ✅ **complete** | 30/30 | — |

## Meta-classifier (Solution D — Dynamic Latent Aggregation, Anti-Saccade)

`meta_classifier_renewed/` evolved into Solution D per the **Master Context** (see `CODE_AUDIT_VS_MASTER_CONTEXT.md`). Ragged tensors (variable trials per subject-task, no upper cap), bootstrap-fill at T=1, latent-space μ‖σ² aggregation with σ² as a cognitive-instability biomarker. Cross-axis stability ratio injected into the gate. **Active pipeline restricted to anti-saccade-B (Master Context §1/§5).** Artifact threshold raised 30°→45° to admit Hypermetria-overshoot events (the "30° trap") — kept events grew 5,736 → 9,378, and all 37 subjects now qualify at min_trials=1.

| Run ID | Config | Status | Folds | Probe |
|---|---|---|:---:|:---:|
| [`meta_classifier_renewed/run_20260630_010113_dynamic_latent/`](meta_classifier_renewed/run_20260630_010113_dynamic_latent/analysis.md) | tasks=2,6 (anti-only), **min_trials=1**, ragged, **artifact_thr=45°**, N=37 | ✅ **complete** | 30/30 | — |
| [`meta_classifier_renewed/run_20260630_022721_dynamic_latent_t01234567/`](meta_classifier_renewed/run_20260630_022721_dynamic_latent_t01234567/analysis.md) | **tasks=0..7 (all-8 ablation)**, min_trials=1, ragged, artifact_thr=45°, N=37 | ✅ **complete** | 30/30 | — |

## `legacy/`

Earlier-format probe reports + first-pass analysis files. Kept for reference; do not edit.
