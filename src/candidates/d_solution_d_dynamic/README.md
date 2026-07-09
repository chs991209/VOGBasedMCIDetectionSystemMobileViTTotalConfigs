# Candidate D — Solution D: DynamicLatentClassifier (Ragged Tensors)

**Status:** ✅ Complete — multiple task-subset ablations.
**Best result (task subset dependent):**
- Anti-saccade only (tasks 2, 6): AUROC 0.710 ± 0.139 (N=37)
- All 8 tasks: **AUROC 0.737 ± 0.148** (N=37)

## What it is

The evolution of Solution C that abandons the strict-parity rectangular bundle in favor of ragged tensors. Each subject contributes ALL their real trials per kept task (no upper cap), with only a MIN_TRIALS admission floor. Bootstrap-fill at T=1 keeps variance finite. Cross-axis energy ratio (computed from raw magnitudes at cache-build time) injected as an extra gate-input feature.

## Architecture summary

- **Backbone:** frozen `MobileViTModel("apple/mobilevit-small")`, `.eval()` mode inside forward
- **Input bundle:** ragged list per subject: `[NUM_TASKS × Tensor[T_ij, 4, 32, 32]]` with `T_ij` variable
- **Task selection:** configurable via `--tasks` CLI flag; MIN_TRIALS admission floor (default 1)
- **Adapter:** `Conv2d(4→3, K=3, padding=1) + BN + ReLU` — no upscale (feeds 32×32 directly to ViT)
- **Feature aggregation:** μ_t + σ²_t across variable T_ij, plus raw **cross-axis stability ratio** per (subject, task) from the cache — total per-task representation `[B, T, 2·D + 1] = [B, T, 1281]`
- **Gate:** shared `Linear(1281, 128, GELU, Linear 128→1)` per task, then `F.softmax(scores, dim=1)` over tasks
- **Fusion:** `(task_repr * gate_weights).sum(dim=1)` → `[B, 1281]`
- **Head:** `Linear(1281, 256) → GELU → Dropout(0.5) → Linear(256, 2)`
- **Training:** `AdamW(lr=1e-4, wd=1e-2)`, patience=30, best-checkpoint restored at inference. Ragged batches via `ragged_collate`.
- **Cache used:** `outputs/cache/data_store_meta_4err.pkl` (4-error CWT, 45° artifact threshold post-rebuild)

## Where the source lives

```
src/meta_classifier_renewed/                # current — this IS Solution D
├── data_processor/data_preprocessing.py    # SubjectBundleDataset ragged + ragged_collate
├── models/dynamic_latent_classifier.py     # DynamicLatentClassifier
├── model_trainers/meta_trainer.py          # tuple-unpacking (logits, gates)
├── evaluators/monte_carlo_evaluator.py     # Solution D evaluator
└── detection_caller/detection_caller.py    # `_dynamic_latent` run-id
```

## Reports

- `outputs/reports/meta_classifier_renewed/run_20260630_010113_dynamic_latent/analysis.md` — anti-saccade only, N=37, AUROC 0.710
- `outputs/reports/meta_classifier_renewed/run_20260630_022721_dynamic_latent_t01234567/analysis.md` — all 8 tasks, N=37, AUROC 0.737

## Related design decisions

- 30°→45° artifact-threshold rebuild recovered 63 % more events and eliminated the "30° trap" that had been dropping severe-MCI Hypermetria overshoots (`CODE_AUDIT_VS_MASTER_CONTEXT.md` §2)
- All-tasks ablation informed by `TASK_CONTRIBUTION_RATES.md` (Vertical > Horizontal, R-paradigm strongest single contributor)
