# Candidate A — Weighted Soft-Vote (`full_experiments_using` + wvote)

**Status:** ✅ Complete — the headline baseline for the project.
**Best result:** Macro AUROC = **0.791 ± 0.142**, Macro Acc = 0.717 (30-fold Monte-Carlo Group CV, N=37).

## What it is

The legacy per-window training + subject-level weighted soft-vote pipeline. All 8 tasks are used; each event-locked CWT window is a training example; at inference, per-window MCI probabilities are aggregated with hand-coded task weights.

## Architecture summary

- **Backbone:** frozen `MobileViTModel("apple/mobilevit-small")`
- **Input:** `[B, 4, 32, 32]` CWT tensor (legacy 4-channel: `{mag_L, re_L, mag_R, re_R}` on the task's axis)
- **Adapter:** `Conv2d(4→3, K=5×1, padding=2×0) + BN + ReLU`, then `F.interpolate(size=(256,256), mode='nearest')`
- **Feature pool:** `last_hidden_state.mean(dim=[2,3])` → `[B, 640]`
- **Task conditioning:** `nn.Embedding(8, 32)`; concat → `[B, 672]`
- **Head:** `Dropout(0.5) + CosineLinear(672→2, scale=10.0)` (L2-normalised metric head)
- **Training:** per-window batches of 32, `AdamW(lr=1e-3, wd=1e-4)`, warmup 5, `ReduceLROnPlateau` on val-AUROC, grad-clip 1.0, patience 30 early-stop, weighted CE
- **Inference:** subject-level **weighted soft-vote**: `Σ(w_t · p) / Σ(w_t)` with

```python
WEIGHTED_VOTE_SCHEME = {0: 0.0, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.0, 5: 1.5, 6: 1.5, 7: 1.5}
```

## Where the source lives

```
src/full_experiments_using/
├── data_processor/data_engineering.py     # EventLockedCWTPipeline + TaskConditionedDataset
├── models/mobile_vit_model.py             # TransferMobileViTClassifier + CosineLinear
├── model_trainers/mobile_vit_trainer.py   # ModelTrainer
├── evaluators/monte_carlo_evaluator.py    # MonteCarloGroupEvaluator (+ _aggregate_subject_prob)
└── detection_caller/detection_caller.py   # WEIGHTED_VOTE_SCHEME lives here
```

## Reports for the headline run

- `outputs/reports/full_experiments_using/run_20260527_205622_full_drop050_pat030_wvote/analysis.md`
- `outputs/reports/full_experiments_using/run_20260527_205622_full_drop050_pat030_wvote/task_contribution_probe.md`
- `WVOTE_SYSTEM_ARCHITECTURE.md` (project-root deep dive)

## Why it's the reference

The headline 0.791 AUROC. Every subsequent candidate is measured against this number.
