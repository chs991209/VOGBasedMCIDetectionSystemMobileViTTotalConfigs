# Candidate C — Solution C: LeakFreeMetaClassifier, 4-Task Strict Parity

**Status:** ✅ Complete.
**Best result:** Macro AUROC = **0.711 ± 0.164**, Macro Acc = 0.550 (N=32, 4 plenty tasks, MAX_TRIALS=10).

## What it is

The first mathematically leak-free rewrite of the meta-classifier line. Distribution-Aware Gated Fusion: per-task μ and σ² over exactly 10 trials in the frozen ViT's latent space, gated through a softmax-across-tasks attention, weighted-sum fused into a single subject-level feature vector, classified through an MLP head.

## Architecture summary

- **Backbone:** frozen `MobileViTModel("apple/mobilevit-small")`
- **Input bundle:** `[B, NUM_TASKS=4, MAX_TRIALS=10, C=4, 32, 32]` — strict parity, no padding mask
- **Task selection:** `KEEP_TASK_IDS = (0, 3, 4, 7)` — HSacA, HSacR, VSacA, VSacR ("plenty tasks" per `PROBLEM.md`)
- **Adapter:** `Conv2d(4→3, K=3, padding=1) + BN + ReLU`, then `F.interpolate((256, 256), mode='nearest')`
- **Feature aggregation:** μ_t + σ²_t across 10 trials in the frozen ViT's `pooler_output` space (D=640) → `[B, 4, 1280]`
- **Gate:** shared `Linear(1280, 128, GELU, Linear 128→1)` per task, then `F.softmax(scores, dim=1)` over tasks
- **Fusion:** `(task_repr * gate_weights).sum(dim=1)` → `[B, 1280]`
- **Head:** `Linear(1280, 128) → GELU → Dropout → Linear(128, 2)`
- **Training:** subject-level batches (batch_size=8), `AdamW(lr=1e-4, wd=1e-2)` — HDLSS regularization, patience=30, **best-checkpoint restored before inference**
- **Decision rule:** strict 0.5 threshold (Youden's J-on-test-set explicitly forbidden after the Option-C leak)

## Where the source lives

```
src/meta_classifier_renewed/  (git history — pre-Solution-D refactor)
├── data_processor/data_preprocessing.py
├── models/leak_free_classifier.py         # LeakFreeMetaClassifier
├── model_trainers/meta_trainer.py
├── evaluators/monte_carlo_evaluator.py
└── detection_caller/detection_caller.py
```

**Note:** `dynamic_latent_classifier.py` in the current `meta_classifier_renewed/` supersedes this file. To reproduce Solution C exactly, check out the commit hash referenced in the run's analysis.md.

## Reports

- `outputs/reports/meta_classifier_renewed/run_20260629_143830_meta_renewed/analysis.md`

## Why it underperformed A

See `META_RENEWED_VS_WVOTE.ko.md` §2.1 — the per-fold training-sample density fell from ~3,400 windows to 22 subject-bundles, a ~155× reduction. The rest of the gap decomposition is in that document.
