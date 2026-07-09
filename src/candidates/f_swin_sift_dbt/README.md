# Candidate F — Swin-Tiny + SIFT-DBT Attention-Weighted Vote

**Status:** 🚧 Under implementation.
**Target result:** > 0.79 AUROC (beat wvote baseline via density transfer + late-fusion attention + XAI).

## What it is

Two-stage hybrid pipeline that inherits the density-transfer idea from the (superseded) MobileViT hybrid (Candidate E) but adopts:

- **Swin-Tiny** (28 M params, ImageNet-pretrained at 224×224) instead of MobileViT-small
- **SIFT-DBT-style late-fusion** (per-task independent inference → attention-weighted vote of logits) instead of feature-level gate-weighted sum
- **Augmented AttentionGate** that sees both `Z_task` (μ ‖ σ² distribution features) AND the task's own predicted `logits_task` — enables an XAI narrative like "trust this task because its own confidence is high"
- **Optional probe-derived attention prior** (`--attention-prior {none, wvote, loo}`) to warm-start the gate with the wvote soft-vote scheme or the direct per-task LOO ΔAUROC signals

## Architecture (both stages)

### Stage 1 — Domain-Adaptive Pretrain

- **Data:** flat 8-task CWT windows (from `TaskConditionedDataset` on the 4-error cache), restricted per-fold to Stage 2's training subjects (inner 90/10 subject split for Stage 1 val monitoring)
- **Model:** legacy adapter (`Conv2d(4→3, K=(5,1)) + BN + ReLU`) + 224×224 upscale + Swin-Tiny UNFROZEN + `Linear(768, 2)` task-agnostic head (no task embedding)
- **Training:** `AdamW(lr=5e-5, wd=1e-4)`, 5-epoch warmup, `ReduceLROnPlateau` on val loss, grad-clip 1.0, AMP autocast, patience 10, epochs cap 50
- **Output artifact:** per-fold `backbone + adapter` state_dict saved to `outputs/checkpoints/run_<id>/fold_NN_stage1.pth`

### Stage 2 — Clinical Fine-Tune (SIFT-DBT)

- **Data:** ragged `SubjectBundleDataset` with `keep_task_ids=(0..7)`, `min_trials=1`, bootstrap-fill at T=1
- **Model:** load Stage 1's `backbone + adapter`; **freeze both** (`.eval()` mode enforced via `.train()` override to protect BN running stats). SharedClassifier + augmented AttentionGate operate on per-task μ ‖ σ² features (+ optional attention_prior offset).
- **Return signature:** `(final_logits, logits_task, W_task)` — for XAI
- **Training:** `AdamW(lr=1e-4, wd=1e-2)` on gate + classifier + attention_prior only, patience 30, best-checkpoint restored

### SIFT-DBT topology (Stage 2 forward, sketched)

```
ragged_bundle → per-trial adapter+backbone forward
    ↓  [T_ij, 768]  per (subject, task)
    ↓
per-task μ, σ² over trials  →  [B, T=8, 1536]  (call this Z_task)
    ↓
    ├──►  SharedClassifier(Z_task)  → logits_task [B, 8, 2]
    │
    └──►  concat[Z_task, logits_task] → AttentionGate → scores [B, 8, 1]
              + attention_prior (per-task learnable offset)
              → F.softmax(scores, dim=1) → W_task [B, 8, 1]

final_logits = (W_task * logits_task).sum(dim=1)  → [B, 2]
```

## Files (to be created in this directory)

```
src/candidates/f_swin_sift_dbt/
├── __init__.py                              (docstring — done)
├── README.md                                (this file)
├── models/
│   ├── __init__.py
│   ├── stage1_backbone_tuner.py             # Swin-Tiny unfrozen + task-agnostic head
│   └── sift_dbt_classifier.py               # SharedClassifier + AttentionGate + late fusion
├── model_trainers/
│   ├── __init__.py
│   ├── stage1_trainer.py                    # window-level supervised trainer
│   └── stage2_trainer.py                    # bundle-level trainer (unpacks 3-tuple return)
├── data_processor/
│   ├── __init__.py
│   └── datasets.py                          # SubjectBundleDataset (ragged) + ragged_collate
│                                             #  + TaskConditionedDataset (flat windows, 3-tuple compat)
├── hybrid_trainer.py                        # 30-fold orchestrator with checkpoint-and-resume
├── attention_priors.py                      # ATTENTION_PRIOR_SCHEMES dict + resolver
└── detection_caller/
    ├── __init__.py
    └── sift_dbt_caller.py                   # CLI entry point
```

## Key config decisions (locked by Gemini's FINAL AUTHORIZATION)

| Item | Value |
|---|---|
| Backbone | `microsoft/swin-tiny-patch4-window7-224` (Swin-Base/Large forbidden) |
| Task scope | NUM_TASKS = 8 (§1/§5 deviation for the ablation) |
| Attention gate input | `concat([Z_task, logits_task])` — augmented variant |
| Task-attention prior | Optional; `--attention-prior {none, wvote, loo}` |
| Attribution isolation | Single run D only (no 4-run ablation) |
| Checkpoint-and-resume | **Mandatory** |
| Coexistence | Do NOT overwrite `src/meta_classifier_renewed/*`; parallel package |
| Runtime target | ~30-35 h |

## Attention-prior scheme constants

Defined in `attention_priors.py`:

```python
ATTENTION_PRIOR_SCHEMES = {
    "none":  None,
    "wvote": [0.0, 0.5, 0.5, 0.5, 0.0, 1.5, 1.5, 1.5],
    "loo":   [-0.48, -0.045, -0.09, +0.465, -0.015, +0.03, -0.045, +0.78],  # 15× LOO ΔAUROC
}
```

## Run-id template

```
run_<ts>_sift_dbt_swin_s1<N>_s2<M>_optA_prior_<flag>
    where N = --stage1-epochs (default 50)
          M = --stage2-epochs (default 500)
          flag ∈ {none, wvote, loo}
          optA = Stage 2 adapter frozen alongside backbone (from prior authorization)
```

## Cross-references

- `SIFT_DBT_PROPOSAL_AUDIT.md` — full pre-implementation audit + Gemini FINAL AUTHORIZATION resolution
- `SWIN_ADAPTION_GUIDE.md` — backbone hyperparameter authorization
- `HYBRID_PIPELINE_EXECUTION_READY.md` — origin of the `.train()` BN override (still applies here)
- `TASK_CONTRIBUTION_RATES.md` — evidence base for the `loo` attention-prior values
- `../a_wvote_softvote/README.md` — origin of the `wvote` attention-prior values
