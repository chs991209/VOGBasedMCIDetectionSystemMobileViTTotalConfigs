# Candidate E — MobileViT + Hybrid Two-Stage (Approved, Superseded Before Launch)

**Status:** ⏸️ Approved by prior FINAL AUTHORIZATION; superseded by Candidate F (Swin SIFT-DBT) before code was written.
**Best result:** — (never launched)

## What it was to be

The two-stage hybrid architecture designed to inject the density advantage of Candidate A (wvote) into the mathematical integrity of Candidate D (Solution D):

- **Stage 1 (Density transfer):** MobileViT-small UNFROZEN, trained on flat 8-task CWT windows per fold. Task-agnostic `Linear(640, 2)` head, no task embedding. Legacy `Conv2d(4→3, K=5×1)` adapter + 256×256 upscale.
- **Stage 2 (Clinical fine-tune):** Load Stage 1 backbone + adapter into `DynamicLatentClassifier` with both FROZEN (backbone + adapter). Train only the softmax gate + classifier head on the SubjectBundleDataset restricted to `--tasks 2,6` (anti-saccade only per Master Context §1/§5).

## Why it was superseded

Gemini's subsequent SIFT-DBT directive replaced this design with:

1. Backbone: MobileViT-small → Swin-Tiny (28 M params, ImageNet-pretrained at 224×224)
2. Aggregation: gate-weighted **feature** fusion → gate-weighted **logit** fusion (SIFT-DBT late-fusion pattern)
3. XAI: `(final_logits, logits_task, W_task)` return signature enables per-subject attention-narrative reporting
4. Task scope: Stage 2 explicit multi-patch attention across ALL 8 tasks (§1/§5 deviation for the ablation)

Candidate F now carries the design forward with these upgrades. Candidate E's design documentation is preserved for context on how F evolved.

## Design docs (never converted to code)

- `HYBRID_PIPELINE_PRE_IMPLEMENTATION_AUDIT.md` — the 6-item resolution matrix
- `HYBRID_PIPELINE_EXECUTION_READY.md` — the final MobileViT-hybrid pre-code checkpoint (includes the `.train()` override for BN safety, applicable to F as well)

## What was extracted forward into F

- Per-fold Stage 1 pretraining (strict train/test parity, no leakage)
- Best-checkpoint-restore pattern from Solution D
- The BN freezing `.train()` override (BN running stats would otherwise drift silently through the frozen backbone)
- `TaskConditionedDataset` 3-tuple compatibility patch
- Inner 90/10 subject split for Stage 1 validation monitoring

## Nothing in this directory beyond this README

By design. E is a documentation-only entry in the candidate catalog. All its ideas live on in F.
