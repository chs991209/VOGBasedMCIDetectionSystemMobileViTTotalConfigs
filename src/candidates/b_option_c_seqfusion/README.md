# Candidate B — Option C: Sequence-Level Trial Fusion (DEPRECATED)

**Status:** ⚰️ Deprecated — padding-mask data-leakage vector suspected.
**Best result (contaminated):** Macro AUROC = 0.978 ± 0.059 — do not cite.

## What it was

An attempt to fuse all trials of all tasks into a single continuous sequence inside MobileViT-small's deepest transformer block, by monkey-patching the self-attention to accept a padding mask. The mask indicated which of the fixed `MAX_TRIALS=20` slots were real vs zero-padded per (subject, task).

## Why it was deprecated

1. **Padding-mask leak hypothesis.** Each subject's `(task, trial_count)` fingerprint was silently exposed to the network via the mask pattern. If HC vs MCI subjects have systematically different per-task trial counts (they do — see `PROBLEM.md`), the model could classify by *counting real slots* rather than by reading the CWT content.
2. **Anomalously high AUROC (0.978) at extreme threshold saturation** — 27 of 30 Youden's-J-derived thresholds landed at 1.000, consistent with polarised model outputs a leak would produce.
3. **The subsequent renewed pipeline (C and D) with zero-padding removed and strict-parity subject filtering reproduced only 0.71 AUROC on the same fold splits** — a 0.27 gap that is best explained by the leak.

Kept in the project as a cautionary tale and a reference for what NOT to do with padded ragged batches.

## Where the source lives

```
src/meta_classifier_using/
├── data_processor/data_preprocessing.py   # SubjectBundleDataset with padding_mask
├── models/meta_mobile_vit.py              # MetaMobileViTClassifier (monkey-patched self-attn)
├── model_trainers/meta_trainer.py         # MetaTrainer with mask plumbing
├── evaluators/monte_carlo_evaluator.py    # Youden's J threshold applied on test set (also a leak)
└── detection_caller/detection_caller.py   # `_meta` run-id
```

## Reports

- `outputs/reports/meta_classifier_using/run_20260531_002841_meta/analysis.md`

## Related documentation

- `PROBLEM.md` / `PROBLEM.ko.md` — the strict-parity analysis that arose from investigating this candidate's aftermath
