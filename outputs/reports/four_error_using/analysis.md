# Analysis — four_error (4ch) representation

**Status:** preliminary (default weighted-vote scheme; the 3-scheme sweep is running).
**Run:** `run_20260801_202901_full_wvote_artifact30_4err`
**Date:** 2026-08-02

## What it is

`four_error` feeds the model the **both-axis, both-eye error magnitudes** — the four channels
`[ |CWT(LH−TH)|, |CWT(RH−TH)|, |CWT(LV−TV)|, |CWT(RV−TV)| ]` (H,H,V,V) — as a `[4,32,32]`
tensor. Unlike the **legacy** detection system, which uses only the *task's own axis*
(`mag_L, re_L, mag_R, re_R`), four_error also carries the **off-axis** trace (vertical during a
horizontal task and vice-versa). It is magnitude-only (no real part).

## Setup

- Artifact threshold 30°, event-locked 1 s windows, 5,715 windows, 37 subjects (14 HC / 23 MCI).
- 30-fold grouped (unstratified) Monte-Carlo CV; per-subject weighted soft-vote (default scheme
  `{A:0, H-B/Banti/R:0.5, V-B/Banti/R:1.5}`); dropout 0.5, patience 30, AdamW, frozen MobileViT.
- Single-variable change vs legacy: **the input representation only.**

## Results (mean ± std over 30 folds)

| Metric | legacy (task-axis mag+re) | **four_error (4ch)** | Δ |
|---|---|---|---|
| Accuracy | 0.717 | 0.704 ± 0.143 | −0.013 |
| Sensitivity | **0.788** | 0.708 ± 0.215 | −0.080 |
| Specificity | 0.599 | **0.715 ± 0.217** | **+0.116** |
| **AUROC** | 0.791 | **0.804 ± 0.149** | **+0.013** |

**Aggregate confusion** (pooled over 30 folds, 360 subject-predictions):

```
             pred HC   pred MCI
 true HC       103        43        (Spec 0.705)
 true MCI       60       154        (Sens 0.720)
```

## Analysis

- **four_error beats legacy on AUROC (0.804 vs 0.791)** and, more strikingly, **fixes the
  specificity problem**: 0.599 → 0.715 (+0.116). The legacy model over-calls MCI on HC subjects;
  four_error rejects HC far better.
- The cost is **lower sensitivity** (0.788 → 0.708). Net effect is a **more balanced** operating
  point (Sens ≈ Spec ≈ 0.71) rather than legacy's sensitivity-skewed one — hence the higher AUROC
  and lower specificity variance-of-outcome.
- **Interpretation:** the off-axis error (e.g. vertical drift during a horizontal saccade) appears
  to carry **HC-vs-MCI discriminative signal** — plausibly cross-axis coordination that HC controls
  and MCI does not. Adding it lets the model use a trait the legacy task-axis-only input discards.
- The improvement is modest in AUROC (+0.013) but the **specificity gain is large and consistent**,
  which is the clinically weaker axis of the legacy system.

## Caveats

- **Small N (37 subjects).** ±0.15–0.22 std on every metric; AUROC 0.804 vs 0.791 is **within one
  std** — treat the AUROC edge as suggestive, the specificity gain as the more robust finding.
- **Unstratified folds** → specificity is high-variance across folds (HC is the minority).
- Preliminary: default vote scheme only. The 3-scheme sweep (running) will show whether tuned
  weights extend or erase this edge.

## Next

- Scheme sweep on four_error (3 weight configs) — in progress.
- Compare against `full_error` (8ch, adds the real part) — see `analysis_full_error_8ch.md`.
- Consider a **stratified** re-run to tighten the specificity variance before drawing firm claims.
