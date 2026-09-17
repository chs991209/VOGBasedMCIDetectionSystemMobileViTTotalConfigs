# Leftover-regions experiment — results (PARTIAL)

**What it is:** the "not-used regions" system. Instead of the event-locked windows
(around each saccade), it takes 1-second windows from the stretches **left over**
between kept windows — the inter-saccade / fixation gaps — and runs the same
four_error CWT scalograms + image model + weighted vote on them.

**Status: incomplete — stopped at 13 of 30 repetitions.** (Run
`run_20260907_163739_..._strat_4err_leftover`, artifact thr 30°, stratified
HC=4/MCI=8 test, vote 0 .5 .5 .5 0 1.5 1.5 1.5.)

## Partial result (13 repetitions)
| Metric | Value |
|---|---|
| AUROC | **0.686 ± 0.162** |
| Accuracy | 0.718 |
| Sensitivity | 0.808 |
| Specificity | 0.500 |

## Read
- Much weaker than the main **event/used-window** system (~0.87). The fixation /
  inter-saccade gaps carry far less HC-vs-MCI signal than the saccades themselves.
- High sensitivity / low specificity (0.81 / 0.50): it over-calls MCI — the leftover
  windows don't separate the classes well.

## Note
This is a **partial** number (13/30) — treat as indicative only. Re-run to 30
repetitions for a final figure.
