# Zero-Trial Analysis

**Cache state:** `outputs/cache/data_store_meta_4err.pkl`
**Artifact threshold:** **45.0°** (events with `max|eye − target| > threshold` are rejected)
**Subjects scanned:** 37

## Current state (at the 45.0° threshold)

- **Subjects with ≥1 zero-trial task:** **0**
- **Total zero-trial (subject, task) pairs:** **0**
- See `task_zero_trial_summary.csv` (per-task counts; all zeros if no zero-trial cells)
- See `subjects_with_zero_trials.csv` (per-subject details; empty if no affected subjects)

## How this changed across the 30°→45° threshold rebuild

| | 30° threshold (historical) | 45° threshold (current) |
|---|:--:|:--:|
| Total kept events | 5,736 | **9,378** (+63%) |
| Subjects with ≥1 zero-trial task | **14** | **0** |
| Total zero-trial (subject, task) pairs | 18 | **0** |
| HSacBanti zero-trial subjects | 8 | 0 |
| VSacBanti zero-trial subjects | 8 | 0 |
| HSacB zero-trial subjects | 1 | 0 |
| VSacB zero-trial subjects | 1 | 0 |

The 30°→45° change rescued the Hypermetria-overshoot events that the original threshold had been trapping. Each of the 14 previously-zero-trial subjects now has ≥1 (typically 5–29) trials of the tasks they were missing. The 30°-trap analysis is preserved in this directory's git history; the current CSVs reflect the post-rebuild state.

## Source documents

- `CODE_AUDIT_VS_MASTER_CONTEXT.md` — the audit that drove the 30°→45° change
- `TASK_CONTRIBUTION_RATES.md` — empirical per-task contribution data from the wvote probe
- `PROBLEM.md` — original strict-parity analysis (now superseded by Solution D + 45° rebuild)
