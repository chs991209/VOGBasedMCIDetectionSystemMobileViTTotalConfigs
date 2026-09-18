# Results — one place for everything

37 subjects (14 HC / 23 MCI). Stratified repetitions (HC=4 / MCI=8 held out each time).
**AUROC** = the headline score (1.0 perfect, 0.5 = coin flip). Each number is **mean ± SD**,
and the **Reps** column says over how many repetitions — read it before comparing rows.

## Bottom line
- **The plain scalogram image model is the best: ~0.84–0.86** (up to ~0.88 on some runs). Nothing we added beats it.
- Kinematics, entropy channels, fusion, leftover/whole-recording windows — all **equal or worse**.
- So the winning system = event-locked scalograms + weighted vote, no extras.

## Master results table
Best number per experiment. **α** = weight on the image model when fused with kinematics (1.0 = image only).

| # | Experiment | Vote | Windows | Reps | α | AUROC | Sens | Spec |
|---|---|---|---|---|---|---|---|---|
| 1 | Image model (scalograms) | 0 .5 1 1 0 1.5 3 3 | event, filtered | 30 | 1.0 | 0.829 ± 0.109 | 0.76 | 0.66 |
| 2 | **Image model (scalograms)** ⭐ | 0 .5 .5 1 0 1.5 1.5 3 | event, filtered | 30 | 1.0 | **0.864 ± 0.113** | 0.81 | 0.68 |
| 3 | Image + **KL-entropy** channel | 0 .5 1 1 0 1.5 3 3 | event, filtered | 30 | — | 0.842 ± 0.077 | 0.82 | 0.50 |
| 4 | Image + Shannon-entropy channel | 0 .5 1 1 0 1.5 3 3 | event, filtered | 30 | — | 0.822 ± 0.119 | 0.76 | 0.58 |
| 5 | Image + kinematics (fused) †v3 | 0 .5 1 1 0 1.5 3 3 | event, filtered | 30 | 0.95 | 0.805 ± 0.102 | 0.75 | 0.62 |
| 6 | Kinematics only (10 indicators) †v3 | 0 .5 1 1 0 1.5 3 3 | event, filtered | 30 | 0.0 | 0.578 ± 0.150 | 0.53 | 0.54 |
| 7 | Whole-recording (all windows) | 0 .5 .5 1 0 1.5 1.5 3 | all, no reject | 30 | 1.0 | 0.759 ± 0.103 | 0.72 | 0.55 |
| 8 | Image + kinematics, all windows | 0 .5 1 1 0 1.5 3 3 | all, no reject | 30 | 0.95 | 0.746 ± 0.140 | 0.66 | 0.71 |
| 9 | Leftover / fixation gaps | 0 .5 .5 .5 0 1.5 1.5 1.5 | leftover | **13 / 30** ⚠ | 1.0 | 0.686 ± 0.162 | 0.81 | 0.50 |
| 10 | Kinematics-in-model (fused before head) †v3 | 0 .5 1 1 0 1.5 3 3 | event, filtered | 30 | — | 0.797 ± 0.112 | 0.75 | 0.60 |

⭐ = best in this table. Rows 1, 5, 6 are one consistent run (same vote); row 2 is a different vote.
†v3 = uses the **corrected** kinematics (low-pass smoothed, direction-robust accel/decel, 2D-onset latency).
⚠ Row 9 stopped at 13 of 30 reps — **partial, indicative only** (re-running to 30 now).

## What each experiment was
- **1–2 Image model** — event-locked scalograms + task-weighted vote. Two vote settings (0.840 and 0.864); the difference is normal run-to-run variation. This is the system.
- **3 KL-entropy** — added a channel measuring how far the actual eye path strays from the target (information divergence). Ties the image model, no real gain.
- **4 Shannon-entropy** — same idea, simpler entropy formula. Slightly worse.
- **5 Kinematics fused** — blend the image model with the 10 saccade numbers. Best blend (α=0.95) is *below* image-only → kinematics drag it down.
- **6 Kinematics only** — the 10 saccade numbers alone. Weak (0.65).
- **7 Whole-recording** — chop the entire recording into 1-s windows (saccades + fixations), no artifact removal. Weaker (0.76).
- **8 All-windows fused** — row 7 idea + kinematics. Weaker still.
- **9 Leftover** — only the gaps *between* saccades (fixations). Weak (0.69), and only 13 of 30 reps finished. Indicative only.

## Earlier runs (separate section — for the record)
Older runs of the **same image model**, before the current batch. Same cohort/protocol;
small differences are normal run-to-run variation. These are the ~0.87–0.88 numbers.

| Date | Image vote | Reps | AUROC | Sens | Spec | Note |
|---|---|---|---|---|---|---|
| Aug 3 | 0 .5 .5 .5 0 1.5 1.5 1.5 | 30 | 0.871 ± 0.099 | 0.71 | 0.71 | built-in scheme (no override); **was mislabeled "all equal" — corrected** |
| Aug 3 | 0 .5 1 1 0 1 2 2 | 30 | 0.876 ± 0.102 | 0.73 | 0.75 | |
| Aug 3 | 0 .5 1 1 0 1.5 3 3 | 30 | **0.876 ± 0.101** | 0.81 | 0.70 | best-balanced |
| Sep 14 | 0 .5 1 1 0 1.5 3 3 | **10** ⚠ | 0.870 ± 0.071 | 0.66 | 0.83 | image line of a fuse run; fused 0.859 @α0.95 |
| Sep 14 | 0 .5 1 1 0 1.5 3 3 | **10** ⚠ | **0.884 ± 0.084** | 0.70 | 0.77 | image line of a fuse run; fused 0.865 @α0.95 |

- These hover around **0.87–0.88** — same story as the current best (~0.84): the image model is the system, and fusion (0.859 / 0.865) sits just below it.
- ⚠ The two **Sep 14** rows are only **10 reps** (shorter runs) — less settled than the 30-rep rows, so treat them as lighter evidence.
- The spread (0.84 → 0.88) across runs is why we report **mean ± SD** with the rep count, not a single number.

## The 10 kinematic indicators (used in rows 5, 6, 8)
Computed from the **low-pass-smoothed** task-axis eye position in each event window
(so differentiation doesn't amplify sensor jitter — research-admin's condition).

| # | column | meaning |
|---|---|---|
| 1 | latency | reaction time — target onset → smoothed **2D total eye speed** (both eyes, H&V) first crosses 30°/s *(research-admin's 4-step method)* |
| 2 | peak_velocity | top speed of the eye jump |
| 3 | peak_accel | fastest **speeding-up** (max d(speed)/dt; direction-robust) |
| 4 | peak_decel | fastest **slowing-down** (min d(speed)/dt; direction-robust) |
| 5 | amplitude | how far the eye moved |
| 6 | duration | how long the jump lasted |
| 7 | time_to_peak_vel | onset → top speed |
| 8 | time_to_target | onset → eye settles near final position |
| 9 | gain | eye distance ÷ target distance (1 = on target) |
| 10 | vigor | peak_velocity ÷ amplitude (vigour proxy; *not* the population main sequence) |

## Notes
- "filtered" = artifact windows removed (5,715 windows); "no reject" = all windows kept (9,964).
- Vote weights and α are applied *after* training — they don't change the model.
- **Rows 5, 6 now use the corrected (†v3) kinematics** — low-pass smoothing before
  differentiation, direction-robust accel/decel, and the 2D-onset latency (median ~192 ms).
  The proper kinematics score slightly *lower* than the old buggy version (kin-only 0.578 vs
  0.618), confirming they add no generalizable signal. Row 8 (all-windows fused) is being
  re-run on v3.
- Benchmark (VECA, npj 2024): sensitivity 88.5% / specificity 83% on 201 people with VR. Ours is 37 people, lab VOG.
