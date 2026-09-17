# Kinematic gate × CWT late fusion — results

**Plan (revised):** the kinematic gate uses **all three feature groups —
velocity (high, low) + latency + variance** (the velocity-only variant is
dropped). Late fusion at the probability/vote stage:
`P_fused = α·P_cwt + (1−α)·P_kin`.

- **P_cwt** — the four_error CWT weighted-vote probability, obtained by inference
  from the saved per-fold checkpoints (`stratified/default`), on the exact same
  `RandomState(42)` folds. No CWT retraining.
- **P_kin** — logistic on the task-weighted per-subject kinematic vector
  `[v_high, v_low, latency, v_var]`, trained per fold on the train subjects.
- 37 subjects (14 HC / 23 MCI), stratified 30-fold (HC=4 / MCI=8 test), artifact 30°.

## Reference points
| System | AUROC | Sens | Spec | Acc |
|---|---|---|---|---|
| CWT only (checkpoint inference) | 0.856 ± 0.092 | 0.821 | 0.667 | 0.769 |
| Kinematic only (all 3 features) | 0.614 ± 0.126 | 0.675 | 0.458 | 0.603 |

## [1] Gated fusion 1:1 (α = 0.5), all 3 features
| AUROC | Sens | Spec | Acc |
|---|---|---|---|
| **0.725 ± 0.141** | 0.708 | 0.492 | 0.636 |

Equal weight lets the weak gate (0.61) drag down the strong CWT (0.86).

## [2] Unbalanced ratio sweep (α = weight on CWT), all 3 features
| α (CWT : KIN) | AUROC | Sens | Spec | Acc |
|---|---|---|---|---|
| 0.50 (1:1) | 0.725 ± 0.141 | 0.708 | 0.492 | 0.636 |
| 0.60 | 0.757 ± 0.139 | 0.758 | 0.517 | 0.678 |
| 0.70 | 0.800 ± 0.131 | 0.796 | 0.558 | 0.717 |
| 0.75 | 0.823 ± 0.120 | 0.812 | 0.583 | 0.736 |
| 0.80 | 0.849 ± 0.102 | 0.821 | 0.583 | 0.742 |
| 0.85 | 0.872 ± 0.085 | 0.833 | 0.625 | 0.764 |
| **0.90** | **0.874 ± 0.080** | 0.846 | 0.592 | 0.761 |
| 0.95 | 0.872 ± 0.082 | 0.817 | 0.642 | 0.758 |

**Best fused AUROC = 0.874 at α = 0.90** (CWT 90% / kinematics 10%), marginally
above CWT-only (0.856) but within one standard deviation.

## Read
- The kinematic gate helps only as a **light-touch** term (~10% weight); at
  balanced weight it dilutes the CWT.
- The gain over CWT-only is small and not decisive on 37 subjects.
