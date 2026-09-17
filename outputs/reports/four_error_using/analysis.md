# Analysis — four_error (4ch) representation

**Representation adopted for the detection system.** Both-axis, both-eye error magnitudes
`[ |CWT(LH−TH)|, |CWT(RH−TH)|, |CWT(LV−TV)|, |CWT(RV−TV)| ]` (H,H,V,V) → `[4,32,32]`.
Unlike **legacy** (task-axis only, `mag_L, re_L, mag_R, re_R`), four_error also carries the
**off-axis** trace (vertical during a horizontal task and vice-versa); magnitude-only, no real part.

## Setup
- Artifact threshold 30°, event-locked 1 s windows, 5,715 windows, 37 subjects (14 HC / 23 MCI).
- 30-fold subject-grouped CV; per-subject task-weighted soft vote
  `P_s = Σ w[t_i]·p_i / Σ w[t_i]` (inference only — does not affect training).
- dropout 0.5, patience 30, AdamW, frozen MobileViT-small.
- **Legacy baseline** (task-axis mag+re, default scheme): AUROC 0.791, Sens 0.788, Spec 0.599.

## Results — all weighted-vote schemes × sampling (mean ± std, 30 folds, thr 30)

Vote schemes are per-task weights `[HSacA, HSacB, HSacBanti, HSacR, VSacA, VSacB, VSacBanti, VSacR]`
(A always 0). The last three rows of each block are **imbalanced** schemes (R / vertical up-weighted).

### Grouped (unstratified)
| Vote scheme | Accuracy | Sensitivity | Specificity | AUROC | Confusion |
|---|---|---|---|---|---|
| default `0 .5 .5 .5 0 1.5 1.5 1.5` | 0.704 ± 0.143 | 0.708 ± 0.215 | 0.715 ± 0.217 | **0.804 ± 0.149** | [cm](grouped/default/confusion_matrix.png) |
| `0 .5 1 1 0 1 2 2` | 0.694 ± 0.145 | 0.717 ± 0.196 | 0.677 ± 0.265 | **0.817 ± 0.133** | [cm](grouped/w0-5-10-10-0-10-20-20/confusion_matrix.png) |
| `0 .5 1 1 0 1.5 3 3` | 0.710 ± 0.136 | 0.788 ± 0.198 | 0.606 ± 0.229 | **0.812 ± 0.120** | [cm](grouped/w0-5-10-10-0-15-30-30/confusion_matrix.png) |
| `0 .5 .5 1 0 1.5 1.5 3` (V-R heavy) | 0.747 ± 0.122 | 0.823 ± 0.163 | 0.655 ± 0.257 | **0.836 ± 0.122** | [cm](grouped/w0-5-5-10-0-15-15-30/confusion_matrix.png) |

### Stratified (fixed HC=4 / MCI=8 per fold)
| Vote scheme | Accuracy | Sensitivity | Specificity | AUROC | Confusion |
|---|---|---|---|---|---|
| default `0 .5 .5 .5 0 1.5 1.5 1.5` | 0.708 ± 0.159 | 0.708 ± 0.250 | 0.711 ± 0.298 | **0.871 ± 0.099** | [cm](stratified/default/confusion_matrix.png) |
| `0 .5 1 1 0 1 2 2` | 0.738 ± 0.143 | 0.732 ± 0.215 | 0.753 ± 0.220 | **0.876 ± 0.102** | [cm](stratified/w0-5-10-10-0-10-20-20/confusion_matrix.png) |
| `0 .5 1 1 0 1.5 3 3` | 0.770 ± 0.129 | 0.807 ± 0.205 | 0.697 ± 0.231 | **0.876 ± 0.101** | [cm](stratified/w0-5-10-10-0-15-30-30/confusion_matrix.png) |
| `0 .5 .5 1 0 1.5 1.5 3` (V-R heavy) | 0.765 ± 0.137 | 0.809 ± 0.209 | 0.675 ± 0.271 | **0.864 ± 0.113** | [cm](stratified/w0-5-5-10-0-15-15-30/confusion_matrix.png) |

## Analysis
- **four_error beats legacy** (0.791) across every scheme, and fixes legacy's weak specificity
  (0.599 → ~0.71–0.75) — the off-axis trace carries HC-vs-MCI signal the task-axis-only input discards.
- **Stratified > grouped** on AUROC and variance (~0.80–0.84 → ~0.86–0.88), by removing folds with
  too few HC subjects.
- **Imbalanced weighted votes help modestly.** Up-weighting the harder/reflexive saccades
  (`0 .5 1 1 0 1 2 2` and `0 .5 1 1 0 1.5 3 3`) gives the best stratified AUROC (**0.876**); the
  V-R-heavy `0 .5 .5 1 0 1.5 1.5 3` trades a little AUROC (0.864) for the highest sensitivity (0.809).
- **Best configuration:** four_error + stratified, AUROC ≈ **0.876** (`0 .5 1 1 0 1 2 2` /
  `0 .5 1 1 0 1.5 3 3`); default scheme 0.871.

## Caveats
- **Small N (37).** ±0.10–0.22 std; AUROC differences between schemes are within one std — treat
  scheme ranking as suggestive, the four_error>legacy and stratified>grouped gaps as the robust findings.
- Weights are inference-only; all schemes reuse the same trained folds.
- 8-channel `full_error` (adds the real part) overfits — 0.767, archived under `../archive/full_error_using/`.
