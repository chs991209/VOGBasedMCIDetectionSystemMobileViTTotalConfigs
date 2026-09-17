# VOG-MCI Detection — Results & Methods

_Auto-generated from completed run logs. **Only successful (30/30-fold) runs are included; halted/mid-training runs are excluded** (listed at the bottom)._

## 1. What the system does

Event-locked continuous-wavelet (CWT) scalograms of eye-vs-target gaze error feed a **task-conditioned, frozen-MobileViT** classifier scored per window; a per-subject **task-weighted soft vote** produces the HC-vs-MCI probability. 37 subjects (14 HC / 23 MCI), subject-grouped 30-fold cross-validation.

## 2. Code: representations, flags, sampling

Entry point: `src/four_error_using/detection_caller/detection_caller.py`.

**Signal representation** (`--signal-mode`), the CWT channels fed to the model:

| Mode | Channels | Meaning |
|---|---|---|
| `legacy` (default) | 4 | `[mag_L, re_L, mag_R, re_R]` — task-axis only, per eye (magnitude + real part) |
| `four_error` | 4 | `[|CWT(LH-TH)|, |CWT(RH-TH)|, |CWT(LV-TV)|, |CWT(RV-TV)|]` — both axes, magnitude only |
| `full_error` | 8 | both axes × both eyes × (magnitude, real) |

**Sampling** (`--stratified`): default = grouped random 70/30 subject splits (`GroupShuffleSplit`, class ratio varies per fold); `--stratified` = fixed HC=4/MCI=8 test subjects per fold (balanced), random membership.

**Aggregation weights** (`--vote-weights "w0 … w7"` or `--weighted-vote` for the built-in scheme): per-task weights applied in the subject-level soft vote `P_s = Σ_i w[t_i]·p_i / Σ_i w[t_i]` (inference only; does not affect training).

**Other flags:** `--artifact-threshold` (gaze-error rejection, deg), `--batch-size`, `--dropout`, `--patience`. A6000: TF32 + cuDNN autotune + pinned transfers enabled.

## 3. Results (completed runs, threshold 30, 30-fold)

**Legacy baseline (task-axis mag+re, default scheme): AUROC 0.791, Sens 0.788, Spec 0.599.**

### 3.1 four_error (4-channel, both-axis magnitude)

| Vote scheme | Sampling | Accuracy | Sensitivity | Specificity | AUROC | Confusion |
|---|---|---|---|---|---|---|
| default (0·.5·.5·.5·0·1.5·1.5·1.5) | grouped | 0.704 ± 0.143 | 0.708 ± 0.215 | 0.715 ± 0.217 | **0.804 ± 0.149** | [cm](four_error_using/grouped/default/confusion_matrix.png) |
| 0 .5 1 1 0 1 2 2 | grouped | 0.694 ± 0.145 | 0.717 ± 0.196 | 0.677 ± 0.265 | **0.817 ± 0.133** | [cm](four_error_using/grouped/w0-5-10-10-0-10-20-20/confusion_matrix.png) |
| 0 .5 1 1 0 1.5 3 3 | grouped | 0.710 ± 0.136 | 0.788 ± 0.198 | 0.606 ± 0.229 | **0.812 ± 0.120** | [cm](four_error_using/grouped/w0-5-10-10-0-15-30-30/confusion_matrix.png) |
| 0 .5 .5 1 0 1.5 1.5 3 | grouped | 0.747 ± 0.122 | 0.823 ± 0.163 | 0.655 ± 0.257 | **0.836 ± 0.122** | [cm](four_error_using/grouped/w0-5-5-10-0-15-15-30/confusion_matrix.png) |
| default (0·.5·.5·.5·0·1.5·1.5·1.5) | stratified | 0.708 ± 0.159 | 0.708 ± 0.250 | 0.711 ± 0.298 | **0.871 ± 0.099** | [cm](four_error_using/stratified/default/confusion_matrix.png) |
| 0 .5 1 1 0 1 2 2 | stratified | 0.738 ± 0.143 | 0.732 ± 0.215 | 0.753 ± 0.220 | **0.876 ± 0.102** | [cm](four_error_using/stratified/w0-5-10-10-0-10-20-20/confusion_matrix.png) |
| 0 .5 1 1 0 1.5 3 3 | stratified | 0.770 ± 0.129 | 0.807 ± 0.205 | 0.697 ± 0.231 | **0.876 ± 0.101** | [cm](four_error_using/stratified/w0-5-10-10-0-15-30-30/confusion_matrix.png) |
| 0 .5 .5 1 0 1.5 1.5 3 | stratified | 0.765 ± 0.137 | 0.809 ± 0.209 | 0.675 ± 0.271 | **0.864 ± 0.113** | [cm](four_error_using/stratified/w0-5-5-10-0-15-15-30/confusion_matrix.png) |

### 3.2 full_error (8-channel, both-axis magnitude + real)

| Vote scheme | Sampling | Accuracy | Sensitivity | Specificity | AUROC | Confusion |
|---|---|---|---|---|---|---|
| default (0·.5·.5·.5·0·1.5·1.5·1.5) | grouped | 0.694 ± 0.133 | 0.691 ± 0.225 | 0.722 ± 0.244 | **0.767 ± 0.158** | [cm](archive/full_error_using/grouped/default/confusion_matrix.png) |

## 4. Findings

- **four_error (both-axis, 4ch) beats legacy** (0.791) across all schemes.
- **Stratified folds raise AUROC and cut its variance** (four_error ~0.80–0.84 grouped → ~0.86–0.88 stratified) by removing folds with too few HC subjects.
- **full_error (8ch) is worst (0.767)** — adding real-part channels overfits (37 subjects).
- **Best configuration: four_error + stratified, AUROC ≈ 0.876.**

## 5. Confusion matrices

One aggregate 2×2 matrix per study run (over 30 folds, ~360 subject-predictions), organised as `outputs/reports/{mode}_using/{sampling}/{scheme}/confusion_matrix.png`; linked in the tables above.

## 6. Excluded (halted) runs

These runs were stopped mid-training and are **not** in the results above:

- `20260907_163415_full_wvote_artifact30_strat_4err_leftover`
- `20260907_163739_full_wvote_artifact30_strat_4err_leftover`
