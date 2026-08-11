# System Outline — final decisions

Conclusionary configuration of the VOG-based MCI detection system. (Full run tables + confusion
matrices: [`outputs/reports/RESULTS.md`](outputs/reports/RESULTS.md) and [`key_results/`](key_results).)

## Pipeline (adopted)
1. **Input representation — `four_error` (4-channel, magnitude, both axes).** Per event-locked
   1 s window, the CWT **magnitudes** of the four eye-vs-target errors
   `[ |CWT(LH−TH)|, |CWT(RH−TH)|, |CWT(LV−TV)|, |CWT(RV−TV)| ]` → `[4, 32, 32]`.
   *(Chosen over legacy task-axis `mag+re` and over the 8-channel `full_error`.)*
2. **Model.** Conv adapter `4→3` → nearest-upscale to 256² → **frozen MobileViT-small** → mean-pool
   (640-d) ⊕ **task embedding** (32-d) → `CosineLinear(672→2)` head. Only adapter + embedding +
   head train (1,669 params); backbone frozen. Loss: **class-weighted cross-entropy**.
3. **Evaluation — stratified 30-fold subject-grouped CV.** Fixed **HC=4 / MCI=8** test subjects per
   fold (balanced), random membership; 37 subjects (14 HC / 23 MCI), artifact threshold 30°.
4. **Aggregation — per-subject task-weighted soft vote.** `P_s = Σ_i w[t_i]·p_i / Σ_i w[t_i]`,
   threshold 0.5. Weights are a per-task scheme (inference only; does not affect training).

## Adopted weight schemes
- **default** `[0, .5, .5, .5, 0, 1.5, 1.5, 1.5]` — A tasks excluded, Vertical 3× Horizontal.
- **larger-to-Saccade-R** `[0, .5, .5, 1, 0, 1.5, 1.5, 3]` — Vertical R weighted highest (3.0).

## Headline results (threshold 30, 30-fold)
| Configuration | AUROC | Sens | Spec |
|---|---|---|---|
| **four_error · stratified · default** | **0.871** | 0.708 | 0.711 |
| four_error · stratified · larger-to-R | 0.864 | **0.809** | 0.675 |
| four_error · non-stratified · default | 0.804 | 0.708 | 0.715 |
| legacy (task-axis mag+re) · default | 0.791 | 0.788 | 0.599 |
| full_error (8-channel) · default | 0.767 | 0.691 | 0.722 |

## Decisions
- **Adopt `four_error` (both-axis magnitude, 4ch) + stratified folds.** Best AUROC (~0.87), beats
  legacy (0.791).
- **Reject `full_error` (8ch, +real part):** overfits on 37 subjects (0.767).
- **Stratified sampling is standard:** raises AUROC ~0.05–0.07 and cuts fold variance vs grouped.
- Use the **larger-to-R** scheme when higher sensitivity is preferred (0.809 vs 0.708).

## Run it
```bash
python src/four_error_using/detection_caller/detection_caller.py \
    --signal-mode four_error --stratified --artifact-threshold 30 \
    --vote-weights "0.0 0.5 0.5 1.0 0.0 1.5 1.5 3.0"     # or --weighted-vote for the default scheme
```
