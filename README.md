# VOG-MCI Detection — Dual-Pipeline Ablation Study

Edge AI pipeline for detecting **Mild Cognitive Impairment (MCI)** from **Video Oculography (VOG)** saccade tracking errors. Deployed on **NVIDIA Jetson AGX Orin 64GB**.

The project is structured as a side-by-side comparison of two pipelines that share the same model architecture (frozen MobileViT-small + small trainable head) and same evaluation protocol (subject-level 30-fold Monte-Carlo Group CV), but differ in **which subset of the 8 VOG saccade tasks they feed the model**. A common task-contribution probe lets us measure each task's marginal effect on detection.

---

## 1. Project Goal

Build the smallest training recipe that beats the **66 % majority-class prior** on a held-out subject set. Concretely we need to know:

1. Does **isolating the highest-cognitive-load paradigm** (Anti-Saccade B) maximise the clinical signal? (the doctor's hypothesis)
2. Or does **feeding all 8 saccade paradigms** to a task-conditioned model — more data, more diversity — beat the isolation?
3. Of the 8 tasks, **which actually contribute** to the subject-level vote, and by how much?

The dual-pipeline layout exists to answer (1) vs (2) by direct comparison, and the probe answers (3).

---

## 2. Repository Layout

```
VOGBasedMCIDetectionSystemMobileViTSingleExperimentInference/
├── data/                                  ← symlink to subject CSVs (HC_csv_24_25 / MCI_csv_25_26 / MCI_plus_csv)
├── src/
│   ├── paths.py                           ← shared: absolute paths anchored to PROJECT_ROOT
│   ├── probe_generators/                  ← shared: TaskWiseProbeGenerator (multi-grouping contribution probe)
│   │   └── probe_generator.py
│   ├── detection_caller/                  ← top-level dispatcher (pure subprocess shim)
│   │   └── detection_caller.py
│   ├── two_experiments_using/             ← 2-task pipeline: Anti-Saccade B only
│   │   ├── data_processor/data_engineering.py
│   │   ├── models/mobile_vit_model.py
│   │   ├── model_trainers/mobile_vit_trainer.py
│   │   ├── evaluators/monte_carlo_evaluator.py
│   │   └── detection_caller/detection_caller.py
│   └── full_experiments_using/            ← 8-task pipeline: all saccade paradigms
│       ├── data_processor/data_engineering.py
│       ├── models/mobile_vit_model.py
│       ├── model_trainers/mobile_vit_trainer.py
│       ├── evaluators/monte_carlo_evaluator.py
│       └── detection_caller/detection_caller.py
├── outputs/                               ← auto-created on first run
│   ├── cache/                             ← preprocessing caches (config-keyed pickles)
│   │   ├── data_store.pkl                 ← 2-exp tensors
│   │   └── data_store_full.pkl            ← 8-exp tensors (separate key)
│   ├── checkpoints/run_<id>/              ← per-fold best-metric .pth files
│   ├── logs/run_<id>.log                  ← stream + file logging
│   ├── reports/                           ← analysis markdown + confusion-matrix PNGs
│   │   ├── INDEX.md
│   │   ├── run_<id>/                      ← one folder per run
│   │   │   ├── (intermediate_)analysis.md
│   │   │   └── confusion_matrix.png
│   │   └── legacy/                        ← superseded artifacts
│   └── legacy/                            ← legacy .pth from earlier 8-task work
├── requirements.txt
└── README.md
```

The two sub-packages (`two_experiments_using/` and `full_experiments_using/`) are **file-for-file symmetric**. Each contains its own `data_processor`, `models`, `model_trainers`, `evaluators`, and `detection_caller` modules with the same public class names — but each parameterises them for its own scope.

---

## 3. The Two Pipelines

### 3.1 `two_experiments_using/` — Anti-Saccade B Isolation (2 tasks)

| Aspect | Value |
|---|---|
| Task filter | `Horizontal Saccade B (anti)` → ID 0, `Vertical Saccade B (anti)` → ID 1 — every other file dropped at the data layer |
| Target inversion | Unconditional `target *= -1` (filter guarantees only anti-saccade files reach this step) |
| Task embedding | `Embedding(num_tasks=2, embedding_dim=16)` → CosineLinear concat dim 656 |
| Default dropout | **0.5** (aggressive regularization for the tiny train fold) |
| Default patience | **40** on val-loss |
| LR schedule | CosineAnnealingLR (T_max = max_epochs) |
| Best-checkpoint metric | val-loss |

**Advantage relative to project goal:**
- Tests the clinical hypothesis directly — anti-saccade is the paradigm the literature flags as most discriminative for MCI.
- Smaller model (16-dim task embedding vs 32) and simpler decision space (binary vs 8-way task conditioning).
- Faster per-fold wall time — ~120–160 training windows per fold means each epoch is cheap.

**Trade-off observed in practice:**
- Filtering out 6 of 8 task types is **4× data starvation** per fold. Best val-loss is often achieved in the first 1–10 epochs (before training stabilises), and the model frequently collapses to a single-class predictor. The completed 30-fold BASE run yielded Macro AUROC ≈ 0.39 — below random.

### 3.2 `full_experiments_using/` — Full 8-Task Pipeline (legacy origin)

| Aspect | Value |
|---|---|
| Task filter | All 8 saccade paradigms (A / B / B-anti / R, both axes) |
| Target inversion | Conditional — only when `"(anti)" in clean_task` |
| Task embedding | `Embedding(num_tasks=8, embedding_dim=32)` → CosineLinear concat dim 672 |
| Default dropout | **0.3** (preserves the legacy 8-task recipe) |
| Default patience | **40** on val-AUROC |
| LR schedule | Linear warmup (5 epochs) + ReduceLROnPlateau on val-AUROC (factor 0.2, patience 10) |
| Best-checkpoint metric | val-AUROC |
| Extras | grad-clip norm=1.0 |

**Advantage relative to project goal:**
- 4× more training data per fold — the frozen MobileViT backbone has enough examples to anchor a meaningful representation.
- Task embedding lets the model condition predictions per paradigm; this is what makes the multi-grouping probe (§5) informative.
- ReduceLROnPlateau + warmup are more forgiving than cosine schedule on a small noisy val set — fewer "best-at-epoch-1" collapses.

**Trade-off:**
- ~4× slower per fold; full 30-fold run takes hours-to-days depending on GPU contention.
- Includes data the doctors flagged as low-information (reflexive saccades), which may dilute the signal — *unless* the model's task embedding learns to down-weight those.

### 3.3 When to Run Which

| Goal | Pipeline | Flags |
|---|---|---|
| Test the clinical-isolation hypothesis | `two_experiments_using` | (default) |
| Get the best raw accuracy / AUROC | `full_experiments_using` | (default) |
| Identify which task carries the signal | `full_experiments_using` | (default); read the probe report's by-task / by-axis / by-type / by-inhibition sections |
| Compare aug vs no-aug | either pipeline | `--augment` |
| Ablate dropout | `full_experiments_using` | `--dropout 0.5` (or any value) |
| Ablate early-stopping patience | either pipeline | `--patience 30` |

---

## 4. Data Engineering (shared signal processing)

Both pipelines use the same CWT preprocessing.

### 4.1 Signal Processing

| Parameter | Value | Rationale |
|---|---|---|
| Wavelet | `pywt.cwt` with `cmor4.0-1.0` | Complex Morlet, bandwidth=4.0, center freq=1.0 Hz |
| Frequency range | 15 – 60 Hz | Hard high-pass; isolates oculomotor tremor band |
| Frequency bins | 32 (log-spaced) | Guarantees exact 8× integer upscale to 256 |
| Time bins | 32 | Same — exact 8× upscale to 256 |
| Pre / post stimulus | 0.2 s / 0.8 s | 1 s window centered on each target transition |
| Sampling rate | auto-detected per file (default 120 Hz) | |
| Artifact rejection | `|err| > 30.0` | Per-window threshold on both eyes |
| Channels | `[mag_L, re_L, mag_R, re_R]` | Binocular: sparsified magnitude + raw real, per eye |

**Output tensor geometry: `[4, 32, 32]` — fixed across both pipelines.**

### 4.2 Sparsification & Compression

```
1. Magnitude     →  M = sqrt(Re² + Im²)
2. Threshold     →  p85 = percentile(M, 85)
                    M[M < p85]  = 1e-3
                    M[M == 0.0] = 1e-3        (log-safety)
3. Saturated log →  S = 10 * log10(M)
4. Z-score       →  Z = (S − mean(S)) / (std(S) + 1e-8)
```

The 85th-percentile hard threshold creates the sparse, edge-like representation that the frozen MobileViT backbone reads as if it were a Sobel-filtered natural image.

### 4.3 Preprocessing Cache

Each pipeline writes its own cache file under `outputs/cache/`:
- 2-exp: `data_store.pkl` (~7.5 MB, 229 epochs)
- full-exp: `data_store_full.pkl` (~30 MB, ~900 epochs)

Cache is keyed by a config signature (pre/post-sec, freq range, bins, wavelet bandwidth, artifact threshold, task map). If any of those change, the cache invalidates and rebuilds. On Jetson: fresh build ~5–6 s; reload from cache <0.05 s.

---

## 5. Model Architecture (shared)

```
Input: [B, 4, 32, 32]
    │
    ▼
[Asymmetric Spatial Adapter]
    Conv2d(4→3, kernel=(5,1), padding=(2,0))    ← vertical Sobel-like edge filter
    BatchNorm2d(3) + ReLU
    Output: [B, 3, 32, 32]
    │
    ▼
[Nearest-Neighbor Upscale 8×]
    F.interpolate(..., size=(256,256), mode='nearest')
    Output: [B, 3, 256, 256]
    │
    ▼
[Frozen MobileViT Backbone]  apple/mobilevit-small
    GlobalAveragePool(last_hidden_state)
    Output: [B, 640]
    │
    ▼
[Task Embedding]
    2-exp:   Embedding(2, 16)   ← lookup by axis  ∈ {0, 1}
    full-exp: Embedding(8, 32)  ← lookup by task ∈ {0..7}
    │
    ▼
[Feature Fusion]
    torch.cat([vision_features, task_embedding], dim=1)
    2-exp:   [B, 656]
    full-exp: [B, 672]
    │
    ▼
[Metric Learning Head]
    Dropout(p=dropout)                              ← configurable per pipeline / flag
    CosineLinear(in=..., out=2, scale=10.0)
    Output: [B, 2]    (logits in cosine-similarity space)
```

The backbone is **always frozen** (`requires_grad=False`); only the adapter, task embedding, and CosineLinear head get gradients. Total trainable params: ~1.7 k for 2-exp, ~3 k for full-exp.

---

## 6. Evaluation Protocol (shared)

| Setting | Value |
|---|---|
| Splitter | `GroupShuffleSplit(n_splits=30, test_size=0.3, random_state=42)` keyed on `subject_id` |
| Leakage guard | `train_subjs.isdisjoint(test_subjs)` assertion per fold |
| Aggregation | Subject-level soft-voting: `subject_prob = mean(softmax(logits)[:, 1] over all of that subject's windows)`; threshold 0.5 |
| Primary metric | Accuracy > 0.66 (the MCI-majority prior) |
| Secondary metrics | Sensitivity, Specificity, AUROC, PPV (Precision), NPV, F1 (MCI) |
| Per-run artifacts | one folder under `outputs/reports/run_<id>/` with `(intermediate_)analysis.md`, `confusion_matrix.png`, and (after lifecycle ends) the probe report |

---

## 7. Task-Contribution Probe (the headline diagnostic)

After all 30 folds finish, `TaskWiseProbeGenerator.generate_markdown_report()` writes a multi-section markdown that slices subject-level metrics four ways simultaneously:

| Grouping (8-task) | Groups | Question |
|---|---|---|
| **by-task** | individual tasks 0–7 | which task alone classifies subjects best? Drop which task hurts the ensemble most? |
| **by-axis** | Horizontal {0,1,2,3} vs Vertical {4,5,6,7} | does the signal live on one eye-movement axis? |
| **by-type** | A {0,4} vs B {1,5} vs B-anti {2,6} vs R {3,7} | which saccade paradigm carries the signal? |
| **by-inhibition** | Reflexive {0,1,3,4,5,7} vs Anti-saccade {2,6} | does the *clinical hypothesis* hold — anti-saccades carry the most discriminative signal? |

For each group the probe computes:
- **Standalone metric** — subject-level metric using *only* that group's windows.
- **Leave-One-Out ΔAUROC** — `AUROC_full − AUROC_without_group` per fold. **Positive Δ means the group contributes positively to detection.**

A final **cross-grouping ranking** sorts every group (across every grouping) by mean ΔAUROC and suggests a weighting action (`up-weight 1.5×`, `keep 1.0×`, `neutral`, `down-weight 0.5×`, `drop`).

The 2-exp scope has only the `by-task` grouping (the other slicings are degenerate with 2 tasks).

---

## 8. Running the Project

### 8.1 Install

```bash
pip install -r requirements.txt
```

Key deps: `torch 2.11`, `transformers 5.9`, `pywavelets 1.8`, `scipy 1.15`, `scikit-learn 1.7`, `pandas 2.3`, `numpy 2.2`, `jetson-stats` (for `jtop` GPU telemetry).

### 8.2 CLI — three equivalent entry points

```bash
# (1) Top-level dispatcher (recommended)
python src/detection_caller/detection_caller.py [--full-experiments-using] [--augment] [--dropout F] [--patience N]

# (2) 2-experiment pipeline directly
python src/two_experiments_using/detection_caller/detection_caller.py [--augment] [--patience N]

# (3) Full-experiment pipeline directly
python src/full_experiments_using/detection_caller/detection_caller.py [--augment] [--dropout F] [--patience N]
```

The top-level dispatcher subprocess-launches the selected sub-package's entry point, forwarding the relevant flags. Each subprocess runs in its own Python interpreter — no shared `sys.modules`, no risk of cross-contamination, safe to run multiple concurrently.

### 8.3 Flag reference

| Flag | Applies to | Effect |
|---|---|---|
| (no flag) | dispatcher | runs `two_experiments_using` |
| `--full-experiments-using` | dispatcher | routes to `full_experiments_using` instead |
| `--augment` | both | wraps the train Subset in `AugmentedSubset` (freq+time SpecAugment-style masking, train-only). Run id suffixed `_aug`. |
| `--dropout F` | full-exp only | overrides head dropout. Run id suffixed `_dropNNN` when ≠ default 0.3. Dispatcher silently drops it (with warning) in 2-exp mode. |
| `--patience N` | both | early-stop patience on the monitored val metric. Default 40. Run id suffixed `_patNNN` when ≠ 40. |

### 8.4 Ablation recipes

```bash
# 2-exp BASE (clinical-hypothesis isolation)
python src/detection_caller/detection_caller.py

# 2-exp + augmentation
python src/detection_caller/detection_caller.py --augment

# 2-exp + shorter patience (matches legacy)
python src/detection_caller/detection_caller.py --patience 30

# Full-exp plain baseline
python src/detection_caller/detection_caller.py --full-experiments-using

# Full-exp + augmentation
python src/detection_caller/detection_caller.py --full-experiments-using --augment

# Full-exp + dropout 0.5 (stronger regularization)
python src/detection_caller/detection_caller.py --full-experiments-using --dropout 0.5

# Full-exp + dropout 0.5 + patience 30 (closer to legacy)
python src/detection_caller/detection_caller.py --full-experiments-using --dropout 0.5 --patience 30
```

Every combination produces a uniquely-tagged `run_id`, so concurrent runs never collide on log / checkpoint / report paths.

### 8.5 Live monitoring

```bash
tail -f outputs/logs/run_*.log
```

GPU telemetry on Jetson:

```bash
sudo /usr/bin/python3 -c "from jtop import jtop, time
with jtop() as j:
    if j.ok(): time.sleep(1); print(j.gpu, j.memory['RAM'], j.temperature)"
```

---

## 9. Configuration Knobs & Their Advantages

| Knob | Lower / Off | Higher / On | When to prefer |
|---|---|---|---|
| **Pipeline scope** (2-exp vs full-exp) | Clinical isolation, smaller model, faster | More data, learns per-task patterns, enables contribution probe | Use 2-exp to *test the hypothesis*; full-exp to *get the headline number* |
| **Augmentation** | No regularization at input | Train-only freq+time masking; smaller train-val gap | Always worth trying alongside the no-aug baseline |
| **Dropout** (head) | 0.3 (less regularization, more capacity) | 0.5 (aggressive regularization, prevents memorization) | 0.5 is safer on the small dataset (~3 k trainable params); 0.3 if dropout 0.5 over-regularises and prevents fitting |
| **Patience** (early-stop) | 30 (matches legacy; less drift past best-checkpoint) | 40 (more chances to find a deeper minimum) | 30 keeps the in-memory final-epoch model closer to the saved best-checkpoint; 40 explores more |
| **Reload best checkpoint at inference** | Off (default — uses in-memory final-epoch model) | On | Currently off by design — see §11 |

---

## 10. Known Behavioural Quirks

1. **Reported metrics use the final-epoch model, not the best-checkpoint.** The trainer saves the best-val-metric `.pth` per fold, but returns its in-memory model object — which is `patience` epochs past the best snapshot. This means **the `.pth` files on disk and the reported numbers don't refer to the same model**. By design (user-confirmed). The drift this causes scales with `--patience`.

2. **`GroupShuffleSplit` does not stratify by class.** Some folds draw very few HC or MCI subjects; macro Std on Sens/Spec is therefore large for the first several folds and shrinks as more folds finish.

3. **`N_eff` in the analysis tables is back-derived.** The evaluator doesn't log the raw per-fold confusion matrix; the analysis recovers TN/FP/FN/TP from `(Acc, Sens, Spec, N_test_subj)` by exhaustive integer search. Macro/pooled aggregates use the *logged* metrics directly — they're exact.

4. **No `--dropout` on 2-exp.** The 2-exp dropout is fixed at 0.5 by design (per the Gemini-inspired regularization decision documented in the dev log). The dispatcher's `--dropout` flag is silently dropped when routing to 2-exp.

5. **Same `random_state=42` across pipelines** — fold N is the same subjects in 2-exp and full-exp, so per-fold metrics are directly comparable (when both pipelines have completed that fold).

---

## 11. System Requirements

| Requirement | Spec |
|---|---|
| Evaluation protocol | 30-fold subject-grouped MC CV, 70/30 |
| Primary metric | Accuracy > 0.66 (majority prior) |
| Secondary metrics | Sensitivity, Specificity, AUROC, PPV, NPV, F1 |
| Edge target | NVIDIA Jetson AGX Orin 64GB (MAXN power profile) |
| Input tensor | `[4, 32, 32]` — fixed |
| Backbone | `apple/mobilevit-small` (frozen) |
| Concurrent runs supported | Yes — verified up to 4 simultaneous, each in its own Python subprocess |
