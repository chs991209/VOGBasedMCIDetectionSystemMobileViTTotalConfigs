# Code map — organized by usage

What each file is for, grouped by the job it does. Main system lives in
`src/four_error_using/`. Everything under `src/archive/` is old/unused.

## 1. Run an experiment (entry points — you call these)
| File | Use it to |
|---|---|
| `src/four_error_using/detection_caller/detection_caller.py` | **The main launcher.** Runs the scalogram image model + weighted vote. Flags: `--region {event,leftover,all}`, `--no-artifact-reject`, `--vote-weights`, `--fuse-kinematic`, `--fuse-alpha`, `--entropy`, `--entropy-signal`, `--n-splits`. |
| `src/four_error_using/detection_caller/kinematic_caller.py` | Run the kinematics-only experiment (saccade indicators, no images). |

## 2. Turn raw CSV → model inputs (data prep)
| File | Use it to |
|---|---|
| `src/four_error_using/data_processor/data_engineer.py` | Load VOG CSVs, cut event/leftover/all windows, make the 4-channel CWT scalograms, add the optional entropy channel. Excludes `New_data`. |
| `src/four_error_using/data_processor/kinematic_features.py` | From the same trials, compute the 10 saccade numbers (latency, peak velocity, amplitude, gain, …). |

## 3. The model (what learns)
| File | Use it to |
|---|---|
| `src/four_error_using/models/task_conditioned_classifier.py` | The whole model: frozen MobileViT + small trainable adapter + task head. |
| `src/four_error_using/models/layers/frozen_mobilevit_backbone.py` | Frozen MobileViT feature extractor. |
| `src/four_error_using/models/layers/conv_adapter.py` | Small trainable adapter (the part that actually learns). |
| `src/four_error_using/models/layers/task_embedding.py` | Tells the model which VOG task a window came from. |
| `src/four_error_using/models/layers/cosine_linear.py` | The final HC-vs-MCI decision layer. |

## 4. Train + evaluate (the loop)
| File | Use it to |
|---|---|
| `src/four_error_using/model_trainers/model_trainer.py` | Train one fold (early stop, best-AUROC checkpoint). |
| `src/four_error_using/evaluators/repetitive_validator.py` | The 30-repetition stratified cross-validation; applies weighted vote + kinematic fusion; dumps `window_probs.csv`. |
| `src/four_error_using/evaluators/kinematic_validator.py` | Same idea but for the kinematics-only experiment. |

## 5. Pictures & reports (look at results)
| File | Use it to |
|---|---|
| `src/graph_generators/generator.py` | Draw the annotated eye-trajectory plots. |
| `src/scalograms_generators/*` | Draw the scalogram images (per-trial, mean, variance, squared-difference maps). |
| `src/spectrograms_generators/short_term_fourier_transform_generators/*` | Same but STFT spectrograms instead of wavelets. |
| `src/probe_generators/probe_generator.py` | Report which VOG task contributed most. |

## 6. Shared
| File | Use it to |
|---|---|
| `src/paths.py` | All the folder locations (data, outputs, checkpoints) in one place. |

## 7. Archive (ignore — kept for reference)
- `src/archive/candidates/f_swin_sift_dbt/` — the discarded Swin + SIFT-DBT model.
- `src/archive/detection_caller/` — the old caller before the rename.

---
**Typical flow:** `detection_caller.py` → `data_engineer.py` (make scalograms) →
`task_conditioned_classifier.py` (model) → `repetitive_validator.py` +
`model_trainer.py` (30× train/test) → results in `outputs/` and `key_results/`.
