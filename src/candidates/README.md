# Pipeline Candidates — Catalog

Every architectural approach explored during this project, catalogued as a self-contained sub-package. Historical candidates (A–E) hold documentation + pointers to their original source locations (kept for reproducibility of past runs). The current implementation target (F) contains full source code inside this package.

Code redundancy across candidates is intentional. Each sub-package should be readable in isolation.

---

## Status table

| ID | Name | Status | Best AUROC | Source of truth |
|:--:|---|:--:|:--:|---|
| **A** | wvote soft-vote (`full_experiments_using` + wvote scheme) | ✅ complete (headline baseline) | **0.791 ± 0.142** | `src/full_experiments_using/` |
| **B** | Option C — sequence-level trial fusion in MobileViT | ⚰️ deprecated (padding-mask leak) | 0.978 (contaminated) | `src/meta_classifier_using/` |
| **C** | Solution C — LeakFreeMetaClassifier, 4-task strict parity | ✅ complete | 0.711 ± 0.164 | `src/meta_classifier_renewed/` @ commit before Solution D refactor |
| **D** | Solution D — DynamicLatentClassifier, ragged tensors + cross-axis ratio | ✅ complete (multiple runs) | 0.710 (2 tasks) / **0.737** (8 tasks) | `src/meta_classifier_renewed/` current |
| **E** | MobileViT + Hybrid two-stage (density transfer, anti-saccade only Stage 2) | ⏸️ approved, superseded by F before launch | — | design docs only |
| **F** | **Swin-Tiny + SIFT-DBT Attention-Weighted Vote** | 🚧 **under implementation** | — | `src/candidates/f_swin_sift_dbt/` |

---

## Comparison at a glance

| Aspect | A wvote | C Solution C | D Solution D | F Swin SIFT-DBT (target) |
|---|---|---|---|---|
| Backbone | MobileViT-small frozen | MobileViT-small frozen | MobileViT-small frozen | Swin-Tiny — S1 unfrozen, S2 frozen |
| Training paradigm | Per-window CE | Per-subject bundle | Per-subject ragged bundle | Hybrid: S1 per-window + S2 per-bundle |
| Task surface | 8 | 4 (plenty) | 2 (anti-only) or 8 (ablation) | 8 |
| Task aggregation | Weighted soft-vote at inference | Softmax gate → weighted-sum of features | Same as C, ragged trials, μ‖σ² gate | **Attention-weighted vote of per-task logits** (late fusion) |
| Task weighting | Hard-coded scheme (0,0.5,0.5,0.5,0,1.5,1.5,1.5) | Learned scalar gate | Learned scalar gate | Learned attention + optional probe-derived prior |
| Cross-axis ratio | ❌ | ❌ | ✅ (in 4-error cache) | ✅ (inherited from D's cache) |
| Fused dim | (from 4-error) 5120 concat | 4 · D = 2560 concat | 2·D + 1 = 1281 gate input | 2·D + 1 = 1537 per-task; 2 final |
| XAI signal | Probe report post-hoc | none | none | **native — `logits_task` + `W_task` per subject** |
| Best-checkpoint restore | not applicable | ✅ (post-drift-bug fix) | ✅ | ✅ |
| Compute per fold | ~22 min | ~30 s | ~30–60 s | Stage 1: ~60 min; Stage 2: ~90 s |
| Total 30-fold wall time | ~11 h | ~15 min | ~15–30 min | **~30–35 h** (Swin S1 dominates) |
| Checkpoint-and-resume | ❌ not needed | ❌ | ❌ | ✅ **mandatory** for the 30 h continuous run |

---

## Where the numbers came from

- `outputs/reports/full_experiments_using/run_20260527_205622_full_drop050_pat030_wvote/analysis.md`
- `outputs/reports/meta_classifier_using/run_20260531_002841_meta/analysis.md`
- `outputs/reports/meta_classifier_renewed/run_20260629_143830_meta_renewed/analysis.md`
- `outputs/reports/meta_classifier_renewed/run_20260630_010113_dynamic_latent/analysis.md` (Solution D anti-saccade, N=37)
- `outputs/reports/meta_classifier_renewed/run_20260630_022721_dynamic_latent_t01234567/analysis.md` (Solution D all-8 ablation, N=37)

---

## Cross-references

| Concern | Document |
|---|---|
| Task-scope / anti-saccade rationale | `PROBLEM.md`, `PROBLEM.ko.md`, `CODE_AUDIT_VS_MASTER_CONTEXT.md` |
| A → wvote system architecture | `WVOTE_SYSTEM_ARCHITECTURE.md` |
| Per-task LOO ΔAUROC evidence (drives F's attention prior) | `TASK_CONTRIBUTION_RATES.md` |
| D → why it underperforms A | `META_RENEWED_VS_WVOTE.ko.md` |
| E → MobileViT hybrid design docs | `HYBRID_PIPELINE_PRE_IMPLEMENTATION_AUDIT.md`, `HYBRID_PIPELINE_EXECUTION_READY.md` |
| F → Swin-Tiny + SIFT-DBT audit + guide | `SIFT_DBT_PROPOSAL_AUDIT.md`, `SWIN_ADAPTION_GUIDE.md` |
