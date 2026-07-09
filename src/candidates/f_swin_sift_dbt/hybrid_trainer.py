"""Hybrid orchestrator — 30-fold Monte-Carlo Group CV × (Stage 1 → Stage 2).

Per fold:
  1. `GroupShuffleSplit(random_state=42)` chooses train_subjs / test_subjs.
  2. Inner 90/10 split of train_subjs → stage1_train_subjs / stage1_val_subjs
     (Stage 1 validation-monitor set only; no test-subject leakage).
  3. Stage 1: train Swin-Tiny backbone-tuner on flat windows restricted to
     stage1_train_subjs; save best-val-loss (adapter + backbone) state.
  4. Stage 2: instantiate SIFT-DBT classifier with the Stage 1 extractor
     frozen; train gate + shared classifier + attention_prior on the ragged
     SubjectBundleDataset restricted to train_subjs (full inner 100 %).
  5. Evaluate on test_subjs; append fold_metrics.
  6. Persist orchestrator state (completed_folds, fold_metrics) so the run
     can resume from the last completed fold on relaunch.
"""
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from candidates.f_swin_sift_dbt.data_processor.datasets import (
    FlatWindowDataset,
    SubjectBundleDataset,
    flat_indices_for_subjects,
    ragged_collate,
)
from candidates.f_swin_sift_dbt.models.stage1_backbone_tuner import Stage1SwinBackboneTuner
from candidates.f_swin_sift_dbt.models.sift_dbt_classifier import SIFT_DBTClassifier
from candidates.f_swin_sift_dbt.model_trainers.stage1_trainer import Stage1Trainer
from candidates.f_swin_sift_dbt.model_trainers.stage2_trainer import Stage2Trainer, _auroc

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint-and-resume state
# ────────────────────────────────────────────────────────────────────────────

def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    tmp.replace(state_path)


def _load_state(state_path: Path) -> Optional[dict]:
    if not state_path.exists():
        return None
    try:
        with open(state_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("Failed to load resume state (%s); starting fresh", e)
        return None


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────

class HybridTrainer:
    def __init__(
        self,
        flat_dataset: FlatWindowDataset,
        bundle_dataset: SubjectBundleDataset,
        checkpoint_dir: Path,
        num_tasks: int = 8,
        n_splits: int = 30,
        stage1_epochs: int = 50,
        stage1_patience: int = 10,
        stage1_lr: float = 5e-5,
        stage1_batch_size: int = 32,
        stage2_epochs: int = 500,
        stage2_patience: int = 30,
        stage2_lr: float = 1e-4,
        stage2_batch_size: int = 8,
        stage2_dropout: float = 0.5,
        attention_prior: Optional[torch.Tensor] = None,
        inner_val_frac: float = 0.10,
        inner_val_seed: int = 42,
        resume_from: Optional[Path] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flat_ds = flat_dataset
        self.bundle_ds = bundle_dataset
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.checkpoint_dir / "state.pkl"

        self.num_tasks = num_tasks
        self.n_splits = n_splits
        self.stage1_epochs = stage1_epochs
        self.stage1_patience = stage1_patience
        self.stage1_lr = stage1_lr
        self.stage1_batch_size = stage1_batch_size
        self.stage2_epochs = stage2_epochs
        self.stage2_patience = stage2_patience
        self.stage2_lr = stage2_lr
        self.stage2_batch_size = stage2_batch_size
        self.stage2_dropout = stage2_dropout
        self.attention_prior = attention_prior
        self.inner_val_frac = inner_val_frac
        self.inner_val_seed = inner_val_seed
        self.resume_from = resume_from

    # ── Per-fold Stage 1 → Stage 2 ────────────────────────────────────────
    def _run_stage1(self, train_subj_ids, val_subj_ids, fold_idx):
        # Flat windows from THIS fold's inner train/val split
        train_flat_idx = flat_indices_for_subjects(self.flat_ds, train_subj_ids)
        val_flat_idx = flat_indices_for_subjects(self.flat_ds, val_subj_ids)
        train_ds = Subset(self.flat_ds, train_flat_idx)
        val_ds = Subset(self.flat_ds, val_flat_idx)
        logger.info(
            "Stage 1 fold %d — flat windows: train %d (subj=%d)  val %d (subj=%d)",
            fold_idx + 1, len(train_ds), len(train_subj_ids),
            len(val_ds), len(val_subj_ids),
        )
        ckpt_s1 = self.checkpoint_dir / f"fold_{fold_idx:02d}_stage1.pth"
        model = Stage1SwinBackboneTuner(in_channels=4, num_classes=2)
        trainer = Stage1Trainer(
            model, self.device, checkpoint_path=ckpt_s1,
            lr=self.stage1_lr, weight_decay=1e-4,
        )
        best_state, best_val_loss = trainer.train(
            train_ds, val_ds,
            epochs=self.stage1_epochs,
            batch_size=self.stage1_batch_size,
            patience=self.stage1_patience,
        )
        del model, trainer
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return best_state, best_val_loss

    def _run_stage2(self, extractor_state, train_subj_ids, test_subj_ids, fold_idx):
        # Subset of SubjectBundleDataset by subject_id membership
        train_b_idx = [i for i, s in enumerate(self.bundle_ds.subject_ids) if s in train_subj_ids]
        test_b_idx  = [i for i, s in enumerate(self.bundle_ds.subject_ids) if s in test_subj_ids]
        train_subset = Subset(self.bundle_ds, train_b_idx)
        test_subset  = Subset(self.bundle_ds, test_b_idx)
        logger.info(
            "Stage 2 fold %d — bundles: train %d (subj=%d)  test %d (subj=%d)",
            fold_idx + 1, len(train_subset), len(train_subj_ids),
            len(test_subset), len(test_subj_ids),
        )
        ckpt_s2 = self.checkpoint_dir / f"fold_{fold_idx:02d}_stage2.pth"

        model = SIFT_DBTClassifier(
            num_tasks=self.num_tasks,
            num_classes=2,
            in_channels=4,
            dropout=self.stage2_dropout,
            attention_prior=self.attention_prior,
            pretrained_extractor=extractor_state,
        )
        trainer = Stage2Trainer(
            model, self.device, checkpoint_path=ckpt_s2,
            lr=self.stage2_lr, weight_decay=1e-2,
        )
        trained_model = trainer.train(
            train_subset, test_subset,
            epochs=self.stage2_epochs,
            batch_size=self.stage2_batch_size,
            patience=self.stage2_patience,
        )

        # Final evaluation with the (best-checkpoint-restored) model
        fold_metrics = self._evaluate(trained_model, test_subset)
        del model, trainer, trained_model
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return fold_metrics

    def _evaluate(self, model, test_subset):
        model.eval()
        loader = DataLoader(test_subset, batch_size=self.stage2_batch_size,
                             shuffle=False, collate_fn=ragged_collate)
        all_probs, all_lbls, all_sids = [], [], []
        all_logits_task, all_W_task = [], []
        with torch.no_grad():
            for ragged_bundle, ratios, y, sids in loader:
                final_logits, logits_task, W_task = model(ragged_bundle, ratios)
                probs = torch.softmax(final_logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs.tolist())
                all_lbls.extend(y.numpy().tolist())
                all_sids.extend(sids)
                all_logits_task.append(logits_task.cpu().numpy())
                all_W_task.append(W_task.cpu().numpy())
        true_arr = np.array(all_lbls)
        prob_arr = np.array(all_probs)
        pred_arr = (prob_arr > 0.5).astype(int)
        tp = int(np.sum((true_arr == 1) & (pred_arr == 1)))
        tn = int(np.sum((true_arr == 0) & (pred_arr == 0)))
        fp = int(np.sum((true_arr == 0) & (pred_arr == 1)))
        fn = int(np.sum((true_arr == 1) & (pred_arr == 0)))
        acc = (tp + tn) / max(len(true_arr), 1)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        auroc = _auroc(true_arr, prob_arr)
        return {
            "acc": acc, "sens": sens, "spec": spec, "auroc": auroc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "true": true_arr.tolist(), "prob": prob_arr.tolist(),
            "sids": all_sids,
            "logits_task": np.concatenate(all_logits_task, axis=0).tolist(),
            "W_task":      np.concatenate(all_W_task,      axis=0).tolist(),
        }

    # ── Public entry ──────────────────────────────────────────────────────
    def run(self):
        subj_ids = np.array(self.bundle_ds.subject_ids)
        y_arr = self.bundle_ds.y.numpy()
        indices = np.arange(len(self.bundle_ds))

        # Resume state, if any
        state = _load_state(self.resume_from or self.state_path)
        if state is None:
            state = {"completed_folds": [], "fold_metrics": [None] * self.n_splits}
        else:
            logger.info(
                "Resumed from %s: %d/%d folds already complete",
                self.state_path, len(state["completed_folds"]), self.n_splits,
            )

        logger.info(
            "Hybrid CV — %d folds | device=%s | tasks=%d | S1(epochs=%d, patience=%d, lr=%.0e)"
            " | S2(epochs=%d, patience=%d, lr=%.0e) | attention_prior=%s",
            self.n_splits, self.device, self.num_tasks,
            self.stage1_epochs, self.stage1_patience, self.stage1_lr,
            self.stage2_epochs, self.stage2_patience, self.stage2_lr,
            "on" if self.attention_prior is not None else "off",
        )

        gss = GroupShuffleSplit(n_splits=self.n_splits, test_size=0.3, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(gss.split(indices, y_arr, groups=subj_ids)):
            if fold in state["completed_folds"]:
                logger.info("Fold %02d/%d — already complete, skipping (resume)",
                             fold + 1, self.n_splits)
                continue
            train_subjs = set(subj_ids[train_idx])
            test_subjs = set(subj_ids[test_idx])
            assert train_subjs.isdisjoint(test_subjs), "Subject leakage!"

            logger.info("=" * 60)
            logger.info(
                "=== Fold %02d/%d | train: %d subj | test: %d subj ===",
                fold + 1, self.n_splits, len(train_subjs), len(test_subjs),
            )

            # Inner 90/10 subject split for Stage 1 val
            inner_gss = GroupShuffleSplit(
                n_splits=1, test_size=self.inner_val_frac,
                random_state=self.inner_val_seed,
            )
            inner_train_idx, inner_val_idx = next(
                inner_gss.split(np.arange(len(train_subjs)),
                                groups=np.array(sorted(train_subjs)))
            )
            train_subjs_sorted = sorted(train_subjs)
            stage1_train_subjs = {train_subjs_sorted[i] for i in inner_train_idx}
            stage1_val_subjs   = {train_subjs_sorted[i] for i in inner_val_idx}

            # Stage 1
            extractor_state, s1_val_loss = self._run_stage1(
                stage1_train_subjs, stage1_val_subjs, fold,
            )

            # Stage 2 (uses FULL train_subjs, including the inner-val ones)
            fold_metrics = self._run_stage2(extractor_state, train_subjs, test_subjs, fold)
            fold_metrics["stage1_best_val_loss"] = s1_val_loss

            state["fold_metrics"][fold] = fold_metrics
            state["completed_folds"].append(fold)
            _save_state(self.state_path, state)

            logger.info(
                "-> Fold %02d Result | Acc=%.3f  Sens=%.3f  Spec=%.3f  AUROC=%.3f  (S1 val-loss=%.4f)",
                fold + 1, fold_metrics["acc"], fold_metrics["sens"],
                fold_metrics["spec"], fold_metrics["auroc"], s1_val_loss,
            )

        # Aggregate
        completed = [m for m in state["fold_metrics"] if m is not None]
        accs = [m["acc"] for m in completed]
        senss = [m["sens"] for m in completed]
        specs = [m["spec"] for m in completed]
        aurocs = [m["auroc"] for m in completed]
        logger.info("=" * 60)
        logger.info("  Hybrid SIFT-DBT MC-CV Results (%d folds)", len(completed))
        logger.info("  Accuracy    : %.3f ± %.3f", float(np.mean(accs)), float(np.std(accs)))
        logger.info("  Sensitivity : %.3f ± %.3f", float(np.mean(senss)), float(np.std(senss)))
        logger.info("  Specificity : %.3f ± %.3f", float(np.mean(specs)), float(np.std(specs)))
        logger.info("  AUROC       : %.3f ± %.3f", float(np.mean(aurocs)), float(np.std(aurocs)))
        logger.info("=" * 60)
        return state
