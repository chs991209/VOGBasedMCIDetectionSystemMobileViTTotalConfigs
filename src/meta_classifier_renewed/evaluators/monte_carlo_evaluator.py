"""Renewed meta-classifier MC Group CV evaluator for Solution D."""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from transformers import MobileViTModel

from meta_classifier_renewed.models.dynamic_latent_classifier import DynamicLatentClassifier
from meta_classifier_renewed.model_trainers.meta_trainer import MetaTrainer
from meta_classifier_renewed.data_processor.data_preprocessing import ragged_collate

logger = logging.getLogger(__name__)

class MetaMonteCarloGroupEvaluator:
    def __init__(
        self, dataset, max_epochs: int = 500, batch_size: int = 8, n_splits: int = 30,
        checkpoint_dir: Optional[Path] = None, early_stop_patience: int = 40,
        dropout: float = 0.5, lr: float = 1e-4, weight_decay: float = 1e-2,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.early_stop_patience = early_stop_patience
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

    def _infer_subjects(self, model, val_subset: Subset):
        model.eval()
        # [Solution D] ragged_collate 사용 — 4-tuple: bundle, ratios, label, sid
        loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, collate_fn=ragged_collate)
        out = {}
        with torch.no_grad():
            for ragged_bundle, ratios, label, sid in loader:
                logits, _gates = model(ragged_bundle, ratios)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                for s, p, l in zip(sid, probs, label.numpy()):
                    out[str(s)] = (int(l), float(p))
        return out

    def run(self):
        y_arr = self.dataset.y.numpy()
        subj_ids = np.array(self.dataset.subject_ids)
        indices = np.arange(len(self.dataset))

        fold_metrics = []
        logger.info(
            "Renewed MC Group CV (Dynamic Latent Aggregation) — %d folds | device=%s | dropout=%.2f | lr=%.0e | wd=%.0e",
            self.n_splits, self.device, self.dropout, self.lr, self.weight_decay
        )

        shared_backbone = MobileViTModel.from_pretrained("apple/mobilevit-small")
        for p in shared_backbone.parameters():
            p.requires_grad = False
        shared_backbone.eval()
        shared_backbone = shared_backbone.to(self.device)

        gss = GroupShuffleSplit(n_splits=self.n_splits, test_size=0.3, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(gss.split(indices, y_arr, groups=subj_ids)):
            train_subjs = set(subj_ids[train_idx])
            test_subjs = set(subj_ids[test_idx])
            assert train_subjs.isdisjoint(test_subjs), "Subject leakage detected!"

            logger.info("=== Fold %02d/%d | train: %d subj | test: %d subj ===", fold + 1, self.n_splits, len(train_subjs), len(test_subjs))

            model = DynamicLatentClassifier(
                num_tasks=self.dataset.NUM_TASKS,
                dropout_rate=self.dropout,
                shared_backbone=shared_backbone,
            )
            trainer = MetaTrainer(
                model, device=self.device, checkpoint_dir=self.checkpoint_dir, fold_idx=fold,
                lr=self.lr, weight_decay=self.weight_decay,
            )
            trained = trainer.train_model(
                Subset(self.dataset, train_idx), Subset(self.dataset, test_idx),
                max_epochs=self.max_epochs, batch_size=self.batch_size, early_stop_patience=self.early_stop_patience,
            )

            preds = self._infer_subjects(trained, Subset(self.dataset, test_idx))
            true_arr = np.array([v[0] for v in preds.values()])
            prob_arr = np.array([v[1] for v in preds.values()])

            pred_arr = (prob_arr > 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(true_arr, pred_arr, labels=[0, 1]).ravel()
            acc = accuracy_score(true_arr, pred_arr)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            try: auroc = float(roc_auc_score(true_arr, prob_arr))
            except Exception: auroc = 0.5

            fold_metrics.append({"acc": acc, "sens": sens, "spec": spec, "auroc": auroc})
            logger.info("-> Fold %02d Result | Acc=%.3f Sens=%.3f Spec=%.3f AUROC=%.3f", fold + 1, acc, sens, spec, auroc)

            del model, trainer, trained
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        accs = [m["acc"] for m in fold_metrics]
        aurocs = [m["auroc"] for m in fold_metrics]
        logger.info("=" * 60)
        logger.info("  Dynamic MC-CV Results (%d folds) - STRICT 0.5 THRESHOLD", self.n_splits)
        logger.info("  Accuracy    : %.3f ± %.3f", float(np.mean(accs)), float(np.std(accs)))
        logger.info("  AUROC       : %.3f ± %.3f", float(np.mean(aurocs)), float(np.std(aurocs)))
        logger.info("=" * 60)

        return fold_metrics