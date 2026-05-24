"""Full-experiment (8-task) Monte-Carlo Group CV evaluator.

Mirrors the 2-experiment evaluator's layout/conventions: DI'd probe,
per-fold checkpoint dir, logger, optional AugmentedSubset on the train
split. Uses fully-qualified package imports so this works as long as
`src/` is on sys.path (set up by the entry-point caller).
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from full_experiments_using.data_processor.data_engineering import AugmentedSubset
from full_experiments_using.models.mobile_vit_model import TransferMobileViTClassifier
from full_experiments_using.model_trainers.mobile_vit_trainer import ModelTrainer, _auroc

logger = logging.getLogger(__name__)


class MonteCarloGroupEvaluator:
    def __init__(
        self,
        dataset,
        max_epochs: int = 500,
        batch_size: int = 32,
        n_splits: int = 30,
        probe=None,
        checkpoint_dir: Optional[Path] = None,
        num_tasks: int = 8,
        augment: bool = False,
        early_stop_patience: int = 40,
        dropout: float = 0.3,
    ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.dataset = dataset
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.probe = probe
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.num_tasks = num_tasks
        self.augment = augment
        self.early_stop_patience = early_stop_patience
        self.dropout = float(dropout)

    def _infer_subjects(self, model, test_idx, subj_ids, fold_idx):
        model.eval()
        subj_probs: dict = defaultdict(list)
        subj_labels: dict = {}

        loader = DataLoader(
            Subset(self.dataset, test_idx),
            batch_size=self.batch_size,
            shuffle=False,
        )
        offset = 0
        with torch.no_grad():
            for inputs, tasks, labels in loader:
                inputs = inputs.to(self.device)
                tasks = tasks.to(self.device)
                probs = torch.softmax(model(inputs, tasks), dim=1)[:, 1].cpu().numpy()

                tasks_np = tasks.cpu().numpy()
                labels_np = labels.numpy()
                sid_slice = subj_ids[offset:offset + len(labels)]

                if self.probe is not None:
                    for i in range(len(labels_np)):
                        self.probe.add_window(
                            fold_idx=fold_idx,
                            subject_id=str(sid_slice[i]),
                            task_id=int(tasks_np[i]),
                            true_label=int(labels_np[i]),
                            prob=float(probs[i]),
                        )

                for sid, p, l in zip(sid_slice, probs, labels_np):
                    subj_probs[sid].append(float(p))
                    subj_labels[sid] = int(l)
                offset += len(labels)

        return {
            sid: (subj_labels[sid], float(np.mean(subj_probs[sid])))
            for sid in subj_probs
        }

    def run(self):
        y_arr = self.dataset.y.numpy()
        subj_ids = np.array(self.dataset.subject_ids)
        indices = np.arange(len(self.dataset))

        fold_metrics = []
        logger.info(
            "Stratified MC Group CV — %d folds | device=%s | num_tasks=%d | augment=%s | dropout=%.2f",
            self.n_splits, self.device, self.num_tasks, self.augment, self.dropout,
        )

        gss = GroupShuffleSplit(n_splits=self.n_splits, test_size=0.3, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(gss.split(indices, y_arr, groups=subj_ids)):
            train_subjs = set(subj_ids[train_idx])
            test_subjs = set(subj_ids[test_idx])
            assert train_subjs.isdisjoint(test_subjs), "Subject leakage detected!"

            logger.info(
                "=== Fold %02d/%d | train: %d subj / %d windows | test: %d subj / %d windows ===",
                fold + 1, self.n_splits,
                len(train_subjs), len(train_idx),
                len(test_subjs), len(test_idx),
            )

            model = TransferMobileViTClassifier(
                num_classes=2, in_channels=4, num_tasks=self.num_tasks,
                dropout=self.dropout,
            )
            trainer = ModelTrainer(
                model,
                device=self.device,
                checkpoint_dir=self.checkpoint_dir,
                fold_idx=fold,
            )

            train_subset = (
                AugmentedSubset(self.dataset, train_idx)
                if self.augment
                else Subset(self.dataset, train_idx)
            )
            test_subset = Subset(self.dataset, test_idx)

            trained_model = trainer.train_model(
                train_subset,
                test_subset,
                max_epochs=self.max_epochs,
                batch_size=self.batch_size,
                early_stop_patience=self.early_stop_patience,
            )

            subj_preds = self._infer_subjects(trained_model, test_idx, subj_ids, fold_idx=fold)

            true_arr = np.array([v[0] for v in subj_preds.values()])
            prob_arr = np.array([v[1] for v in subj_preds.values()])
            pred_arr = (prob_arr > 0.5).astype(int)

            tp = int(np.sum((true_arr == 1) & (pred_arr == 1)))
            tn = int(np.sum((true_arr == 0) & (pred_arr == 0)))
            fp = int(np.sum((true_arr == 0) & (pred_arr == 1)))
            fn = int(np.sum((true_arr == 1) & (pred_arr == 0)))

            acc = (tp + tn) / len(true_arr)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            auroc = _auroc(true_arr, prob_arr)

            fold_metrics.append({'acc': acc, 'sens': sens, 'spec': spec, 'auroc': auroc})
            logger.info(
                "-> Fold %02d Result | Acc=%.3f Sens=%.3f Spec=%.3f AUROC=%.3f",
                fold + 1, acc, sens, spec, auroc,
            )

        accs = [m['acc'] for m in fold_metrics]
        senss = [m['sens'] for m in fold_metrics]
        specs = [m['spec'] for m in fold_metrics]
        aurocs = [m['auroc'] for m in fold_metrics]

        logger.info("=" * 60)
        logger.info("  Stratified MC-CV Results (%d folds)", self.n_splits)
        logger.info("  Accuracy    : %.3f ± %.3f", float(np.mean(accs)), float(np.std(accs)))
        logger.info("  Sensitivity : %.3f ± %.3f", float(np.mean(senss)), float(np.std(senss)))
        logger.info("  Specificity : %.3f ± %.3f", float(np.mean(specs)), float(np.std(specs)))
        logger.info("  AUROC       : %.3f ± %.3f", float(np.mean(aurocs)), float(np.std(aurocs)))
        logger.info("=" * 60)

        if self.probe is not None:
            logger.info("Evaluator lifecycle end: writing probe report...")
            self.probe.generate_markdown_report()

        return {
            'accuracy': (float(np.mean(accs)), float(np.std(accs))),
            'sensitivity': (float(np.mean(senss)), float(np.std(senss))),
            'specificity': (float(np.mean(specs)), float(np.std(specs))),
            'auroc': (float(np.mean(aurocs)), float(np.std(aurocs))),
            'fold_metrics': fold_metrics,
        }
