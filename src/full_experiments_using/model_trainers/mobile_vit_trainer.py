"""8-experiment trainer.

Preserves the original 8-task training recipe (linear warm-up, gradient
clipping, ReduceLROnPlateau on val AUROC, best-AUROC checkpointing) and
plugs into the project's output/logging conventions (per-fold checkpoint
file under outputs/checkpoints/run_<ts>_8exp/, logger instead of print).
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _auroc(true_labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    y_sorted = true_labels[order]
    n_pos = np.sum(true_labels == 1)
    n_neg = np.sum(true_labels == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    tprs, fprs = [0.0], [0.0]
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)
    return float(np.trapezoid(tprs, fprs))


def _get_labels(dataset) -> torch.Tensor:
    """Extract label tensor from Dataset, Subset, or AugmentedSubset."""
    if hasattr(dataset, "y"):
        return dataset.y
    return dataset.dataset.y[dataset.indices]


class ModelTrainer:
    WARMUP_EPOCHS = 5
    GRAD_CLIP = 1.0

    def __init__(
        self,
        model,
        device="auto",
        checkpoint_dir: Optional[Path] = None,
        fold_idx: Optional[int] = None,
    ):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.fold_idx = fold_idx

    def _checkpoint_path(self) -> Optional[Path]:
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tag = f"fold_{self.fold_idx:02d}" if self.fold_idx is not None else "single"
        return self.checkpoint_dir / f"{tag}_best.pth"

    def train_model(
        self,
        train_dataset,
        val_dataset,
        max_epochs: int = 500,
        batch_size: int = 32,
        early_stop_patience: int = 40,
    ):
        logger.info(
            "Device: %s | max_epochs=%d | batch=%d | es_patience=%d",
            self.device, max_epochs, batch_size, early_stop_patience,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        label_counts = Counter(_get_labels(train_dataset).tolist())
        total = sum(label_counts.values())
        alpha = torch.tensor(
            [total / (2 * label_counts[i]) for i in range(2)], dtype=torch.float32
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=alpha)
        logger.info("CE weights — HC: %.3f  MCI: %.3f", alpha[0], alpha[1])

        base_lr = self.optimizer.param_groups[0]['lr']

        # ReduceLROnPlateau monitors val AUROC (higher = better → mode='max')
        plateau = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.2, patience=10, min_lr=1e-6
        )

        best_auroc = -1.0
        best_epoch = 0
        no_improve = 0
        ckpt_path = self._checkpoint_path()

        for epoch in range(max_epochs):
            # Linear warm-up (epochs 0 … WARMUP_EPOCHS-1)
            if epoch < self.WARMUP_EPOCHS:
                lr = base_lr * (epoch + 1) / self.WARMUP_EPOCHS
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr

            # Train
            self.model.train()
            t_loss = t_correct = t_total = 0
            for inputs, tasks, labels in train_loader:
                inputs = inputs.to(self.device)
                tasks = tasks.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    out = self.model(inputs, tasks)
                    loss = criterion(out, labels)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=self.GRAD_CLIP,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=self.GRAD_CLIP,
                    )
                    self.optimizer.step()

                t_loss += loss.item() * inputs.size(0)
                _, pred = out.max(1)
                t_total += labels.size(0)
                t_correct += pred.eq(labels).sum().item()

            # Validate
            self.model.eval()
            v_loss = v_correct = v_total = 0
            all_probs: list = []
            all_lbls: list = []

            with torch.no_grad():
                for inputs, tasks, labels in val_loader:
                    inputs_d = inputs.to(self.device)
                    tasks_d = tasks.to(self.device)
                    labels_d = labels.to(self.device)

                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        out = self.model(inputs_d, tasks_d)
                        loss = criterion(out, labels_d)

                    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    all_probs.extend(probs.tolist())
                    all_lbls.extend(labels.numpy().tolist())

                    v_loss += loss.item() * inputs.size(0)
                    _, pred = out.max(1)
                    v_total += labels_d.size(0)
                    v_correct += pred.eq(labels_d).sum().item()

            v_auroc = _auroc(np.array(all_lbls), np.array(all_probs))
            v_acc = 100. * v_correct / max(v_total, 1)
            t_acc = 100. * t_correct / max(t_total, 1)
            cur_lr = self.optimizer.param_groups[0]['lr']

            logger.info(
                "Ep [%03d/%d] T-Acc: %5.1f%%  V-Acc: %5.1f%%  V-AUROC: %.4f  LR: %.2e",
                epoch + 1, max_epochs, t_acc, v_acc, v_auroc, cur_lr,
            )

            # Plateau scheduler kicks in after warm-up
            if epoch >= self.WARMUP_EPOCHS:
                plateau.step(v_auroc)

            # Early stopping on val AUROC
            if v_auroc > best_auroc:
                best_auroc = v_auroc
                best_epoch = epoch + 1
                no_improve = 0
                if ckpt_path is not None:
                    torch.save(self.model.state_dict(), ckpt_path)
                    logger.info("  ↳ new best AUROC; saved checkpoint: %s", ckpt_path)
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    logger.info(
                        "Early stop — best AUROC %.4f at epoch %d (no improvement for %d epochs)",
                        best_auroc, best_epoch, early_stop_patience,
                    )
                    break

        logger.info(
            "Done. Best AUROC: %.4f (epoch %d) → %s",
            best_auroc, best_epoch, ckpt_path if ckpt_path else "in-memory only",
        )
        return self.model
