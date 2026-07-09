"""Stage 1 trainer — window-level supervised training of the Swin-Tiny extractor.

Task-agnostic — the Stage 1 model does not use task_id as an input. The trainer
just consumes (tensor, task_id, label, sid) tuples from FlatWindowDataset and
ignores task_id (task_id lives in the ragged bundle at Stage 2).

Monitoring:
  · Class-weighted cross-entropy (inverse-frequency alphas from train labels)
  · 5-epoch linear warm-up
  · ReduceLROnPlateau on val loss (mode='min')
  · Grad-clip 1.0
  · AMP autocast + GradScaler on CUDA
  · Early stop on val loss (patience default 10)
  · Best-val-loss extractor state persisted to disk each time val improves
"""
import logging
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


def _labels_of(dataset) -> list:
    """Extract labels from Dataset or Subset — for class-weight computation."""
    if hasattr(dataset, "y"):
        return dataset.y.tolist()
    return dataset.dataset.y[torch.as_tensor(dataset.indices, dtype=torch.long)].tolist()


class Stage1Trainer:
    WARMUP_EPOCHS = 5
    GRAD_CLIP = 1.0

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        checkpoint_path: Optional[Path] = None,
        lr: float = 5e-5,
        weight_decay: float = 1e-4,
    ):
        self.device = device
        self.model = model.to(device)
        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.lr = lr
        self.wd = weight_decay
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    def _class_weights(self, train_ds) -> torch.Tensor:
        labels = _labels_of(train_ds)
        counts = Counter(labels)
        total = sum(counts.values())
        return torch.tensor(
            [total / (2 * counts.get(i, 1)) for i in range(2)],
            dtype=torch.float32,
        )

    def train(
        self,
        train_ds,
        val_ds,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
    ) -> Tuple[Any, float]:
        """Fit and return (best_extractor_state, best_val_loss).

        `best_extractor_state` is a dict `{"adapter": sd, "backbone": sd}` from
        the model's `.export_extractor_state()` method, taken at the epoch of
        lowest val loss.
        """
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        alpha = self._class_weights(train_ds).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=alpha)
        logger.info("Stage 1 CE weights — HC: %.3f  MCI: %.3f", alpha[0], alpha[1])

        base_lr = self.optimizer.param_groups[0]["lr"]
        plateau = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.2, patience=5, min_lr=1e-7
        )

        best_val_loss = float("inf")
        best_epoch = 0
        no_improve = 0
        best_state: Optional[dict] = None

        for epoch in range(epochs):
            if epoch < self.WARMUP_EPOCHS:
                lr = base_lr * (epoch + 1) / self.WARMUP_EPOCHS
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

            # ── Train ────────────────────────────────────────────────
            self.model.train()
            t_loss = t_correct = t_total = 0
            for x, _task_id, y, _sid in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(x)
                    loss = criterion(logits, y)
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
                t_loss += loss.item() * y.size(0)
                _, pred = logits.max(1)
                t_correct += pred.eq(y).sum().item()
                t_total += y.size(0)

            # ── Validate ─────────────────────────────────────────────
            self.model.eval()
            v_loss = v_correct = v_total = 0
            with torch.no_grad():
                for x, _task_id, y, _sid in val_loader:
                    x_d = x.to(self.device)
                    y_d = y.to(self.device)
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        logits = self.model(x_d)
                        loss = criterion(logits, y_d)
                    v_loss += loss.item() * y_d.size(0)
                    _, pred = logits.max(1)
                    v_correct += pred.eq(y_d).sum().item()
                    v_total += y_d.size(0)

            train_loss_avg = t_loss / max(t_total, 1)
            train_acc = 100.0 * t_correct / max(t_total, 1)
            val_loss_avg = v_loss / max(v_total, 1)
            val_acc = 100.0 * v_correct / max(v_total, 1)
            cur_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                "S1-Ep [%03d/%d] T-Loss: %.4f  T-Acc: %5.1f%%  V-Loss: %.4f  V-Acc: %5.1f%%  LR: %.2e",
                epoch + 1, epochs, train_loss_avg, train_acc,
                val_loss_avg, val_acc, cur_lr,
            )

            if epoch >= self.WARMUP_EPOCHS:
                plateau.step(val_loss_avg)

            # Early stop on val loss
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_epoch = epoch + 1
                no_improve = 0
                best_state = self.model.export_extractor_state()
                if self.checkpoint_path is not None:
                    torch.save(best_state, self.checkpoint_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(
                        "Stage 1 early stop — best val loss %.4f at epoch %d",
                        best_val_loss, best_epoch,
                    )
                    break

        logger.info("Stage 1 done. Best val loss: %.4f (epoch %d)", best_val_loss, best_epoch)
        return best_state, best_val_loss
