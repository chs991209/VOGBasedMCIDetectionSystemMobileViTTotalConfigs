import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(
        self,
        model,
        device="auto",
        checkpoint_dir: Optional[Path] = None,
        fold_idx: Optional[int] = None,
    ):
        if device in ("auto", "cuda"):
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

        # Memory Graph Filter
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3, weight_decay=1e-4
        )

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.fold_idx = fold_idx

    def _checkpoint_path(self) -> Optional[Path]:
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tag = f"fold_{self.fold_idx:02d}" if self.fold_idx is not None else "single"
        return self.checkpoint_dir / f"{tag}_best.pth"

    def train_model(self, train_dataset, val_dataset, max_epochs=500, batch_size=32, patience=40):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Inverse frequency weighting
        label_counts = Counter(train_dataset.y.tolist()) if hasattr(train_dataset, "y") else Counter(
            int(y) for _, _, y in train_dataset
        )
        total = sum(label_counts.values())
        alpha = torch.tensor(
            [total / (2 * label_counts[i]) for i in range(2)], dtype=torch.float32
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=alpha)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        ckpt_path = self._checkpoint_path()

        for epoch in range(max_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, tasks, labels in train_loader:
                inputs, tasks, labels = inputs.to(self.device), tasks.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(inputs, tasks)
                    loss = criterion(outputs, labels)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, tasks, labels in val_loader:
                    inputs, tasks, labels = inputs.to(self.device), tasks.to(self.device), labels.to(self.device)
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        outputs = self.model(inputs, tasks)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            scheduler.step()
            val_loss_avg = val_loss / max(val_total, 1)
            logger.info(
                "Epoch [%02d/%d] | Train Acc: %.1f%% | Val Acc: %.1f%% | Val Loss: %.4f",
                epoch + 1, max_epochs,
                100. * train_correct / max(train_total, 1),
                100. * val_correct / max(val_total, 1),
                val_loss_avg,
            )

            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                epochs_without_improvement = 0
                if ckpt_path is not None:
                    torch.save(self.model.state_dict(), ckpt_path)
                    logger.info("  ↳ new best val loss; saved checkpoint: %s", ckpt_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stop at epoch %d (no val-loss improvement for %d epochs; best=%.4f)",
                        epoch + 1, patience, best_val_loss,
                    )
                    break

        return self.model
