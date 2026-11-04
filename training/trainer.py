"""
Training loop for the FFNN classifier.
Includes metric tracking, early stopping, and checkpointing.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class ClassifierTrainer:
    """Trains the supervised QUICClassifier."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5,
        )
        self.patience = patience
        self.history: dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "val_f1": [], "lr": [],
        }

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
    ) -> dict:
        """Full training loop with early stopping."""
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        print(f"\n{'='*60}")
        print(f"  CLASSIFIER TRAINING  ({epochs} epochs max)")
        print(f"{'='*60}")
        print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}"
              f"  {'Train Acc':>9}  {'Val Acc':>7}  {'Val F1':>6}  {'LR':>8}")
        print(f"  {'-'*54}")

        for epoch in range(1, epochs + 1):
            # ── Train ────────────────────────────────────
            self.model.train()
            train_loss = 0.0
            all_preds, all_labels = [], []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * X_batch.size(0)
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

            train_loss /= len(train_loader.dataset)
            train_acc = accuracy_score(all_labels, all_preds)

            # ── Validate ─────────────────────────────────
            val_loss, val_acc, val_f1 = self._validate(val_loader)

            # ── Scheduler ────────────────────────────────
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── History ──────────────────────────────────
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(current_lr)

            # ── Print ────────────────────────────────────
            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  {epoch:5d}  {train_loss:10.4f}  {val_loss:10.4f}"
                    f"  {train_acc:9.4f}  {val_acc:7.4f}  {val_f1:6.4f}"
                    f"  {current_lr:8.1e}"
                )

            # ── Early stopping ───────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"  ↳ Early stopping at epoch {epoch} (patience={self.patience})")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        print(f"  Best val loss: {best_val_loss:.4f}")
        return self.history

    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)

                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        return val_loss, val_acc, f1




