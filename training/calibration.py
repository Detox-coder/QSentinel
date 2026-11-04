"""
Confidence Calibration — Temperature scaling and calibration analysis.

Neural networks (especially deep ones) are systematically overconfident.
A predicted 95% confidence may only be correct 70% of the time on OOD data.

This module implements:
  1. Temperature scaling (post-hoc calibration)
  2. Calibration curve computation (reliability diagrams)
  3. Expected Calibration Error (ECE) metric

Usage (training):
    calibrator = TemperatureCalibrator()
    temperature = calibrator.fit(model, val_loader, device)
    # Save temperature with model checkpoint

Usage (inference):
    logits = model(x)
    calibrated_logits = logits / temperature
    probs = softmax(calibrated_logits)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class TemperatureCalibrator:
    """
    Learn optimal temperature T for post-hoc logit calibration.

    After training, divide logits by T before softmax to obtain
    calibrated probability estimates.
    """

    def __init__(self):
        self.temperature: float = 1.0
        self.ece_before: float = 0.0
        self.ece_after: float = 0.0

    def fit(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 200,
    ) -> float:
        """
        Find optimal temperature using NLL on validation set.

        Returns the learned temperature value.
        """
        model.eval()

        # Collect all logits and labels from validation set
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                all_logits.append(logits.cpu())
                all_labels.append(y_batch)

        logits_tensor = torch.cat(all_logits)
        labels_tensor = torch.cat(all_labels)

        # Measure ECE before calibration
        probs_before = F.softmax(logits_tensor, dim=1).numpy()
        labels_np = labels_tensor.numpy()
        self.ece_before = self._compute_ece(probs_before, labels_np)

        # Optimize temperature
        temperature = nn.Parameter(torch.ones(1) * 1.5)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits_tensor / temperature.clamp(min=0.01)
            loss = criterion(scaled_logits, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.temperature = float(temperature.item())
        # Clamp to reasonable range
        self.temperature = max(0.1, min(self.temperature, 10.0))

        # Measure ECE after calibration
        probs_after = F.softmax(logits_tensor / self.temperature, dim=1).numpy()
        self.ece_after = self._compute_ece(probs_after, labels_np)

        print(f"  [CALIBRATION] Temperature: {self.temperature:.4f}")
        print(f"  [CALIBRATION] ECE before: {self.ece_before:.4f}")
        print(f"  [CALIBRATION] ECE after:  {self.ece_after:.4f}")

        return self.temperature

    @staticmethod
    def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
        """
        Compute Expected Calibration Error.

        ECE = Σ (|bin_size|/N) * |accuracy(bin) - confidence(bin)|
        """
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences >= lo) & (confidences < hi)
            if mask.sum() == 0:
                continue
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_size = mask.sum() / len(labels)
            ece += bin_size * abs(bin_acc - bin_conf)

        return float(ece)

    def calibration_curve(
        self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
    ) -> dict:
        """
        Compute calibration curve data for reliability diagram.

        Returns dict with bin_centers, bin_accuracies, bin_confidences, bin_counts.
        """
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences >= lo) & (confidences < hi)
            count = int(mask.sum())
            if count == 0:
                continue
            bin_centers.append((lo + hi) / 2)
            bin_accuracies.append(float(accuracies[mask].mean()))
            bin_confidences.append(float(confidences[mask].mean()))
            bin_counts.append(count)

        return {
            "bin_centers": bin_centers,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
            "ece": self._compute_ece(probs, labels, n_bins),
        }

    def summary(self) -> str:
        return (
            f"TemperatureCalibrator:\n"
            f"  Temperature: {self.temperature:.4f}\n"
            f"  ECE before:  {self.ece_before:.4f}\n"
            f"  ECE after:   {self.ece_after:.4f}"
        )
