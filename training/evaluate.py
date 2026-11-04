"""
Evaluation module: confusion matrix, per-class metrics, FPR, and anomaly detection analysis.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)


class Evaluator:
    """Comprehensive model evaluation for classifier and Isolation Forest."""

    def __init__(self, class_names: list[str], device: torch.device):
        self.class_names = class_names
        self.device = device

    # ── Classifier Evaluation ────────────────────────────

    def evaluate_classifier(
        self,
        model: nn.Module,
        test_loader,
        title: str = "Test Set",
    ) -> dict:
        """Full evaluation of the supervised classifier."""
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                logits = model(X_batch)
                probs = torch.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Classification report
        print(f"\n{'='*60}")
        print(f"  CLASSIFIER EVALUATION — {title}")
        print(f"{'='*60}")
        print(classification_report(
            all_labels, all_preds,
            target_names=self.class_names,
            digits=4, zero_division=0,
        ))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self._print_confusion_matrix(cm)

        # Per-class FPR
        fpr_dict = self._compute_fpr(all_labels, all_preds)
        print("\n  Per-class False Positive Rate (FPR):")
        for cls_name, fpr_val in fpr_dict.items():
            marker = " ⚠️" if fpr_val > 0.05 else " ✓"
            print(f"    {cls_name:30s} FPR = {fpr_val:.4f}{marker}")

        # Overall metrics
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        results = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "fpr": fpr_dict,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
        }

        print(f"\n  Overall: Acc={acc:.4f}  Prec={prec:.4f}  "
              f"Rec={rec:.4f}  F1={f1:.4f}")

        return results

    # ── Anomaly Detector Evaluation ────────────────────────

    def evaluate_anomaly_detector(
        self,
        detector,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float,
    ) -> dict:
        """Evaluate Isolation Forest anomaly detection on test set."""
        scores = detector.anomaly_score(X_test)

        # Determine benign vs malicious ground truth
        benign_classes = set()
        for i, name in enumerate(self.class_names):
            if name.startswith("benign"):
                benign_classes.add(i)

        is_benign_true = np.array([y in benign_classes for y in y_test])
        is_anomaly_pred = scores > threshold

        # Metrics
        tp = int(np.sum(~is_benign_true & is_anomaly_pred))   # malicious detected
        fp = int(np.sum(is_benign_true & is_anomaly_pred))    # benign flagged
        tn = int(np.sum(is_benign_true & ~is_anomaly_pred))   # benign passed
        fn = int(np.sum(~is_benign_true & ~is_anomaly_pred))  # malicious missed

        detection_rate = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * detection_rate / max(precision + detection_rate, 1e-9)

        print(f"\n{'='*60}")
        print(f"  ISOLATION FOREST ANOMALY DETECTION EVALUATION")
        print(f"{'='*60}")
        print(f"  Threshold:       {threshold:.4f}")
        print(f"  TP (malicious detected):  {tp}")
        print(f"  FP (benign flagged):      {fp}")
        print(f"  TN (benign passed):       {tn}")
        print(f"  FN (malicious missed):    {fn}")
        print(f"  Detection Rate:  {detection_rate:.4f}")
        print(f"  False Positive:  {fpr:.4f}")
        print(f"  Precision:       {precision:.4f}")
        print(f"  F1 Score:        {f1:.4f}")

        # Per-class anomaly score stats
        print(f"\n  Per-class anomaly score statistics:")
        print(f"  {'Class':30s} {'Mean':>10} {'Std':>10} {'Median':>10} {'>Thresh':>8}")
        for i, name in enumerate(self.class_names):
            mask = y_test == i
            if mask.sum() == 0:
                continue
            cls_scores = scores[mask]
            pct_above = np.mean(cls_scores > threshold) * 100
            print(f"  {name:30s} {np.mean(cls_scores):10.4f} "
                  f"{np.std(cls_scores):10.4f} {np.median(cls_scores):10.4f} "
                  f"{pct_above:7.1f}%")

        return {
            "scores": scores,
            "threshold": threshold,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "detection_rate": detection_rate,
            "fpr": fpr,
            "precision": precision,
            "f1": f1,
        }

    # ── Sample Predictions ───────────────────────────────

    def print_sample_predictions(
        self,
        model: nn.Module,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        n_samples: int = 10,
    ):
        """Print sample predictions vs ground truth."""
        model.eval()
        indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        X_sample = X_test[indices].to(self.device)
        y_sample = y_test[indices]

        with torch.no_grad():
            logits = model(X_sample)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1).cpu()
            confs = probs.max(dim=-1).values.cpu()

        print(f"\n{'='*60}")
        print(f"  SAMPLE PREDICTIONS")
        print(f"{'='*60}")
        print(f"  {'#':>3}  {'True Label':25s} {'Predicted':25s} {'Conf':>6} {'Match':>5}")
        print(f"  {'-'*70}")
        for i in range(len(indices)):
            true_name = self.class_names[y_sample[i]]
            pred_name = self.class_names[preds[i]]
            conf = confs[i].item()
            match = "✓" if preds[i] == y_sample[i] else "✗"
            print(f"  {i+1:3d}  {true_name:25s} {pred_name:25s} {conf:6.3f} {match:>5}")

    # ── Internal ─────────────────────────────────────────

    def _print_confusion_matrix(self, cm: np.ndarray):
        n = len(self.class_names)
        # Abbreviate names for matrix display
        short_names = [name[:12] for name in self.class_names]

        print("\n  Confusion Matrix:")
        # Header
        header = "  " + " " * 14 + "".join(f"{s:>13s}" for s in short_names)
        print(header)
        print("  " + " " * 14 + "-" * (13 * n))
        for i in range(n):
            row = f"  {short_names[i]:>12s} |"
            for j in range(n):
                val = cm[i, j]
                row += f"{val:12d} "
            print(row)

    def _compute_fpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute per-class false positive rate."""
        fpr_dict = {}
        classes = np.unique(np.concatenate([y_true, y_pred]))
        for cls_idx in classes:
            # FP: predicted as this class but actually not
            fp = int(np.sum((y_pred == cls_idx) & (y_true != cls_idx)))
            # TN: not predicted as this class and actually not
            tn = int(np.sum((y_pred != cls_idx) & (y_true != cls_idx)))
            fpr = fp / max(fp + tn, 1)
            fpr_dict[self.class_names[cls_idx]] = fpr
        return fpr_dict
