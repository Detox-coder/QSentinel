"""
Ensemble decision system combining classifier confidence + Isolation Forest anomaly score.

Decision logic:
  1. If IForest anomaly score > threshold → flag as anomalous regardless of classifier
  2. If classifier confidence < min_confidence → flag as suspicious
  3. Otherwise → use classifier prediction

Output: (prediction, confidence, is_anomalous, decision_source)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EnsembleDecisionSystem:
    """
    Combines QUICClassifier + QUICIsolationForest into a robust decision system.

    Classifier provides class predictions with confidence.
    Isolation Forest provides anomaly detection independent of class labels.
    """

    def __init__(self, classifier, anomaly_detector, class_names: list,
                 anomaly_threshold: float,
                 min_confidence: float = 0.70,
                 anomaly_weight: float = 0.3):
        self.classifier = classifier
        self.anomaly_detector = anomaly_detector
        self.class_names = class_names
        self.anomaly_threshold = anomaly_threshold
        self.min_confidence = min_confidence
        self.anomaly_weight = anomaly_weight
        self._benign_indices = {
            i for i, n in enumerate(class_names) if n.startswith("benign")
        }
        self._malicious_indices = {
            i for i, n in enumerate(class_names) if not n.startswith("benign")
        }

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> dict:
        """
        Full ensemble prediction.

        Returns dict with:
          - prediction: class indices
          - confidence: per-sample confidence
          - anomaly_score: IForest anomaly score
          - is_anomalous: boolean per sample
          - decision_source: 'classifier', 'iforest_override', or 'low_confidence'
        """
        self.classifier.eval()

        # Classifier output
        logits = self.classifier(X)
        probs = F.softmax(logits, dim=1)
        cls_confidence, cls_pred = probs.max(dim=1)

        # Isolation Forest anomaly score (operates on numpy)
        X_np = X.cpu().numpy()
        anomaly_scores_np = self.anomaly_detector.anomaly_score(X_np)
        is_anomalous_np = anomaly_scores_np > self.anomaly_threshold

        # Decision fusion
        n = X.shape[0]
        final_pred = cls_pred.clone()
        final_confidence = cls_confidence.clone()
        sources = ["classifier"] * n

        for i in range(n):
            pred_class = cls_pred[i].item()
            conf = cls_confidence[i].item()
            anom = is_anomalous_np[i]
            score = anomaly_scores_np[i]

            # Rule 1: IForest says anomalous but classifier says benign → override
            if anom and pred_class in self._benign_indices:
                # Find highest-confidence malicious class
                mal_probs = [(j, probs[i, j].item()) for j in self._malicious_indices]
                if mal_probs:
                    best_mal = max(mal_probs, key=lambda x: x[1])
                    final_pred[i] = best_mal[0]
                    final_confidence[i] = 1.0 - (conf * (1 - self.anomaly_weight))
                    sources[i] = "iforest_override"

            # Rule 2: Low classifier confidence → suspicious
            elif conf < self.min_confidence:
                # Use weighted combination: boost malicious if anomaly score is high
                if score > self.anomaly_threshold * 0.5:
                    mal_probs = [(j, probs[i, j].item()) for j in self._malicious_indices]
                    if mal_probs:
                        best_mal = max(mal_probs, key=lambda x: x[1])
                        final_pred[i] = best_mal[0]
                        sources[i] = "low_confidence"

        return {
            "prediction": final_pred.cpu().numpy(),
            "confidence": final_confidence.cpu().numpy(),
            "anomaly_score": anomaly_scores_np,
            "is_anomalous": is_anomalous_np,
            "decision_source": sources,
            "cls_pred": cls_pred.cpu().numpy(),
            "cls_confidence": cls_confidence.cpu().numpy(),
        }

    @torch.no_grad()
    def predict_simple(self, X: torch.Tensor) -> np.ndarray:
        """Quick prediction returning just class indices."""
        return self.predict(X)["prediction"]

    def summary(self) -> str:
        return (
            f"EnsembleDecisionSystem:\n"
            f"  Classes: {len(self.class_names)}\n"
            f"  Anomaly threshold: {self.anomaly_threshold:.4f}\n"
            f"  Min confidence: {self.min_confidence:.2f}\n"
            f"  Anomaly weight: {self.anomaly_weight:.2f}\n"
            f"  Benign classes: {sorted(self.class_names[i] for i in self._benign_indices)}\n"
            f"  Malicious classes: {sorted(self.class_names[i] for i in self._malicious_indices)}"
        )
