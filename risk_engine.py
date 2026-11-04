from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class FlowVerdict:
    flow_id: str
    prediction: str
    prediction_idx: int
    confidence: float
    risk_score: float
    risk_level: str
    anomaly_score: float
    is_blend_suspect: bool
    is_anomalous: bool
    decision_source: str
    explanation: list[str]
    top_features: list[dict]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self):
        return (
            f"FlowVerdict(id={self.flow_id}, pred={self.prediction}, "
            f"risk={self.risk_score:.3f}/{self.risk_level}, "
            f"conf={self.confidence:.3f}, blend={self.is_blend_suspect})"
        )

class RiskScoringEngine:

    def __init__(
        self,
        anomaly_threshold: float,
        w_uncertainty: float = 0.40,
        w_anomaly: float = 0.25,
        w_blend: float = 0.10,
        w_entropy: float = 0.25,
        blend_conf_range: tuple[float, float] = (0.30, 0.70),
        blend_anomaly_ratio: float = 0.6,
    ):
        self.anomaly_threshold = anomaly_threshold
        self.w_uncertainty = w_uncertainty
        self.w_anomaly = w_anomaly
        self.w_blend = w_blend
        self.w_entropy = w_entropy
        self.blend_conf_range = blend_conf_range
        self.blend_anomaly_ratio = blend_anomaly_ratio

    def compute(
        self,
        confidence: np.ndarray,
        anomaly_scores: np.ndarray,
        class_probs: np.ndarray,
        is_pred_benign: np.ndarray,
    ) -> dict:

        n = len(confidence)
        uncertainty = 1.0 - confidence
        norm_anomaly = np.clip(anomaly_scores / (self.anomaly_threshold * 2), 0, 1)

        blend_scores = np.zeros(n)
        for i in range(n):
            conf = confidence[i]
            anom = anomaly_scores[i]
            in_conf_range = (self.blend_conf_range[0] <= conf <= self.blend_conf_range[1])
            has_anomaly = anom > self.anomaly_threshold * self.blend_anomaly_ratio
            if in_conf_range and has_anomaly:
                # Both conditions met: genuine blend suspect
                conf_center = 0.5
                conf_dist = max(0, 1.0 - abs(conf - conf_center) / 0.20)
                anom_ratio = min(anom / self.anomaly_threshold, 1.0)
                blend_scores[i] = conf_dist * 0.4 + anom_ratio * 0.6
            elif has_anomaly and is_pred_benign[i]:
                # Anomaly on benign prediction: moderate blend signal
                blend_scores[i] = 0.4
            # NOTE: Removed the 0.3 floor for confidence-only (no anomaly) case.
            # This was the #1 false positive source (59% of all FPs).

        entropy = -np.sum(class_probs * np.log(class_probs + 1e-10), axis=1)
        max_entropy = np.log(class_probs.shape[1])
        norm_entropy = entropy / max_entropy

        risk_scores = (
            self.w_uncertainty * uncertainty
            + self.w_anomaly * norm_anomaly
            + self.w_blend * blend_scores
            + self.w_entropy * norm_entropy
        )
        risk_scores = np.clip(risk_scores, 0, 1)

        risk_levels = []
        for score in risk_scores:
            if score < 0.30:
                risk_levels.append(RiskLevel.LOW.value)
            elif score < 0.55:
                risk_levels.append(RiskLevel.MEDIUM.value)
            elif score < 0.80:
                risk_levels.append(RiskLevel.HIGH.value)
            else:
                risk_levels.append(RiskLevel.CRITICAL.value)

        blend_suspects = blend_scores > 0.40

        return {
            "risk_scores": risk_scores,
            "risk_levels": risk_levels,
            "blend_suspects": blend_suspects,
            "components": {
                "uncertainty": uncertainty,
                "norm_anomaly": norm_anomaly,
                "blend_scores": blend_scores,
                "norm_entropy": norm_entropy,
            },
        }



class FeatureExplainer:

    def __init__(self, classifier: nn.Module, feature_names: list[str]):
        self.classifier = classifier
        self.feature_names = feature_names

    def explain(self, X: torch.Tensor, predictions: np.ndarray,
                top_k: int = 5) -> list[list[dict]]:

        self.classifier.eval()
        X_grad = X.clone().detach().requires_grad_(True)

        logits = self.classifier(X_grad)

        pred_tensor = torch.tensor(predictions, dtype=torch.long, device=X.device)
        target_logits = logits.gather(1, pred_tensor.unsqueeze(1)).squeeze(1)
        target_logits.sum().backward()

        grads = X_grad.grad.data.cpu().numpy()
        inputs = X.detach().cpu().numpy()
        saliency = np.abs(grads * inputs)

        results = []
        for i in range(len(X)):
            top_idx = np.argsort(saliency[i])[::-1][:top_k]
            features = []
            for j in top_idx:
                features.append({
                    "feature": self.feature_names[j],
                    "importance": float(saliency[i, j]),
                    "value": float(inputs[i, j]),
                    "direction": "↑" if grads[i, j] > 0 else "↓",
                })
            results.append(features)

        return results


class TemporalTracker:

    def __init__(self, window_size: int = 20, escalation_threshold: int = 3):
        self.window_size = window_size
        self.escalation_threshold = escalation_threshold
        self.source_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.alerts: list[dict] = []

    def update(self, source_id: str, verdict: FlowVerdict) -> Optional[dict]:
        history = self.source_history[source_id]
        history.append({
            "flow_id": verdict.flow_id,
            "risk_level": verdict.risk_level,
            "risk_score": verdict.risk_score,
            "prediction": verdict.prediction,
            "timestamp": verdict.timestamp,
        })

        if len(history) < 3:
            return None

        recent = list(history)[-5:]
        levels = [h["risk_level"] for h in recent]
        scores = [h["risk_score"] for h in recent]

        level_values = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        level_nums = [level_values.get(l, 0) for l in levels]

        if len(level_nums) >= 3:
            increasing_count = sum(
                1 for i in range(1, len(level_nums))
                if level_nums[i] > level_nums[i - 1]
            )
            max_level = max(level_nums)

            if increasing_count >= 2 and max_level >= 2:
                alert = {
                    "type": "ESCALATION",
                    "source": source_id,
                    "pattern": " → ".join(levels),
                    "current_risk": levels[-1],
                    "score_trend": [round(s, 3) for s in scores],
                    "flow_count": len(history),
                    "severity": "HIGH" if max_level >= 3 else "MEDIUM",
                }
                self.alerts.append(alert)
                return alert

        high_count = sum(1 for l in levels if l in ("HIGH", "CRITICAL"))
        if high_count >= self.escalation_threshold:
            alert = {
                "type": "SUSTAINED_THREAT",
                "source": source_id,
                "pattern": " → ".join(levels),
                "current_risk": levels[-1],
                "high_flow_count": high_count,
                "severity": "CRITICAL" if any(l == "CRITICAL" for l in levels) else "HIGH",
            }
            self.alerts.append(alert)
            return alert

        return None

    def get_source_summary(self, source_id: str) -> dict:
        """Get risk summary for a source."""
        history = list(self.source_history.get(source_id, []))
        if not history:
            return {"source": source_id, "flow_count": 0}

        scores = [h["risk_score"] for h in history]
        levels = [h["risk_level"] for h in history]
        return {
            "source": source_id,
            "flow_count": len(history),
            "mean_risk": float(np.mean(scores)),
            "max_risk": float(np.max(scores)),
            "risk_distribution": {
                l: levels.count(l) for l in set(levels)
            },
        }



class RiskBasedIDS:

    def __init__(
        self,
        classifier: nn.Module,
        anomaly_detector,
        class_names: list[str],
        feature_names: list[str],
        anomaly_threshold: float,
        device: torch.device = torch.device("cpu"),
    ):
        self.classifier = classifier.to(device)
        self.anomaly_detector = anomaly_detector
        self.class_names = class_names
        self.feature_names = feature_names
        self.device = device

        self._benign_ids = {i for i, n in enumerate(class_names) if n.startswith("benign")}
        self._malicious_ids = {i for i, n in enumerate(class_names) if not n.startswith("benign")}

        self.risk_engine = RiskScoringEngine(anomaly_threshold)
        self.explainer = FeatureExplainer(classifier, feature_names)
        self.temporal = TemporalTracker()
        self.anomaly_threshold = anomaly_threshold

        self._total_analyzed = 0
        self._alerts_raised = 0

    def analyze_batch(
        self,
        X: np.ndarray,
        flow_ids: Optional[list[str]] = None,
        source_ids: Optional[list[str]] = None,
    ) -> list[FlowVerdict]:

        n = len(X)
        if flow_ids is None:
            flow_ids = [f"flow_{self._total_analyzed + i}" for i in range(n)]
        if source_ids is None:
            source_ids = ["unknown"] * n

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.classifier.eval()

        logits = self.classifier(X_t)
        probs = F.softmax(logits, dim=1).detach()
        confidence, predictions = probs.max(dim=1)
        confidence_np = confidence.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        probs_np = probs.cpu().numpy()

        # Isolation Forest anomaly scoring (numpy-based)
        anomaly_scores = self.anomaly_detector.anomaly_score(X)

        is_pred_benign = np.array([p in self._benign_ids for p in predictions_np])
        risk = self.risk_engine.compute(
            confidence_np, anomaly_scores, probs_np, is_pred_benign,
        )

        explanations = self.explainer.explain(X_t, predictions_np, top_k=5)

        verdicts = []
        for i in range(n):
            pred_idx = int(predictions_np[i])
            pred_name = self.class_names[pred_idx]
            conf = float(confidence_np[i])
            r_score = float(risk["risk_scores"][i])
            r_level = risk["risk_levels"][i]
            is_blend = bool(risk["blend_suspects"][i])
            is_anom = bool(anomaly_scores[i] > self.anomaly_threshold)

            if is_anom and pred_idx in self._benign_ids:
                source = "iforest_flagged"
            elif is_blend:
                source = "blend_detection"
            elif conf < 0.50:
                source = "low_confidence"
            else:
                source = "classifier"

            explain_strs = []
            if is_anom:
                explain_strs.append(
                    f"IForest anomaly score ({anomaly_scores[i]:.1f}) exceeds threshold ({self.anomaly_threshold:.1f})"
                )
            if is_blend:
                explain_strs.append(
                    f"Blend suspect: moderate confidence ({conf:.3f}) with elevated anomaly"
                )
            if r_level in ("HIGH", "CRITICAL"):
                explain_strs.append(
                    f"Risk components: uncertainty={risk['components']['uncertainty'][i]:.3f}, "
                    f"anomaly={risk['components']['norm_anomaly'][i]:.3f}, "
                    f"entropy={risk['components']['norm_entropy'][i]:.3f}"
                )
            if not explain_strs:
                explain_strs.append(f"Normal classification with {conf:.1%} confidence")

            verdict = FlowVerdict(
                flow_id=flow_ids[i],
                prediction=pred_name,
                prediction_idx=pred_idx,
                confidence=conf,
                risk_score=r_score,
                risk_level=r_level,
                anomaly_score=float(anomaly_scores[i]),
                is_blend_suspect=is_blend,
                is_anomalous=is_anom,
                decision_source=source,
                explanation=explain_strs,
                top_features=explanations[i],
            )
            verdicts.append(verdict)

            alert = self.temporal.update(source_ids[i], verdict)
            if alert:
                self._alerts_raised += 1

        self._total_analyzed += n
        return verdicts

    def analyze_single(self, X: np.ndarray, flow_id: str = "flow",
                       source_id: str = "unknown") -> FlowVerdict:
        """Analyze a single flow."""
        return self.analyze_batch(
            X.reshape(1, -1), [flow_id], [source_id],
        )[0]

    def get_stats(self) -> dict:
        return {
            "total_analyzed": self._total_analyzed,
            "alerts_raised": self._alerts_raised,
            "active_sources": len(self.temporal.source_history),
        }

    def summary(self) -> str:
        return (
            f"RiskBasedIDS:\n"
            f"  Features: {len(self.feature_names)}\n"
            f"  Classes: {len(self.class_names)} ({', '.join(self.class_names)})\n"
            f"  Anomaly threshold: {self.anomaly_threshold:.4f}\n"
            f"  Device: {self.device}\n"
            f"  Total analyzed: {self._total_analyzed}\n"
            f"  Alerts raised: {self._alerts_raised}"
        )
