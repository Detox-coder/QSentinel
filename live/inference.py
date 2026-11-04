"""
Inference engine: loads trained models and runs the full prediction pipeline.

Pipeline: raw features → scale → FFNN classifier + Isolation Forest → risk engine → verdict

v5 changes:
  - Uses feature list from the v5 checkpoint
  - Streaming-aware verdict logic (reduces false positives)
  - Higher confidence thresholds for MALICIOUS verdict
  - Confidence-weighted anomaly dampening
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk_engine import RiskBasedIDS
from live.flow_tracker import ALL_FEATURES, N_FEATURES, features_to_array


class Verdict(str, Enum):
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    MALICIOUS = "MALICIOUS"
    UNKNOWN = "UNKNOWN"


@dataclass
class FlowVerdict:
    """Structured per-flow analysis result."""
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    n_packets: int
    duration: float
    verdict: str
    prediction: str
    confidence: float
    risk_score: float
    anomaly_score: float
    is_uncertain: bool
    explanation: list[str]
    top_features: list[dict]
    timestamp: float

    def to_dict(self) -> dict:
        return asdict(self)


class InferenceEngine:
    """
    Loads models, scales features, delegates to RiskBasedIDS,
    and formats verdicts for the alert system.
    """

    def __init__(self, models_dir: str, device: torch.device = None, debug: bool = False):
        self.device = device or torch.device("cpu")
        self.debug = debug
        self._verdicts = {"SAFE": 0, "SUSPICIOUS": 0, "MALICIOUS": 0, "UNKNOWN": 0}
        self._load_models(models_dir)

    def _load_models(self, models_dir: str):
        self._ensure_sklearn_available()
        import joblib

        cls_file = "classifier_v5.pt"
        iforest_file = "iforest_v5.joblib"
        scaler_file, le_file = "scaler_v5.joblib", "label_encoder_v5.joblib"
        required = [cls_file, iforest_file, scaler_file, le_file]
        missing = [name for name in required if not os.path.exists(os.path.join(models_dir, name))]
        if missing:
            raise FileNotFoundError(f"Missing required model artifacts in {models_dir}: {', '.join(missing)}")
        self._version = "v5"

        self.scaler = joblib.load(os.path.join(models_dir, scaler_file))
        self.le = joblib.load(os.path.join(models_dir, le_file))
        self.class_names = list(self.le.classes_)

        # Classifier (FFNN only)
        cls_ckpt = torch.load(
            os.path.join(models_dir, cls_file),
            map_location=self.device, weights_only=False,
        )
        n_features = cls_ckpt["n_features"]
        n_classes = cls_ckpt["n_classes"]

        from training.classifier import QUICClassifier
        self.classifier = QUICClassifier(n_features, n_classes, hidden_dims=(256, 128, 64), dropout=0.45).to(self.device)
        self.classifier.load_state_dict(cls_ckpt["model_state"])
        self.classifier.eval()

        # Isolation Forest anomaly detector
        from training.isolation_forest import QUICIsolationForest
        self.anomaly_detector = joblib.load(os.path.join(models_dir, iforest_file))
        # Use a default threshold for IForest (anomaly_score > threshold = anomalous)
        anomaly_threshold = cls_ckpt.get("anomaly_threshold", 0.5)

        # Feature list from checkpoint, kept aligned with scaler and classifier.
        self.features = cls_ckpt.get("features", ALL_FEATURES)

        self.ids = RiskBasedIDS(
            classifier=self.classifier,
            anomaly_detector=self.anomaly_detector,
            class_names=self.class_names,
            feature_names=self.features,
            anomaly_threshold=anomaly_threshold,
            device=self.device,
        )

        # v6: Temporal intelligence engine
        from live.temporal_engine import TemporalIntelligenceEngine
        self.temporal_engine = TemporalIntelligenceEngine()

        # v6: Load calibration temperature if saved in checkpoint
        self._temperature = cls_ckpt.get("temperature", 1.0)

        print(f"[ENGINE] Loaded {self._version} FFNN: {n_classes} classes, {n_features} features")
        print(f"[ENGINE] Anomaly detector: Isolation Forest")
        print(f"[ENGINE] Anomaly threshold: {anomaly_threshold:.4f}")
        print(f"[ENGINE] Calibration temperature: {self._temperature:.4f}")
        print(f"[ENGINE] Temporal engine: active")

    @staticmethod
    def _ensure_sklearn_available() -> None:
        """
        Some local setups have CPU Torch in system Python and sklearn in the
        project venv. Torch is imported before this method, so appending the
        venv site-packages lets joblib unpickle the scaler without shadowing
        the already-loaded Torch module.
        """
        try:
            import sklearn  # noqa: F401
            return
        except ModuleNotFoundError:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            candidates = [
                os.path.join(project_root, ".venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"),
                os.path.join(project_root, ".venv", "Lib", "site-packages"),
            ]
            for path in candidates:
                if os.path.isdir(path) and path not in sys.path:
                    sys.path.append(path)
            import sklearn  # noqa: F401

    def analyze(self, flow_record, features_dict: dict) -> FlowVerdict:
        """Run full analysis on a single flow using RiskBasedIDS + temporal intelligence."""
        # Scale — use the feature list from the model checkpoint
        raw = features_to_array(features_dict, feature_names=self.features).reshape(1, -1)

        # v6: Runtime feature clipping to handle OOD values before scaling
        # Clip to reasonable ranges to prevent scaler explosion
        raw = np.clip(raw, -1e7, 1e7)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        scaled = self.scaler.transform(raw).astype(np.float32)
        # Post-scaling safety clamp (prevent extreme z-scores)
        scaled = np.clip(scaled, -6.0, 6.0)

        if self.debug:
            print(f"\n[DEBUG-ML] Flow: {flow_record.flow_id}")
            print(f"[DEBUG-ML] Raw Features: {dict(zip(self.features, raw[0]))}")
            print(f"[DEBUG-ML] Scaled Features: {dict(zip(self.features, scaled[0]))}")

        # Delegate to RiskBasedIDS
        r_verdict = self.ids.analyze_single(scaled, flow_id=flow_record.flow_id, source_id=flow_record.src_ip)

        # v6: Apply temperature calibration if available
        if hasattr(self, '_temperature') and self._temperature != 1.0:
            # Recalibrate the confidence using temperature
            r_verdict.confidence = min(r_verdict.confidence, r_verdict.confidence ** (1.0 / self._temperature))

        if self.debug:
            print(f"[DEBUG-ML] Prediction: {r_verdict.prediction} ({r_verdict.confidence:.4f})")
            print(f"[DEBUG-ML] Anomaly Score: {r_verdict.anomaly_score:.4f} (Threshold: {self.ids.anomaly_threshold:.4f})")
            print(f"[DEBUG-ML] Risk Score: {r_verdict.risk_score:.4f} (Level: {r_verdict.risk_level})")
            print(f"[DEBUG-ML] Decision Source: {r_verdict.decision_source}")

        # ── v6.1: Confidence-weighted fusion verdict ─────────────
        #
        # KEY DESIGN PRINCIPLE: The anomaly detector must NEVER unilaterally
        # override a high-confidence benign classifier prediction. The
        # classifier is the primary decision maker; the IForest is a
        # secondary signal that can ESCALATE but not DICTATE.
        #
        is_benign = r_verdict.prediction.startswith("benign")
        is_malicious_pred = r_verdict.prediction.startswith("malicious_")
        raw_anomaly = r_verdict.anomaly_score
        conf = r_verdict.confidence

        # Confidence-weighted anomaly dampening:
        # High-confidence benign predictions strongly suppress anomaly.
        # At conf=0.98 → dampening=0.1, conf=0.70 → dampening=0.5
        if is_benign and conf > 0.7:
            dampening = max(0.1, 0.5 - 0.4 * ((conf - 0.7) / 0.3))
        elif is_benign and conf > 0.5:
            dampening = 0.6
        else:
            dampening = 1.0
        effective_anomaly = raw_anomaly * dampening
        high_anomaly = effective_anomaly > self.ids.anomaly_threshold

        # Step 1: Classifier-driven base verdict
        if is_malicious_pred and conf > 0.60:
            verdict_str = Verdict.MALICIOUS.value
        elif is_malicious_pred:
            verdict_str = Verdict.SUSPICIOUS.value
        elif is_benign and conf > 0.70:
            verdict_str = Verdict.SAFE.value
        else:
            # Low-confidence benign or unknown
            verdict_str = Verdict.SUSPICIOUS.value if high_anomaly else Verdict.SAFE.value

        # Step 2: Anomaly escalation (NEVER to MALICIOUS alone)
        # High-confidence benign (>0.7) requires 3x threshold for escalation.
        # This prevents the IForest from overriding confident classifier decisions.
        if high_anomaly and verdict_str == Verdict.SAFE.value:
            escalate = True
            if is_benign and conf > 0.7:
                # Only escalate for extreme outliers
                escalate = effective_anomaly > self.ids.anomaly_threshold * 3.0
            if escalate:
                verdict_str = Verdict.SUSPICIOUS.value
                r_verdict.decision_source = "anomaly_escalation"
                r_verdict.explanation.append(
                    f"Dampened anomaly ({effective_anomaly:.1f}) exceeds threshold "
                    f"(raw: {raw_anomaly:.1f}, dampening: {dampening:.2f})"
                )

        # Step 3: Malicious prediction floor — never fully SAFE
        if is_malicious_pred and verdict_str == Verdict.SAFE.value:
            verdict_str = Verdict.SUSPICIOUS.value
            r_verdict.decision_source = "malicious_prediction_floor"

        # Step 4: v6 Temporal intelligence integration
        temporal_ctx = None
        if hasattr(self, 'temporal_engine'):
            temporal_ctx = self.temporal_engine.update(
                source_ip=flow_record.src_ip,
                verdict_str=verdict_str,
                risk_score=r_verdict.risk_score,
                anomaly_score=r_verdict.anomaly_score,
                prediction=r_verdict.prediction,
                flow_features=features_dict,
            )

            # Apply temporal verdict floor
            if temporal_ctx.verdict_floor == "MALICIOUS":
                if verdict_str != Verdict.MALICIOUS.value:
                    verdict_str = Verdict.MALICIOUS.value
                    r_verdict.decision_source = "temporal_escalation"
                    r_verdict.explanation.append(
                        f"[TEMPORAL] {temporal_ctx.escalation_reason}"
                    )
            elif temporal_ctx.verdict_floor == "SUSPICIOUS":
                if verdict_str == Verdict.SAFE.value:
                    verdict_str = Verdict.SUSPICIOUS.value
                    r_verdict.decision_source = "temporal_suspicion"
                    r_verdict.explanation.append(
                        f"[TEMPORAL] {temporal_ctx.escalation_reason}"
                    )

        self._verdicts[verdict_str] += 1

        return FlowVerdict(
            flow_id=flow_record.flow_id,
            src_ip=flow_record.src_ip,
            dst_ip=flow_record.dst_ip,
            src_port=flow_record.src_port,
            dst_port=flow_record.dst_port,
            n_packets=flow_record.n_packets,
            duration=round(flow_record.duration, 4),
            verdict=verdict_str,
            prediction=r_verdict.prediction,
            confidence=round(r_verdict.confidence, 4),
            risk_score=round(r_verdict.risk_score, 4),
            anomaly_score=round(r_verdict.anomaly_score, 4),
            is_uncertain=r_verdict.is_blend_suspect or r_verdict.decision_source == "low_confidence",
            explanation=r_verdict.explanation,
            top_features=r_verdict.top_features,
            timestamp=r_verdict.timestamp,
        )

    @property
    def stats(self) -> dict:
        ids_stats = self.ids.get_stats()
        result = {
            "total_analyzed": ids_stats["total_analyzed"],
            "alerts_raised": ids_stats["alerts_raised"],
            "verdicts": dict(self._verdicts),
        }
        if hasattr(self, 'temporal_engine'):
            result["temporal_sources"] = self.temporal_engine.active_sources
        return result

