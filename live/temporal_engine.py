"""
Temporal Intelligence Engine — session-level behavioral reasoning.

Replaces the per-flow-only paradigm with stateful session analysis.

Key capabilities:
  1. Suspicion accumulation with exponential decay
  2. Cross-flow upload tracking for exfiltration detection
  3. Beacon periodicity detection via IAT autocorrelation
  4. Progressive verdict escalation based on behavioral history
  5. Per-source behavioral profiling with drift detection

Usage:
    engine = TemporalIntelligenceEngine()
    ctx = engine.update(source_ip, flow_verdict, flow_features)
    # ctx.verdict_floor   — minimum verdict level
    # ctx.suspicion_score — accumulated suspicion [0, 1]
    # ctx.escalation_reason — human-readable reason or None
"""

from __future__ import annotations

import time
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TemporalContext:
    """Result of temporal analysis for a single flow."""
    verdict_floor: Optional[str] = None  # None, "SUSPICIOUS", "MALICIOUS"
    suspicion_score: float = 0.0
    escalation_reason: Optional[str] = None
    session_flow_count: int = 0
    cumulative_upload_bytes: float = 0.0
    beacon_confidence: float = 0.0
    behavioral_drift: float = 0.0


@dataclass
class BeaconDetector:
    """Detects periodic beacon patterns across flows."""
    flow_timestamps: deque = field(default_factory=lambda: deque(maxlen=50))
    iat_history: deque = field(default_factory=lambda: deque(maxlen=50))

    def add_flow(self, timestamp: float):
        if self.flow_timestamps:
            iat = timestamp - self.flow_timestamps[-1]
            if 0.1 < iat < 600:  # Ignore sub-100ms or >10min gaps
                self.iat_history.append(iat)
        self.flow_timestamps.append(timestamp)

    @property
    def periodicity_confidence(self) -> float:
        """Estimate how periodic the flow arrival times are.

        Returns 0-1, where 1 means perfectly periodic (beacon-like).
        Uses coefficient of variation of IATs as primary signal.
        """
        if len(self.iat_history) < 4:
            return 0.0

        iats = np.array(self.iat_history)
        mean_iat = iats.mean()
        if mean_iat < 0.01:
            return 0.0

        cv = iats.std() / mean_iat  # Low CV = high periodicity

        # Also check autocorrelation at lag 1
        if len(iats) >= 6:
            centered = iats - mean_iat
            var = np.var(iats)
            if var > 1e-10:
                autocorr = np.correlate(centered[:-1], centered[1:])[0]
                autocorr /= (len(centered) - 1) * var
            else:
                autocorr = 0.0
        else:
            autocorr = 0.0

        # Combine: low CV and high autocorrelation → high confidence
        cv_score = max(0, 1.0 - cv)  # CV < 1 → periodic
        autocorr_score = max(0, autocorr)

        return float(np.clip(0.6 * cv_score + 0.4 * autocorr_score, 0, 1))


@dataclass
class SourceProfile:
    """Accumulated behavioral profile for a single source IP."""
    first_seen: float = 0.0
    last_seen: float = 0.0
    flow_count: int = 0
    suspicion_score: float = 0.0
    last_suspicion_update: float = 0.0

    # Verdict history
    verdict_history: deque = field(default_factory=lambda: deque(maxlen=100))
    prediction_history: deque = field(default_factory=lambda: deque(maxlen=100))
    risk_history: deque = field(default_factory=lambda: deque(maxlen=100))
    anomaly_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # Upload tracking
    cumulative_upload_bytes: float = 0.0
    upload_rate_history: deque = field(default_factory=lambda: deque(maxlen=30))

    # Beacon detection
    beacon: BeaconDetector = field(default_factory=BeaconDetector)

    # Behavioral feature statistics (rolling)
    feature_means: Optional[np.ndarray] = None
    feature_m2: Optional[np.ndarray] = None  # For Welford's online variance
    feature_count: int = 0


class TemporalIntelligenceEngine:
    """
    Stateful session-level reasoning engine.

    Maintains per-source-IP profiles and provides temporal context
    that influences verdict decisions in the inference pipeline.
    """

    # Suspicion decay half-life in seconds
    SUSPICION_HALF_LIFE = 300.0  # 5 minutes
    # Suspicion increments per verdict type
    SUSPICION_INCREMENT = {
        "SAFE": -0.02,
        "SUSPICIOUS": 0.15,
        "MALICIOUS": 0.35,
        "UNKNOWN": 0.05,
    }

    # Escalation thresholds
    ESCALATION_SUSPICION = 0.50     # Force SUSPICIOUS above this
    ESCALATION_MALICIOUS = 0.80     # Force MALICIOUS above this
    UPLOAD_ALERT_BYTES = 500_000    # 500 KB cumulative upload
    UPLOAD_RATE_ALERT = 50_000      # 50 KB/s sustained
    BEACON_ALERT_CONFIDENCE = 0.60  # Beacon detection threshold
    SUSTAINED_THREAT_COUNT = 3      # N suspicious+ flows in window
    SUSTAINED_THREAT_WINDOW = 300   # Window in seconds

    def __init__(self, max_sources: int = 10000):
        self.profiles: dict[str, SourceProfile] = {}
        self.max_sources = max_sources
        self._alerts: list[dict] = []
        self._eviction_count: int = 0

    def update(
        self,
        source_ip: str,
        verdict_str: str,
        risk_score: float,
        anomaly_score: float,
        prediction: str,
        flow_features: dict,
        timestamp: float = None,
    ) -> TemporalContext:
        """
        Update temporal state and return context for verdict adjustment.

        Call this AFTER the initial verdict decision but BEFORE the final
        output, so the temporal context can override/escalate.
        """
        now = timestamp or time.time()
        profile = self._get_or_create(source_ip, now)

        # Update profile
        profile.last_seen = now
        profile.flow_count += 1
        profile.verdict_history.append(verdict_str)
        profile.prediction_history.append(prediction)
        profile.risk_history.append(risk_score)
        profile.anomaly_history.append(anomaly_score)

        # Update suspicion with decay
        self._decay_suspicion(profile, now)
        increment = self.SUSPICION_INCREMENT.get(verdict_str, 0.0)

        # Bonus suspicion for malicious predictions even if verdict is SAFE
        if prediction.startswith("malicious_") and verdict_str == "SAFE":
            increment += 0.05

        profile.suspicion_score = float(np.clip(
            profile.suspicion_score + increment, 0.0, 1.0
        ))
        profile.last_suspicion_update = now

        # Update upload tracking
        upload_bytes = flow_features.get("total_bytes_fwd", 0.0)
        duration = max(flow_features.get("duration", 1.0), 0.01)
        profile.cumulative_upload_bytes += upload_bytes
        profile.upload_rate_history.append(upload_bytes / duration)

        # Update beacon detector
        profile.beacon.add_flow(now)

        # Update behavioral feature statistics (Welford's online algorithm)
        self._update_feature_stats(profile, flow_features)

        # Compute temporal context
        return self._compute_context(profile, now)

    def get_source_summary(self, source_ip: str) -> dict:
        """Get comprehensive summary for a source IP."""
        if source_ip not in self.profiles:
            return {"source": source_ip, "status": "unknown"}

        p = self.profiles[source_ip]
        return {
            "source": source_ip,
            "flow_count": p.flow_count,
            "suspicion_score": round(p.suspicion_score, 4),
            "cumulative_upload_bytes": p.cumulative_upload_bytes,
            "beacon_confidence": round(p.beacon.periodicity_confidence, 4),
            "session_duration": round(p.last_seen - p.first_seen, 1),
            "verdict_distribution": {
                v: list(p.verdict_history).count(v)
                for v in set(p.verdict_history)
            },
            "mean_risk": round(float(np.mean(list(p.risk_history))) if p.risk_history else 0, 4),
            "max_risk": round(float(max(p.risk_history)) if p.risk_history else 0, 4),
        }

    @property
    def alerts(self) -> list[dict]:
        return list(self._alerts)

    @property
    def active_sources(self) -> int:
        return len(self.profiles)

    @property
    def stats(self) -> dict:
        """Engine-level statistics for dashboard consumption."""
        total_flows = sum(p.flow_count for p in self.profiles.values())
        escalation_count = sum(
            1 for p in self.profiles.values()
            if p.suspicion_score >= self.ESCALATION_SUSPICION
        )
        beacon_count = sum(
            1 for p in self.profiles.values()
            if p.beacon.periodicity_confidence > self.BEACON_ALERT_CONFIDENCE
        )
        return {
            "active_sources": len(self.profiles),
            "total_flows": total_flows,
            "escalations": escalation_count,
            "beacon_alerts": beacon_count,
            "evictions": getattr(self, "_eviction_count", 0),
            "alerts": len(self._alerts),
        }

    def get_all_profiles(self) -> dict[str, SourceProfile]:
        """Return all active source profiles (for dashboard iteration)."""
        return dict(self.profiles)

    @property
    def escalation_log(self) -> list[dict]:
        """Return alert history suitable for dashboard display."""
        return list(self._alerts)

    # ── Internal ─────────────────────────────────────────

    def _get_or_create(self, source_ip: str, now: float) -> SourceProfile:
        if source_ip not in self.profiles:
            # Evict oldest if at capacity
            if len(self.profiles) >= self.max_sources:
                oldest_ip = min(self.profiles, key=lambda k: self.profiles[k].last_seen)
                del self.profiles[oldest_ip]
                self._eviction_count += 1

            self.profiles[source_ip] = SourceProfile(first_seen=now, last_seen=now)

        return self.profiles[source_ip]

    def _decay_suspicion(self, profile: SourceProfile, now: float):
        """Apply exponential decay to suspicion score."""
        if profile.last_suspicion_update > 0:
            elapsed = now - profile.last_suspicion_update
            if elapsed > 0:
                decay = math.exp(-math.log(2) * elapsed / self.SUSPICION_HALF_LIFE)
                profile.suspicion_score *= decay

    def _update_feature_stats(self, profile: SourceProfile, features: dict):
        """Welford's online algorithm for running mean and variance."""
        feature_keys = sorted(k for k in features.keys()
                              if isinstance(features[k], (int, float))
                              and k not in ("label", "flow_id", "source"))

        vals = np.array([features.get(k, 0.0) for k in feature_keys], dtype=np.float64)

        if profile.feature_means is None:
            profile.feature_means = vals.copy()
            profile.feature_m2 = np.zeros_like(vals)
            profile.feature_count = 1
        else:
            if len(vals) != len(profile.feature_means):
                return  # Feature vector size changed, skip
            profile.feature_count += 1
            delta = vals - profile.feature_means
            profile.feature_means += delta / profile.feature_count
            delta2 = vals - profile.feature_means
            profile.feature_m2 += delta * delta2

    def _compute_behavioral_drift(self, profile: SourceProfile, features: dict) -> float:
        """Measure how much the current flow deviates from the source's historical profile."""
        if profile.feature_count < 5 or profile.feature_means is None:
            return 0.0

        feature_keys = sorted(k for k in features.keys()
                              if isinstance(features[k], (int, float))
                              and k not in ("label", "flow_id", "source"))

        vals = np.array([features.get(k, 0.0) for k in feature_keys], dtype=np.float64)

        if len(vals) != len(profile.feature_means):
            return 0.0

        variance = profile.feature_m2 / max(profile.feature_count - 1, 1)
        std = np.sqrt(np.clip(variance, 1e-10, None))

        # Z-scores
        z_scores = np.abs(vals - profile.feature_means) / std
        # Mean z-score as drift measure
        return float(np.clip(np.mean(z_scores) / 3.0, 0, 1))  # Normalize to [0, 1]

    def _compute_context(self, profile: SourceProfile, now: float) -> TemporalContext:
        """Compute the temporal context that influences verdict decisions."""
        ctx = TemporalContext(
            suspicion_score=profile.suspicion_score,
            session_flow_count=profile.flow_count,
            cumulative_upload_bytes=profile.cumulative_upload_bytes,
            beacon_confidence=profile.beacon.periodicity_confidence,
        )

        reasons = []

        # Rule 1: Sustained suspicion → force minimum verdict
        if profile.suspicion_score >= self.ESCALATION_MALICIOUS:
            ctx.verdict_floor = "MALICIOUS"
            reasons.append(
                f"Sustained suspicion score {profile.suspicion_score:.2f} "
                f"exceeds MALICIOUS threshold ({self.ESCALATION_MALICIOUS})"
            )
        elif profile.suspicion_score >= self.ESCALATION_SUSPICION:
            ctx.verdict_floor = "SUSPICIOUS"
            reasons.append(
                f"Accumulated suspicion {profile.suspicion_score:.2f} "
                f"exceeds SUSPICIOUS threshold ({self.ESCALATION_SUSPICION})"
            )

        # Rule 2: Excessive cumulative upload → exfiltration alert
        if profile.cumulative_upload_bytes > self.UPLOAD_ALERT_BYTES:
            if ctx.verdict_floor is None:
                ctx.verdict_floor = "SUSPICIOUS"
            reasons.append(
                f"Cumulative upload {profile.cumulative_upload_bytes / 1024:.0f} KB "
                f"exceeds threshold ({self.UPLOAD_ALERT_BYTES / 1024:.0f} KB)"
            )

        # Rule 3: Sustained high upload rate → exfiltration
        if len(profile.upload_rate_history) >= 3:
            recent_rates = list(profile.upload_rate_history)[-5:]
            mean_rate = np.mean(recent_rates)
            if mean_rate > self.UPLOAD_RATE_ALERT:
                if ctx.verdict_floor != "MALICIOUS":
                    ctx.verdict_floor = "SUSPICIOUS"
                reasons.append(
                    f"Sustained upload rate {mean_rate / 1024:.1f} KB/s "
                    f"over {len(recent_rates)} flows"
                )

        # Rule 4: Beacon detection → C2 alert
        beacon_conf = profile.beacon.periodicity_confidence
        ctx.beacon_confidence = beacon_conf
        if beacon_conf > self.BEACON_ALERT_CONFIDENCE:
            if ctx.verdict_floor != "MALICIOUS":
                ctx.verdict_floor = "SUSPICIOUS"
            reasons.append(
                f"Beacon-like periodicity detected "
                f"(confidence: {beacon_conf:.2f})"
            )

        # Rule 5: Sustained threat — N suspicious+ flows in time window
        if len(profile.verdict_history) >= self.SUSTAINED_THREAT_COUNT:
            recent_verdicts = list(profile.verdict_history)[-10:]
            suspicious_count = sum(
                1 for v in recent_verdicts
                if v in ("SUSPICIOUS", "MALICIOUS")
            )
            if suspicious_count >= self.SUSTAINED_THREAT_COUNT:
                ctx.verdict_floor = "MALICIOUS"
                reasons.append(
                    f"Sustained threat: {suspicious_count}/{len(recent_verdicts)} "
                    f"recent flows flagged"
                )
                self._emit_alert("SUSTAINED_THREAT", profile, reasons[-1])

        # Rule 6: Behavioral drift detection
        # (Computed during _update_feature_stats, but we track it here)
        if profile.flow_count >= 5 and profile.feature_means is not None:
            # Check if recent anomaly scores are trending up
            if len(profile.anomaly_history) >= 5:
                recent_anomaly = list(profile.anomaly_history)[-5:]
                if all(recent_anomaly[i] <= recent_anomaly[i + 1]
                       for i in range(len(recent_anomaly) - 1)):
                    # Monotonically increasing anomaly → drift
                    ctx.behavioral_drift = 0.5
                    reasons.append("Anomaly scores trending upward (behavioral drift)")

        if reasons:
            ctx.escalation_reason = "; ".join(reasons)

        return ctx

    def _emit_alert(self, alert_type: str, profile: SourceProfile, detail: str):
        """Record a temporal alert."""
        self._alerts.append({
            "type": alert_type,
            "timestamp": time.time(),
            "flow_count": profile.flow_count,
            "suspicion_score": round(profile.suspicion_score, 4),
            "detail": detail,
        })
