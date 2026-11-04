"""
Alert Manager — Session-based alert aggregation and fatigue reduction.

Reduces raw alert volume by:
1. Session grouping: same source IP + time window → 1 incident
2. Deduplication: same verdict + same source within cooldown → suppress
3. Escalation logic: only alert after N suspicious flows from same source
4. Severity filtering: configurable minimum severity for alerts
"""

from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class AlertIncident:
    incident_id: str
    source_ip: str
    first_seen: float
    last_seen: float
    flow_count: int = 1
    max_severity: str = "SUSPICIOUS"
    verdicts: list = field(default_factory=list)
    suppressed_count: int = 0


class AlertManager:
    """Aggregates and suppresses alerts to reduce operator fatigue."""

    def __init__(
        self,
        session_window_sec: float = 300.0,  # 5 min grouping window
        cooldown_sec: float = 60.0,         # 1 min cooldown per source
        min_suspicious_count: int = 3,      # Require 3 suspicious before alerting
        min_severity: str = "SUSPICIOUS",   # Minimum severity to surface
        max_incidents_per_hour: int = 50,   # Rate limit
    ):
        self.session_window = session_window_sec
        self.cooldown = cooldown_sec
        self.min_suspicious_count = min_suspicious_count
        self.min_severity = min_severity
        self.max_incidents_per_hour = max_incidents_per_hour

        self._incidents: dict[str, AlertIncident] = {}
        self._source_history: dict[str, list] = defaultdict(list)
        self._last_alert_time: dict[str, float] = {}
        self._hourly_count = 0
        self._hour_start = time.time()

        self._severity_order = {"SAFE": 0, "SUSPICIOUS": 1, "MALICIOUS": 2}
        self._stats = {
            "total_flows": 0, "total_alerts_raw": 0,
            "total_alerts_surfaced": 0, "total_suppressed": 0,
            "total_incidents": 0,
        }

    def process_verdict(self, flow_id: str, source_ip: str, verdict: str,
                        confidence: float, risk_score: float,
                        timestamp: float = None) -> dict | None:
        """Process a flow verdict. Returns an alert dict if one should surface."""
        ts = timestamp or time.time()
        self._stats["total_flows"] += 1

        # Only process non-SAFE verdicts
        if verdict == "SAFE":
            return None

        self._stats["total_alerts_raw"] += 1
        self._source_history[source_ip].append({
            "flow_id": flow_id, "verdict": verdict,
            "confidence": confidence, "risk_score": risk_score, "ts": ts,
        })

        # Check severity filter
        if self._severity_order.get(verdict, 0) < self._severity_order.get(self.min_severity, 1):
            self._stats["total_suppressed"] += 1
            return None

        # Check cooldown
        last_alert = self._last_alert_time.get(source_ip, 0)
        if ts - last_alert < self.cooldown and verdict != "MALICIOUS":
            self._stats["total_suppressed"] += 1
            # Update existing incident if within session window
            incident_key = f"{source_ip}_{int(last_alert)}"
            if incident_key in self._incidents:
                inc = self._incidents[incident_key]
                inc.last_seen = ts
                inc.flow_count += 1
                inc.suppressed_count += 1
                if self._severity_order.get(verdict, 0) > self._severity_order.get(inc.max_severity, 0):
                    inc.max_severity = verdict
            return None

        # Check escalation threshold for SUSPICIOUS
        if verdict == "SUSPICIOUS":
            recent = [h for h in self._source_history[source_ip]
                      if ts - h["ts"] < self.session_window]
            suspicious_count = sum(1 for h in recent if h["verdict"] in ("SUSPICIOUS", "MALICIOUS"))
            if suspicious_count < self.min_suspicious_count:
                self._stats["total_suppressed"] += 1
                return None

        # Rate limiting
        if ts - self._hour_start > 3600:
            self._hourly_count = 0
            self._hour_start = ts
        if self._hourly_count >= self.max_incidents_per_hour and verdict != "MALICIOUS":
            self._stats["total_suppressed"] += 1
            return None

        # Create or update incident
        incident_key = f"{source_ip}_{int(ts // self.session_window)}"
        if incident_key in self._incidents:
            inc = self._incidents[incident_key]
            inc.last_seen = ts
            inc.flow_count += 1
            inc.verdicts.append(verdict)
            if self._severity_order.get(verdict, 0) > self._severity_order.get(inc.max_severity, 0):
                inc.max_severity = verdict
            self._stats["total_suppressed"] += 1
            return None  # Already surfaced this incident
        else:
            inc = AlertIncident(
                incident_id=incident_key, source_ip=source_ip,
                first_seen=ts, last_seen=ts,
                max_severity=verdict, verdicts=[verdict],
            )
            self._incidents[incident_key] = inc
            self._last_alert_time[source_ip] = ts
            self._hourly_count += 1
            self._stats["total_alerts_surfaced"] += 1
            self._stats["total_incidents"] += 1

            return {
                "incident_id": incident_key,
                "source_ip": source_ip,
                "severity": verdict,
                "confidence": round(confidence, 4),
                "risk_score": round(risk_score, 4),
                "flow_id": flow_id,
                "timestamp": ts,
            }

    def get_stats(self) -> dict:
        suppression_rate = (self._stats["total_suppressed"] /
                           max(self._stats["total_alerts_raw"], 1) * 100)
        return {
            **self._stats,
            "suppression_rate_pct": round(suppression_rate, 1),
            "active_incidents": len(self._incidents),
        }

    def cleanup(self, max_age_sec: float = 3600):
        """Remove old incidents and history."""
        now = time.time()
        old_keys = [k for k, v in self._incidents.items() if now - v.last_seen > max_age_sec]
        for k in old_keys:
            del self._incidents[k]
        for ip in list(self._source_history.keys()):
            self._source_history[ip] = [
                h for h in self._source_history[ip] if now - h["ts"] < max_age_sec
            ]
            if not self._source_history[ip]:
                del self._source_history[ip]
