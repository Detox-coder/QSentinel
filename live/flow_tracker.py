"""
Flow tracker: groups raw packets into flows and extracts the v5 feature vector.

Flow = 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol).
Completion: timeout (no new packets) or max_packets reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from live.packet_reader import RawPacket

# ── Feature names: must match training pipeline ──────────
SHARED_FEATURES = [
    "duration", "total_packets", "total_bytes_fwd",
    "pkts_per_sec", "bytes_per_sec",
    "all_size_mean", "all_size_std", "all_size_min", "all_size_max",
    "all_size_median", "all_size_skew", "all_size_kurtosis",
    "all_size_q25", "all_size_q75",
    "all_iat_mean", "all_iat_std", "all_iat_min", "all_iat_max",
    "all_iat_skew", "all_iat_kurtosis", "timing_regularity",
    "small_pkt_ratio", "large_pkt_ratio", "size_entropy",
]
MARKOV_FEATURES = ["markov_fwd_fwd", "markov_fwd_bwd", "markov_bwd_fwd", "markov_bwd_bwd"]
TUNNEL_FEATURES = ["iat_cv", "size_iat_ratio", "burst_score", "direction_proxy"]
BIDIRECTIONAL_FEATURES = ["up_down_ratio", "directional_entropy", "req_resp_cadence", "alt_burst_score"]
EXFIL_FEATURES = ["fwd_bytes_per_sec", "bwd_bytes_per_sec", "upload_ratio"]
# v6: Advanced behavioral features for real-world robustness
BEHAVIORAL_FEATURES = [
    "session_asymmetry",        # Upload/download ratio over windowed segments
    "upload_consistency",       # Std of per-segment upload ratios
    "beacon_periodicity",       # Autocorrelation-based IAT regularity
    "temporal_volatility",      # Variance of windowed throughput
    "burst_acceleration",       # Trend of burst sizes over time
    "interaction_density",      # Packets per direction-change episode
    "directional_persistence",  # Longest same-direction run / total packets
    "silence_ratio",            # Fraction of flow with no activity
    "size_modality",            # Number of modes in packet size distribution
    "response_latency_cv",      # CV of request-response gaps
    "iat_trend",                # Linear trend (slope) of IATs over time
    "size_trend",               # Linear trend (slope) of packet sizes over time
]
# Default: all live features. Inference engine uses the checkpoint feature order.
ALL_FEATURES = (SHARED_FEATURES + MARKOV_FEATURES + TUNNEL_FEATURES
                + BIDIRECTIONAL_FEATURES + EXFIL_FEATURES + BEHAVIORAL_FEATURES)
N_FEATURES = len(ALL_FEATURES)



@dataclass
class FlowRecord:
    """Accumulated state for a single flow."""
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    first_seen: float
    last_seen: float
    packet_sizes: list[int] = field(default_factory=list)
    packet_timestamps: list[float] = field(default_factory=list)
    fwd_sizes: list[int] = field(default_factory=list)
    packet_directions: list[int] = field(default_factory=list)  # +1 fwd, -1 bwd

    @property
    def n_packets(self) -> int:
        return len(self.packet_sizes)

    @property
    def duration(self) -> float:
        return max(self.last_seen - self.first_seen, 1e-6)


def _safe(arr, func, default=0.0):
    if len(arr) < 2:
        return default
    try:
        v = func(arr)
        return default if (np.isnan(v) or np.isinf(v)) else float(v)
    except Exception:
        return default


def _entropy(arr):
    if len(arr) < 2:
        return 0.0
    _, counts = np.unique(
        np.digitize(arr, np.linspace(arr.min(), arr.max() + 1e-9, 20)),
        return_counts=True,
    )
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-10)))


def _markov_stats(directions: list[int]) -> dict[str, float]:
    """1st-order Markov transition probabilities for packet direction."""
    n = len(directions)
    if n < 2:
        return {"markov_fwd_fwd": 0.0, "markov_fwd_bwd": 0.0, "markov_bwd_fwd": 0.0, "markov_bwd_bwd": 0.0}

    transitions = {(1, 1): 0, (1, -1): 0, (-1, 1): 0, (-1, -1): 0}
    for i in range(1, n):
        d1 = 1 if directions[i - 1] > 0 else -1
        d2 = 1 if directions[i] > 0 else -1
        transitions[(d1, d2)] += 1

    fwd_total = max(transitions[(1, 1)] + transitions[(1, -1)], 1)
    bwd_total = max(transitions[(-1, 1)] + transitions[(-1, -1)], 1)

    return {
        "markov_fwd_fwd": transitions[(1, 1)] / fwd_total,
        "markov_fwd_bwd": transitions[(1, -1)] / fwd_total,
        "markov_bwd_fwd": transitions[(-1, 1)] / bwd_total,
        "markov_bwd_bwd": transitions[(-1, -1)] / bwd_total,
    }


def extract_features(flow: FlowRecord) -> dict[str, float]:
    """Extract the 32-feature vector from a completed flow."""
    sizes = np.array(flow.packet_sizes, dtype=np.float64)
    times = np.array(flow.packet_timestamps, dtype=np.float64)
    fwd = np.array(flow.fwd_sizes, dtype=np.float64) if flow.fwd_sizes else sizes

    iats = np.diff(times) if len(times) > 1 else np.array([0.0])
    iats = iats[iats >= 0]
    if len(iats) == 0:
        iats = np.array([0.0])

    dur = flow.duration
    n = len(sizes)
    total_fwd = float(fwd.sum()) if len(fwd) > 0 else float(sizes.sum())

    iat_mean = float(iats.mean())
    iat_std = float(iats.std()) if len(iats) > 1 else 0.0
    size_mean = float(sizes.mean()) if n > 0 else 0.0
    size_median = float(np.median(sizes)) if n > 0 else 0.0
    q25 = float(np.percentile(sizes, 25)) if n > 0 else 0.0
    q75 = float(np.percentile(sizes, 75)) if n > 0 else 0.0
    size_skew = _safe(sizes, sp_stats.skew)
    iat_skew = _safe(iats, sp_stats.skew)

    f = {
        "duration": dur,
        "total_packets": float(n),
        "total_bytes_fwd": total_fwd,
        "pkts_per_sec": n / dur if dur > 0 else 0.0,
        "bytes_per_sec": total_fwd / dur if dur > 0 else 0.0,
        "all_size_mean": size_mean,
        "all_size_std": float(sizes.std()) if n > 1 else 0.0,
        "all_size_min": float(sizes.min()) if n > 0 else 0.0,
        "all_size_max": float(sizes.max()) if n > 0 else 0.0,
        "all_size_median": size_median,
        "all_size_skew": size_skew,
        "all_size_kurtosis": _safe(sizes, sp_stats.kurtosis),
        "all_size_q25": q25,
        "all_size_q75": q75,
        "all_iat_mean": iat_mean,
        "all_iat_std": iat_std,
        "all_iat_min": float(iats.min()),
        "all_iat_max": float(iats.max()),
        "all_iat_skew": iat_skew,
        "all_iat_kurtosis": _safe(iats, sp_stats.kurtosis),
        "timing_regularity": float(iat_std / iat_mean) if iat_mean > 1e-8 else 0.0,
        "small_pkt_ratio": float((sizes < 100).sum() / max(n, 1)),
        "large_pkt_ratio": float((sizes > 1000).sum() / max(n, 1)),
        "size_entropy": _entropy(sizes),
    }

    # Markov transition features
    f.update(_markov_stats(flow.packet_directions))

    # Tunnel features
    f.update({
        "iat_cv": min(iat_std / iat_mean, 10.0) if iat_mean > 1e-8 else 0.0,
        "size_iat_ratio": min(size_mean / iat_mean, 1e6) if iat_mean > 1e-8 else 0.0,
        "burst_score": float(np.clip((q75 - q25) / max(size_median, 1), -5, 5)),
        "direction_proxy": float(np.clip(size_skew * iat_skew, -10, 10)),
    })

    # Bidirectional features
    total_bytes = size_mean * n
    bwd_bytes = max(total_bytes - total_fwd, 0)
    up_down_ratio = min(total_fwd / (bwd_bytes + 1), 100.0)
    
    p1 = max(f["markov_fwd_fwd"], 1e-5)
    p2 = max(f["markov_fwd_bwd"], 1e-5)
    p3 = max(f["markov_bwd_fwd"], 1e-5)
    p4 = max(f["markov_bwd_bwd"], 1e-5)
    directional_entropy = -(p1*np.log(p1) + p2*np.log(p2) + p3*np.log(p3) + p4*np.log(p4))
    
    req_resp_cadence = f["markov_fwd_bwd"] + f["markov_bwd_fwd"]
    alt_burst_score = min(req_resp_cadence * float(sizes.std() if n > 1 else 0), 5000.0)
    
    dur = max(flow.duration, 0.1)
    fwd_bytes_per_sec = min(total_fwd / dur, 1e8)
    bwd_bytes_per_sec = min(bwd_bytes / dur, 1e8)
    upload_ratio = min(total_fwd / max(total_bytes, 1.0), 1.0)
    
    f.update({
        "up_down_ratio": float(up_down_ratio),
        "directional_entropy": float(directional_entropy),
        "req_resp_cadence": float(req_resp_cadence),
        "alt_burst_score": float(alt_burst_score),
        "fwd_bytes_per_sec": float(fwd_bytes_per_sec),
        "bwd_bytes_per_sec": float(bwd_bytes_per_sec),
        "upload_ratio": float(upload_ratio),
    })

    # ── v6: Behavioral features ────────────────────────────
    f.update(_behavioral_features(sizes, iats, flow.packet_directions, times, fwd, dur, n))

    for k in f:
        v = f[k]
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            f[k] = 0.0
    return f


def _behavioral_features(
    sizes: np.ndarray,
    iats: np.ndarray,
    directions: list[int],
    times: np.ndarray,
    fwd_sizes: np.ndarray,
    dur: float,
    n: int,
) -> dict[str, float]:
    """v6 advanced behavioral features for real-world robustness."""
    dirs = np.array(directions)

    # 1. Session asymmetry: upload/download ratio over windowed segments
    n_segments = max(1, n // 10)
    segment_ratios = []
    for seg in range(n_segments):
        si = seg * (n // n_segments)
        ei = min((seg + 1) * (n // n_segments), n)
        seg_s = sizes[si:ei]
        seg_d = dirs[si:ei]
        if len(seg_s) < 2:
            continue
        fwd_sum = float(seg_s[seg_d > 0].sum()) if (seg_d > 0).any() else 0.0
        bwd_sum = float(seg_s[seg_d < 0].sum()) if (seg_d < 0).any() else 0.0
        segment_ratios.append(min(fwd_sum / (bwd_sum + 1.0), 20.0))

    session_asymmetry = float(np.mean(segment_ratios)) if segment_ratios else 1.0
    upload_consistency = float(np.std(segment_ratios)) if len(segment_ratios) > 1 else 0.0

    # 3. Beacon periodicity: autocorrelation-based IAT regularity
    beacon_periodicity = 0.0
    if len(iats) >= 6:
        iat_mean = iats.mean()
        if iat_mean > 1e-8:
            beacon_periodicity = float(np.clip(1.0 - iats.std() / iat_mean, 0, 1))

    # 4. Temporal volatility: variance of throughput across time windows
    temporal_volatility = 0.0
    if len(times) > 4 and dur > 0.01:
        n_win = min(5, len(times) // 3)
        if n_win >= 2:
            win_tp = []
            for w in range(n_win):
                ws = w * (n // n_win)
                we = min((w + 1) * (n // n_win), n)
                wt = times[ws:we]
                wd = max(wt[-1] - wt[0], 1e-6) if len(wt) > 1 else 1e-6
                win_tp.append(float(sizes[ws:we].sum()) / wd)
            m_tp = np.mean(win_tp)
            if m_tp > 1e-8:
                temporal_volatility = float(np.clip(np.std(win_tp) / m_tp, 0, 10))

    # 5. Burst acceleration: are bursts getting bigger over time?
    burst_acceleration = 0.0
    if n >= 10:
        half = n // 2
        d = max(float(sizes[:half].mean()), 1.0)
        burst_acceleration = float(np.clip(
            (float(sizes[half:].mean()) - float(sizes[:half].mean())) / d, -2, 2))

    # 6. Interaction density: average packets between direction changes
    dir_changes = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])
    interaction_density = min(float(n / max(dir_changes, 1)), 20.0)

    # 7. Directional persistence: longest same-direction run / total
    max_run, cur_run = 1, 1
    for i in range(1, len(dirs)):
        if dirs[i] == dirs[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    directional_persistence = float(max_run / max(n, 1))

    # 8. Silence ratio: fraction of flow with large gaps
    silence_ratio = 0.0
    if len(iats) > 1 and dur > 0.01:
        iat_m = iats.mean()
        if iat_m > 1e-8:
            silence_ratio = float(np.clip(iats[iats > 2 * iat_m].sum() / dur, 0, 1))

    # 9. Size modality: number of distinct size clusters
    size_modality = 1.0
    if n >= 10:
        hist, _ = np.histogram(sizes, bins=min(20, n // 3))
        size_modality = float(np.clip((hist > 0).sum() / 5.0, 0, 4))

    # 10. Response latency CV: variability of request→response gaps
    response_latency_cv = 0.0
    if len(dirs) >= 4:
        gaps = [times[i] - times[i - 1]
                for i in range(1, min(len(dirs), len(times)))
                if dirs[i - 1] > 0 and dirs[i] < 0 and times[i] - times[i - 1] > 0]
        if len(gaps) >= 3:
            mg = np.mean(gaps)
            if mg > 1e-8:
                response_latency_cv = float(np.clip(np.std(gaps) / mg, 0, 10))

    # 11. IAT trend: linear slope of IATs
    iat_trend = 0.0
    if len(iats) >= 5:
        try:
            slope = np.polyfit(np.arange(len(iats), dtype=np.float64), iats, 1)[0]
            iat_trend = float(np.clip(slope / max(iats.mean(), 1e-8), -2, 2))
        except (np.linalg.LinAlgError, ValueError):
            pass

    # 12. Size trend: linear slope of packet sizes
    size_trend = 0.0
    if n >= 5:
        try:
            slope = np.polyfit(np.arange(n, dtype=np.float64), sizes, 1)[0]
            size_trend = float(np.clip(slope / max(sizes.mean(), 1.0), -2, 2))
        except (np.linalg.LinAlgError, ValueError):
            pass

    return {
        "session_asymmetry": session_asymmetry,
        "upload_consistency": upload_consistency,
        "beacon_periodicity": beacon_periodicity,
        "temporal_volatility": temporal_volatility,
        "burst_acceleration": burst_acceleration,
        "interaction_density": interaction_density,
        "directional_persistence": directional_persistence,
        "silence_ratio": silence_ratio,
        "size_modality": size_modality,
        "response_latency_cv": response_latency_cv,
        "iat_trend": iat_trend,
        "size_trend": size_trend,
    }


def features_to_array(feat: dict, feature_names: list[str] = None) -> np.ndarray:
    names = feature_names or ALL_FEATURES
    return np.array([feat.get(name, 0.0) for name in names], dtype=np.float32)



class FlowTracker:
    """Stateful flow tracker: ingests packets → emits completed flows with features."""

    def __init__(self, timeout: float = 30.0, max_packets: int = 2000, min_packets: int = 10):
        self.timeout = timeout
        self.max_packets = max_packets
        self.min_packets = min_packets
        self.active: dict[str, FlowRecord] = {}
        self._counter = 0
        self._completed = 0

    @staticmethod
    def _key(pkt: RawPacket) -> str:
        eps = sorted([(pkt.src_ip, pkt.src_port), (pkt.dst_ip, pkt.dst_port)])
        return f"{eps[0][0]}:{eps[0][1]}-{eps[1][0]}:{eps[1][1]}"

    def ingest(self, pkt: RawPacket) -> Optional[tuple[FlowRecord, dict]]:
        """Ingest one packet. Returns (flow, features) if flow completed, else None."""
        key = self._key(pkt)

        if key not in self.active:
            self._counter += 1
            self.active[key] = FlowRecord(
                flow_id=f"flow_{self._counter:06d}",
                src_ip=pkt.src_ip, dst_ip=pkt.dst_ip,
                src_port=pkt.src_port, dst_port=pkt.dst_port,
                protocol=pkt.protocol,
                first_seen=pkt.timestamp, last_seen=pkt.timestamp,
            )

        flow = self.active[key]

        # Fix loopback duplication: if same size, direction, and ts diff < 0.001, it's a loopback echo
        is_fwd = (pkt.src_ip == flow.src_ip and pkt.src_port == flow.src_port)
        direction = 1 if is_fwd else -1

        if flow.n_packets > 0:
            last_ts = flow.packet_timestamps[-1]
            last_size = flow.packet_sizes[-1]
            last_dir = flow.packet_directions[-1]
            if pkt.length == last_size and direction == last_dir and (pkt.timestamp - last_ts) < 0.001:
                return None  # Ignore loopback duplicate

        flow.packet_sizes.append(pkt.length)
        flow.packet_timestamps.append(pkt.timestamp)
        flow.last_seen = pkt.timestamp
        flow.packet_directions.append(direction)

        if is_fwd:
            flow.fwd_sizes.append(pkt.length)

        if flow.n_packets >= self.max_packets:
            return self._finalize(key)
        return None

    def check_timeouts(self, now: float) -> list[tuple[FlowRecord, dict]]:
        expired = [k for k, f in self.active.items() if now - f.last_seen > self.timeout]
        return [r for k in expired if (r := self._finalize(k))]

    def flush_all(self) -> list[tuple[FlowRecord, dict]]:
        return [r for k in list(self.active) if (r := self._finalize(k))]

    def _finalize(self, key: str) -> Optional[tuple[FlowRecord, dict]]:
        flow = self.active.pop(key)
        if flow.n_packets < self.min_packets:
            return None
        self._completed += 1
        return flow, extract_features(flow)

    @property
    def stats(self) -> dict:
        return {"active": len(self.active), "completed": self._completed, "total_seen": self._counter}
