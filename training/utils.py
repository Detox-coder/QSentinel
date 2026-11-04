"""
Core training utilities: feature simulation, data augmentation, and wire-realistic generators.

These functions were originally in retrain_v5.py. They are used by:
  - retrain_v6.py (main training script)
  - training/benign_generators.py (12 diverse benign generators)
  - training/expanded_generators.py (15 benign + 10 attack generators)
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── Feature name constants ────────────────────────────────────
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
BEHAVIORAL_FEATURES = [
    "session_asymmetry", "upload_consistency", "beacon_periodicity",
    "temporal_volatility", "burst_acceleration", "interaction_density",
    "directional_persistence", "silence_ratio", "size_modality",
    "response_latency_cv", "iat_trend", "size_trend",
]
ALL_FEATURES = SHARED_FEATURES + MARKOV_FEATURES + TUNNEL_FEATURES + BIDIRECTIONAL_FEATURES + EXFIL_FEATURES + BEHAVIORAL_FEATURES
N_FEATURES = len(ALL_FEATURES)


# ── Helper functions ──────────────────────────────────────────

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


def _markov_stats(directions):
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


# ── Core function: simulate features from packet lists ─────────

def simulate_flow_features(sizes, iats, dirs):
    """
    Compute the full feature dictionary from raw packet-level data.

    Args:
        sizes: list of packet sizes (bytes)
        iats: list of inter-arrival times (seconds), len = len(sizes) - 1
        dirs: list of direction indicators (+1 fwd, -1 bwd)

    Returns:
        dict with all SHARED + MARKOV + TUNNEL + BIDIRECTIONAL + EXFIL + BEHAVIORAL features
    """
    sizes = np.array(sizes, dtype=np.float64)
    iats = np.array(iats, dtype=np.float64) if len(iats) > 0 else np.array([0.0])
    dirs_arr = np.array(dirs, dtype=np.float64)
    n = len(sizes)

    fwd_mask = dirs_arr > 0
    fwd_sizes = sizes[fwd_mask] if fwd_mask.any() else np.array([0.0])
    total_fwd = float(fwd_sizes.sum())

    dur = float(iats.sum()) if len(iats) > 0 else 1e-3
    dur = max(dur, 1e-6)

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
    f.update(_markov_stats(list(dirs)))

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

    # Behavioral features
    f.update(_behavioral_features(sizes, iats, dirs_arr, dur, n))

    # NaN/Inf safety
    for k in f:
        v = f[k]
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            f[k] = 0.0
    return f


def _behavioral_features(sizes, iats, dirs, dur, n):
    """Compute v6 behavioral features from raw packet data."""
    # 1. Session asymmetry
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

    session_asymmetry = float(np.mean(segment_ratios)) if segment_ratios else 0.0
    upload_consistency = float(np.std(segment_ratios)) if len(segment_ratios) > 1 else 0.0

    # 2. Beacon periodicity (autocorrelation of IATs)
    if len(iats) > 10:
        centered = iats - iats.mean()
        norm = float(np.sum(centered**2))
        if norm > 1e-10:
            autocorr = float(np.sum(centered[:-1] * centered[1:])) / norm
        else:
            autocorr = 0.0
    else:
        autocorr = 0.0
    beacon_periodicity = float(np.clip(autocorr, -1, 1))

    # 3. Temporal volatility (variance of windowed throughput)
    n_windows = max(1, n // 20)
    throughputs = []
    for w in range(n_windows):
        si = w * (n // n_windows)
        ei = min((w + 1) * (n // n_windows), n)
        w_sizes = sizes[si:ei]
        w_iats_end = min(ei - 1, len(iats))
        w_iats_start = max(si - 1, 0) if si > 0 else 0
        w_dur = float(iats[w_iats_start:w_iats_end].sum()) if w_iats_end > w_iats_start else 1e-3
        w_dur = max(w_dur, 1e-6)
        throughputs.append(float(w_sizes.sum()) / w_dur)
    temporal_volatility = float(np.std(throughputs)) if len(throughputs) > 1 else 0.0

    # 4. Burst acceleration (trend of burst sizes)
    burst_sizes = []
    current_burst = 0.0
    for i in range(n):
        current_burst += sizes[i]
        if i < len(iats) and iats[i] > 0.1:
            burst_sizes.append(current_burst)
            current_burst = 0.0
    if current_burst > 0:
        burst_sizes.append(current_burst)
    if len(burst_sizes) > 2:
        x = np.arange(len(burst_sizes))
        slope, _, _, _, _ = sp_stats.linregress(x, burst_sizes)
        burst_acceleration = float(np.clip(slope, -1e6, 1e6))
    else:
        burst_acceleration = 0.0

    # 5. Interaction density
    direction_changes = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i-1])
    interaction_density = float(n / max(direction_changes, 1))

    # 6. Directional persistence
    if n > 0:
        max_run = 1
        current_run = 1
        for i in range(1, len(dirs)):
            if dirs[i] == dirs[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        directional_persistence = float(max_run / n)
    else:
        directional_persistence = 0.0

    # 7. Silence ratio
    silence_threshold = 0.5
    silence_time = float(iats[iats > silence_threshold].sum()) if len(iats) > 0 else 0.0
    silence_ratio = min(silence_time / max(dur, 1e-6), 1.0)

    # 8. Size modality
    if n > 5:
        hist, _ = np.histogram(sizes, bins=min(20, n // 2))
        peaks = sum(1 for i in range(1, len(hist)-1) if hist[i] > hist[i-1] and hist[i] > hist[i+1])
        size_modality = float(max(peaks, 1))
    else:
        size_modality = 1.0

    # 9. Response latency CV
    resp_latencies = []
    for i in range(1, min(len(dirs), len(iats)+1)):
        if dirs[i] != dirs[i-1]:
            resp_latencies.append(iats[i-1])
    if len(resp_latencies) > 1:
        rl_mean = np.mean(resp_latencies)
        rl_std = np.std(resp_latencies)
        response_latency_cv = float(rl_std / rl_mean) if rl_mean > 1e-8 else 0.0
    else:
        response_latency_cv = 0.0

    # 10. IAT trend
    if len(iats) > 2:
        x = np.arange(len(iats))
        slope, _, _, _, _ = sp_stats.linregress(x, iats)
        iat_trend = float(np.clip(slope, -1e3, 1e3))
    else:
        iat_trend = 0.0

    # 11. Size trend
    if n > 2:
        x = np.arange(n)
        slope, _, _, _, _ = sp_stats.linregress(x, sizes)
        size_trend = float(np.clip(slope, -1e3, 1e3))
    else:
        size_trend = 0.0

    return {
        "session_asymmetry": session_asymmetry,
        "upload_consistency": upload_consistency,
        "beacon_periodicity": beacon_periodicity,
        "temporal_volatility": min(temporal_volatility, 1e8),
        "burst_acceleration": burst_acceleration,
        "interaction_density": min(interaction_density, 1000.0),
        "directional_persistence": directional_persistence,
        "silence_ratio": silence_ratio,
        "size_modality": size_modality,
        "response_latency_cv": min(response_latency_cv, 50.0),
        "iat_trend": iat_trend,
        "size_trend": size_trend,
    }


# ── Advanced feature augmentation for DataFrames ──────────────

def add_advanced_features(df):
    """Add TUNNEL + BIDIRECTIONAL + EXFIL + BEHAVIORAL features to a DataFrame that only has SHARED + MARKOV."""
    for feat in TUNNEL_FEATURES + BIDIRECTIONAL_FEATURES + EXFIL_FEATURES + BEHAVIORAL_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    # Derive what we can from existing columns
    iat_mean = df["all_iat_mean"].replace(0, np.nan)
    df["iat_cv"] = np.clip(df["all_iat_std"] / iat_mean.fillna(1e-8), 0, 10).fillna(0)
    df["size_iat_ratio"] = np.clip(df["all_size_mean"] / iat_mean.fillna(1e-8), 0, 1e6).fillna(0)
    median = df["all_size_median"].replace(0, 1)
    df["burst_score"] = np.clip((df["all_size_q75"] - df["all_size_q25"]) / median, -5, 5).fillna(0)
    df["direction_proxy"] = np.clip(df["all_size_skew"] * df["all_iat_skew"], -10, 10).fillna(0)

    # Bidirectional features from Markov stats
    df["req_resp_cadence"] = df["markov_fwd_bwd"] + df["markov_bwd_fwd"]
    df["alt_burst_score"] = np.clip(df["req_resp_cadence"] * df["all_size_std"], 0, 5000).fillna(0)

    dur = df["duration"].clip(lower=1e-6)
    fwd = df["total_bytes_fwd"]
    total = df["all_size_mean"] * df["total_packets"]
    bwd = (total - fwd).clip(lower=0)
    df["up_down_ratio"] = np.clip(fwd / (bwd + 1), 0, 100).fillna(0)
    df["fwd_bytes_per_sec"] = np.clip(fwd / dur, 0, 1e8).fillna(0)
    df["bwd_bytes_per_sec"] = np.clip(bwd / dur, 0, 1e8).fillna(0)
    df["upload_ratio"] = np.clip(fwd / total.clip(lower=1), 0, 1).fillna(0)

    p1 = df["markov_fwd_fwd"].clip(lower=1e-5)
    p2 = df["markov_fwd_bwd"].clip(lower=1e-5)
    p3 = df["markov_bwd_fwd"].clip(lower=1e-5)
    p4 = df["markov_bwd_bwd"].clip(lower=1e-5)
    df["directional_entropy"] = -(p1*np.log(p1) + p2*np.log(p2) + p3*np.log(p3) + p4*np.log(p4))

    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df


# ── Data augmentation ─────────────────────────────────────────

def augment(X, y, target_per_class, rng, class_names):
    """Augment dataset to have at least `target_per_class` samples per class."""
    classes = np.unique(y)
    aug_X, aug_y = list(X), list(y)
    for c in classes:
        mask = y == c
        n_existing = mask.sum()
        if n_existing >= target_per_class:
            continue
        n_needed = target_per_class - n_existing
        class_samples = X[mask]
        for _ in range(n_needed):
            idx = rng.integers(0, n_existing)
            sample = class_samples[idx].copy()
            # Gaussian noise augmentation
            noise = rng.normal(0, 0.02, size=sample.shape)
            aug_X.append(sample + noise)
            aug_y.append(c)
    return np.array(aug_X, dtype=np.float32), np.array(aug_y)


# ── Wire-realistic malicious generators ───────────────────────

def gen_c2_beacon(n_samples, rng=None):
    """C2 beacon: periodic small packets with uniform IATs."""
    r = rng or np.random.default_rng(42)
    rows = []
    for _ in range(n_samples):
        n_pkts = r.integers(20, 100)
        base_iat = r.uniform(1.0, 10.0)
        sizes, iats, dirs = [], [], []
        for i in range(n_pkts):
            sizes.append(r.integers(40, 200))
            dirs.append(1 if i % 2 == 0 else -1)
            if i > 0:
                iats.append(base_iat * r.uniform(0.95, 1.05))
        rows.append({**simulate_flow_features(sizes, iats, dirs), "label": "malicious_c2"})
    return pd.DataFrame(rows)


def gen_exfil_burst(n_samples, rng=None):
    """Exfiltration: large outbound bursts."""
    r = rng or np.random.default_rng(42)
    rows = []
    for _ in range(n_samples):
        n_pkts = r.integers(50, 300)
        sizes, iats, dirs = [], [], []
        for i in range(n_pkts):
            if r.random() < 0.85:
                sizes.append(r.integers(800, 1450))
                dirs.append(1)
            else:
                sizes.append(r.integers(40, 100))
                dirs.append(-1)
            if i > 0:
                iats.append(r.uniform(0.001, 0.05))
        rows.append({**simulate_flow_features(sizes, iats, dirs), "label": "malicious_exfiltration"})
    return pd.DataFrame(rows)


def gen_slow_exfil(n_samples, rng=None):
    """Slow exfiltration: trickle out data over long durations."""
    r = rng or np.random.default_rng(42)
    rows = []
    for _ in range(n_samples):
        n_pkts = r.integers(10, 60)
        sizes, iats, dirs = [], [], []
        for i in range(n_pkts):
            sizes.append(r.integers(200, 600))
            dirs.append(1)
            if i > 0:
                iats.append(r.uniform(2.0, 30.0))
        rows.append({**simulate_flow_features(sizes, iats, dirs), "label": "malicious_slow_exfil"})
    return pd.DataFrame(rows)


def gen_tunnel(n_samples, rng=None):
    """Tunnel: encapsulated traffic with uniform sizes."""
    r = rng or np.random.default_rng(42)
    rows = []
    for _ in range(n_samples):
        n_pkts = r.integers(40, 200)
        base_size = r.integers(500, 1200)
        sizes, iats, dirs = [], [], []
        for i in range(n_pkts):
            sizes.append(base_size + r.integers(-20, 20))
            dirs.append(1 if r.random() < 0.5 else -1)
            if i > 0:
                iats.append(r.uniform(0.01, 0.1))
        rows.append({**simulate_flow_features(sizes, iats, dirs), "label": "malicious_tunnel"})
    return pd.DataFrame(rows)


def gen_benign_upload(n_samples, rng=None):
    """Benign upload: large outbound, mixed sizes."""
    r = rng or np.random.default_rng(42)
    rows = []
    for _ in range(n_samples):
        n_pkts = r.integers(50, 300)
        sizes, iats, dirs = [], [], []
        for i in range(n_pkts):
            if r.random() < 0.7:
                sizes.append(r.integers(400, 1450))
                dirs.append(1)
            else:
                sizes.append(r.integers(40, 200))
                dirs.append(-1)
            if i > 0:
                iats.append(r.uniform(0.001, 0.1))
        rows.append({**simulate_flow_features(sizes, iats, dirs), "label": "benign_large_upload"})
    return pd.DataFrame(rows)


def gen_benign_cloud_sync(n_samples, rng=None):
    """Benign cloud sync: periodic bidirectional bursts."""
    r = rng or np.random.default_rng(42)
    rows = []
    for _ in range(n_samples):
        n_pkts = r.integers(20, 150)
        sizes, iats, dirs = [], [], []
        for i in range(n_pkts):
            sizes.append(r.integers(100, 800))
            dirs.append(1 if r.random() < 0.5 else -1)
            if i > 0:
                if r.random() < 0.15:
                    iats.append(r.uniform(1.0, 10.0))
                else:
                    iats.append(r.uniform(0.01, 0.2))
        rows.append({**simulate_flow_features(sizes, iats, dirs), "label": "benign_cloud_sync"})
    return pd.DataFrame(rows)


def gen_benign_video(n_samples, rng=None):
    """Benign video streaming: large inbound, small outbound ACKs."""
    r = rng or np.random.default_rng(42)
    rows = []
    for _ in range(n_samples):
        n_pkts = r.integers(100, 500)
        sizes, iats, dirs = [], [], []
        for i in range(n_pkts):
            if r.random() < 0.9:
                sizes.append(r.integers(1000, 1450))
                dirs.append(-1)
            else:
                sizes.append(r.integers(40, 80))
                dirs.append(1)
            if i > 0:
                iats.append(r.uniform(0.001, 0.03))
        rows.append({**simulate_flow_features(sizes, iats, dirs), "label": "benign_streaming"})
    return pd.DataFrame(rows)
