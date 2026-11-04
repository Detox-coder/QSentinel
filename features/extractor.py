"""
Feature extraction engine.

Converts raw PacketRecord lists into 80+ numerical features per flow,
suitable for ML training.  Outputs to CSV and Parquet.
"""

import os
import math
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from packet_logger import PacketRecord
import config


class FeatureExtractor:
    """Extracts features from logged packet data and exports datasets."""

    BURST_GAP_THRESHOLD = 0.1  # seconds — gap that delineates bursts

    # ── Public API ───────────────────────────────────────

    def extract_flow_features(
        self, packets: List[PacketRecord]
    ) -> Dict[str, float]:
        """Return a flat dict of features for a single flow."""
        if len(packets) < 2:
            return self._empty_features(packets)

        packets.sort(key=lambda p: p.timestamp)
        sizes = np.array([p.size for p in packets], dtype=np.float64)
        dirs = np.array([p.direction for p in packets], dtype=np.float64)
        times = np.array([p.timestamp for p in packets], dtype=np.float64)
        iats = np.diff(times)

        fwd_mask = dirs > 0
        bwd_mask = dirs < 0
        fwd_sizes = sizes[fwd_mask]
        bwd_sizes = sizes[bwd_mask]
        fwd_iats = np.diff(times[fwd_mask]) if fwd_mask.sum() > 1 else np.array([0.0])
        bwd_iats = np.diff(times[bwd_mask]) if bwd_mask.sum() > 1 else np.array([0.0])

        features = {}
        features.update(self._flow_stats(packets, sizes, dirs, times, iats))
        features.update(self._size_stats(sizes, fwd_sizes, bwd_sizes))
        features.update(self._timing_stats(iats, fwd_iats, bwd_iats))
        features.update(self._burst_stats(packets))
        features.update(self._behavior_stats(sizes, dirs, iats))
        features.update(self._markov_stats(dirs))
        features.update(self._histogram_features(sizes))

        # Metadata (not features but needed for labeling)
        features["label"] = packets[0].label
        features["traffic_type"] = packets[0].traffic_type
        features["flow_id"] = packets[0].flow_id

        return features

    def build_dataset(
        self, all_flows: Dict[str, List[PacketRecord]]
    ) -> pd.DataFrame:
        """Build a DataFrame of features from all logged flows."""
        rows = []
        for flow_id, packets in all_flows.items():
            if len(packets) < 2:
                continue
            feat = self.extract_flow_features(packets)
            rows.append(feat)

        df = pd.DataFrame(rows)
        return df

    def export(
        self,
        df: pd.DataFrame,
        output_dir: str = config.OUTPUT_DIR,
        prefix: str = "quic_dataset",
    ) -> dict:
        """Save dataset to CSV and Parquet. Returns file paths."""
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{prefix}.csv")
        parquet_path = os.path.join(output_dir, f"{prefix}.parquet")

        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False, engine="pyarrow")

        print(f"[EXPORT] CSV     → {csv_path}  ({len(df)} rows)")
        print(f"[EXPORT] Parquet → {parquet_path}")
        return {"csv": csv_path, "parquet": parquet_path}

    # ── Feature groups ───────────────────────────────────

    def _flow_stats(self, packets, sizes, dirs, times, iats):
        duration = times[-1] - times[0]
        n_fwd = int((dirs > 0).sum())
        n_bwd = int((dirs < 0).sum())
        bytes_fwd = float(sizes[dirs > 0].sum()) if n_fwd > 0 else 0.0
        bytes_bwd = float(sizes[dirs < 0].sum()) if n_bwd > 0 else 0.0
        total_bytes = bytes_fwd + bytes_bwd

        # Idle / active periods
        idle_mask = iats > self.BURST_GAP_THRESHOLD
        active_mask = ~idle_mask
        idle_times = iats[idle_mask] if idle_mask.any() else np.array([0.0])
        active_times = iats[active_mask] if active_mask.any() else np.array([0.0])

        return {
            "duration": duration,
            "total_packets": len(packets),
            "total_pkts_fwd": n_fwd,
            "total_pkts_bwd": n_bwd,
            "total_bytes_fwd": bytes_fwd,
            "total_bytes_bwd": bytes_bwd,
            "bytes_ratio": bytes_fwd / max(total_bytes, 1),
            "pkts_per_sec": len(packets) / max(duration, 1e-6),
            "bytes_per_sec": total_bytes / max(duration, 1e-6),
            "fwd_pkts_per_sec": n_fwd / max(duration, 1e-6),
            "bwd_pkts_per_sec": n_bwd / max(duration, 1e-6),
            "pkt_ratio": n_fwd / max(n_fwd + n_bwd, 1),
            "idle_time_mean": float(np.mean(idle_times)),
            "idle_time_std": float(np.std(idle_times)),
            "active_time_mean": float(np.mean(active_times)),
        }

    def _size_stats(self, sizes, fwd_sizes, bwd_sizes):
        f = {}
        for prefix, arr in [("all", sizes), ("fwd", fwd_sizes), ("bwd", bwd_sizes)]:
            if len(arr) == 0:
                arr = np.array([0.0])
            f[f"{prefix}_size_mean"] = float(np.mean(arr))
            f[f"{prefix}_size_std"] = float(np.std(arr))
            f[f"{prefix}_size_min"] = float(np.min(arr))
            f[f"{prefix}_size_max"] = float(np.max(arr))
            f[f"{prefix}_size_median"] = float(np.median(arr))
            f[f"{prefix}_size_skew"] = float(sp_stats.skew(arr)) if len(arr) > 2 else 0.0
            f[f"{prefix}_size_kurtosis"] = float(sp_stats.kurtosis(arr)) if len(arr) > 2 else 0.0
            f[f"{prefix}_size_q25"] = float(np.percentile(arr, 25))
            f[f"{prefix}_size_q75"] = float(np.percentile(arr, 75))
        return f

    def _timing_stats(self, iats, fwd_iats, bwd_iats):
        f = {}
        for prefix, arr in [("all", iats), ("fwd", fwd_iats), ("bwd", bwd_iats)]:
            if len(arr) == 0:
                arr = np.array([0.0])
            f[f"{prefix}_iat_mean"] = float(np.mean(arr))
            f[f"{prefix}_iat_std"] = float(np.std(arr))
            f[f"{prefix}_iat_min"] = float(np.min(arr))
            f[f"{prefix}_iat_max"] = float(np.max(arr))
            f[f"{prefix}_iat_skew"] = float(sp_stats.skew(arr)) if len(arr) > 2 else 0.0
            f[f"{prefix}_iat_kurtosis"] = float(sp_stats.kurtosis(arr)) if len(arr) > 2 else 0.0
        f["timing_regularity"] = float(np.std(iats) / max(np.mean(iats), 1e-9))
        return f

    def _burst_stats(self, packets):
        """Identify bursts (groups of packets with small inter-arrival gaps)."""
        if len(packets) < 2:
            return self._default_burst()

        times = [p.timestamp for p in packets]
        sizes = [p.size for p in packets]
        dirs = [p.direction for p in packets]

        bursts = []
        current_burst = {"start": 0, "sizes": [sizes[0]], "dirs": [dirs[0]]}

        for i in range(1, len(packets)):
            gap = times[i] - times[i - 1]
            if gap < self.BURST_GAP_THRESHOLD:
                current_burst["sizes"].append(sizes[i])
                current_burst["dirs"].append(dirs[i])
            else:
                bursts.append(current_burst)
                current_burst = {"start": i, "sizes": [sizes[i]], "dirs": [dirs[i]]}
        bursts.append(current_burst)

        if len(bursts) == 0:
            return self._default_burst()

        burst_sizes = [sum(b["sizes"]) for b in bursts]
        burst_pkt_counts = [len(b["sizes"]) for b in bursts]
        burst_fwd_ratios = [
            sum(1 for d in b["dirs"] if d > 0) / max(len(b["dirs"]), 1)
            for b in bursts
        ]

        inter_burst = []
        for i in range(1, len(bursts)):
            prev_end = bursts[i - 1]["start"] + len(bursts[i - 1]["sizes"]) - 1
            inter_burst.append(times[bursts[i]["start"]] - times[prev_end])
        if not inter_burst:
            inter_burst = [0.0]

        return {
            "burst_count": len(bursts),
            "burst_size_mean": float(np.mean(burst_sizes)),
            "burst_size_std": float(np.std(burst_sizes)),
            "burst_size_max": float(np.max(burst_sizes)),
            "burst_pkt_mean": float(np.mean(burst_pkt_counts)),
            "burst_pkt_std": float(np.std(burst_pkt_counts)),
            "burst_asymmetry": float(np.mean(burst_fwd_ratios)),
            "inter_burst_mean": float(np.mean(inter_burst)),
            "inter_burst_std": float(np.std(inter_burst)),
            "burst_regularity": float(
                np.std(inter_burst) / max(np.mean(inter_burst), 1e-9)
            ),
        }

    def _behavior_stats(self, sizes, dirs, iats):
        # Direction changes
        dir_changes = int(np.sum(np.abs(np.diff(dirs)) > 0))
        n = len(dirs)

        # Longest streaks
        fwd_streak = bwd_streak = curr_streak = 1
        for i in range(1, n):
            if dirs[i] == dirs[i - 1]:
                curr_streak += 1
            else:
                if dirs[i - 1] > 0:
                    fwd_streak = max(fwd_streak, curr_streak)
                else:
                    bwd_streak = max(bwd_streak, curr_streak)
                curr_streak = 1
        if n > 0:
            if dirs[-1] > 0:
                fwd_streak = max(fwd_streak, curr_streak)
            else:
                bwd_streak = max(bwd_streak, curr_streak)

        # Entropy of packet sizes (binned)
        size_hist, _ = np.histogram(sizes, bins=20, density=True)
        size_hist = size_hist[size_hist > 0]
        size_entropy = float(-np.sum(size_hist * np.log2(size_hist + 1e-12)))

        # Entropy of inter-arrival times (binned)
        if len(iats) > 1:
            iat_hist, _ = np.histogram(iats, bins=20, density=True)
            iat_hist = iat_hist[iat_hist > 0]
            iat_entropy = float(-np.sum(iat_hist * np.log2(iat_hist + 1e-12)))
        else:
            iat_entropy = 0.0

        # Small / large packet ratios
        small = float(np.sum(sizes < 100)) / max(n, 1)
        large = float(np.sum(sizes > 1000)) / max(n, 1)

        return {
            "direction_changes": dir_changes,
            "direction_change_rate": dir_changes / max(n - 1, 1),
            "fwd_bwd_ratio": float(np.sum(dirs > 0)) / max(float(np.sum(dirs < 0)), 1),
            "longest_fwd_streak": fwd_streak,
            "longest_bwd_streak": bwd_streak,
            "size_entropy": size_entropy,
            "iat_entropy": iat_entropy,
            "small_pkt_ratio": small,
            "large_pkt_ratio": large,
        }

    def _histogram_features(self, sizes, n_bins=10):
        """Normalized histogram of packet sizes → 10 features."""
        hist, _ = np.histogram(
            sizes, bins=n_bins, range=(0, max(float(np.max(sizes)), 1500)), density=True
        )
        return {f"size_hist_{i}": float(v) for i, v in enumerate(hist)}

    def _markov_stats(self, dirs):
        # 1st order Markov transition probabilities for packet direction
        n = len(dirs)
        if n < 2:
            return {"markov_fwd_fwd": 0.0, "markov_fwd_bwd": 0.0, "markov_bwd_fwd": 0.0, "markov_bwd_bwd": 0.0}
        
        transitions = {(1, 1): 0, (1, -1): 0, (-1, 1): 0, (-1, -1): 0}
        for i in range(1, n):
            d1, d2 = 1 if dirs[i-1] > 0 else -1, 1 if dirs[i] > 0 else -1
            transitions[(d1, d2)] += 1
            
        fwd_total = max(transitions[(1, 1)] + transitions[(1, -1)], 1)
        bwd_total = max(transitions[(-1, 1)] + transitions[(-1, -1)], 1)
        
        return {
            "markov_fwd_fwd": transitions[(1, 1)] / fwd_total,
            "markov_fwd_bwd": transitions[(1, -1)] / fwd_total,
            "markov_bwd_fwd": transitions[(-1, 1)] / bwd_total,
            "markov_bwd_bwd": transitions[(-1, -1)] / bwd_total,
        }

    # ── Helpers ──────────────────────────────────────────

    def _default_burst(self):
        return {
            "burst_count": 0,
            "burst_size_mean": 0.0,
            "burst_size_std": 0.0,
            "burst_size_max": 0.0,
            "burst_pkt_mean": 0.0,
            "burst_pkt_std": 0.0,
            "burst_asymmetry": 0.0,
            "inter_burst_mean": 0.0,
            "inter_burst_std": 0.0,
            "burst_regularity": 0.0,
        }

    def _empty_features(self, packets):
        feat = {k: 0.0 for k in self._all_feature_names()}
        if packets:
            feat["label"] = packets[0].label
            feat["traffic_type"] = packets[0].traffic_type
            feat["flow_id"] = packets[0].flow_id
        return feat

    @staticmethod
    def _all_feature_names():
        """Return list of all numeric feature names (for empty-flow fallback)."""
        names = [
            "duration", "total_packets", "total_pkts_fwd", "total_pkts_bwd",
            "total_bytes_fwd", "total_bytes_bwd", "bytes_ratio",
            "pkts_per_sec", "bytes_per_sec", "fwd_pkts_per_sec", "bwd_pkts_per_sec",
            "pkt_ratio", "idle_time_mean", "idle_time_std", "active_time_mean",
        ]
        for prefix in ["all", "fwd", "bwd"]:
            for stat in ["mean", "std", "min", "max", "median", "skew", "kurtosis", "q25", "q75"]:
                names.append(f"{prefix}_size_{stat}")
        for prefix in ["all", "fwd", "bwd"]:
            for stat in ["mean", "std", "min", "max", "skew", "kurtosis"]:
                names.append(f"{prefix}_iat_{stat}")
        names.append("timing_regularity")
        names.extend([
            "burst_count", "burst_size_mean", "burst_size_std", "burst_size_max",
            "burst_pkt_mean", "burst_pkt_std", "burst_asymmetry",
            "inter_burst_mean", "inter_burst_std", "burst_regularity",
        ])
        names.extend([
            "direction_changes", "direction_change_rate", "fwd_bwd_ratio",
            "longest_fwd_streak", "longest_bwd_streak",
            "size_entropy", "iat_entropy", "small_pkt_ratio", "large_pkt_ratio",
            "markov_fwd_fwd", "markov_fwd_bwd", "markov_bwd_fwd", "markov_bwd_bwd",
        ])
        names.extend([f"size_hist_{i}" for i in range(10)])
        return names
