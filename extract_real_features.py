#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



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
ALL_FEATURES = SHARED_FEATURES + MARKOV_FEATURES + TUNNEL_FEATURES + BIDIRECTIONAL_FEATURES + EXFIL_FEATURES
N_FEATURES = len(ALL_FEATURES)


@dataclass
class RawPacket:
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    length: int
    protocol: str
    is_quic: bool
    sni: str = ""           
    tls_version: str = ""


TSHARK_FIELDS = [
    "frame.time_epoch",
    "ip.src",
    "ip.dst",
    "udp.srcport",
    "udp.dstport",
    "frame.len",
    "quic",
    "_ws.col.Protocol",
    "tls.handshake.extensions_server_name",
    "quic.version",
]


def _build_field_args() -> list[str]:
    args = []
    for f in TSHARK_FIELDS:
        args.extend(["-e", f])
    return args


def _parse_line(line: str) -> Optional[RawPacket]:
    parts = line.strip().split("|")
    if len(parts) < 6:
        return None
    try:
        ts = float(parts[0]) if parts[0] else 0.0
        src_ip = parts[1] or "0.0.0.0"
        dst_ip = parts[2] or "0.0.0.0"
        src_port = int(parts[3]) if parts[3] else 0
        dst_port = int(parts[4]) if parts[4] else 0
        length = int(parts[5]) if parts[5] else 0
        quic_present = bool(parts[6].strip()) if len(parts) > 6 else False
        protocol = parts[7].strip() if len(parts) > 7 else ""
        is_quic = quic_present or "QUIC" in protocol.upper()
        sni = parts[8].strip() if len(parts) > 8 else ""
        tls_version = parts[9].strip() if len(parts) > 9 else ""
        return RawPacket(ts, src_ip, dst_ip, src_port, dst_port, length,
                         protocol, is_quic, sni, tls_version)
    except (ValueError, IndexError):
        return None


def read_pcap_with_sni(pcap_path: str, quic_only: bool = True) -> Generator[RawPacket, None, None]:
    if not os.path.exists(pcap_path):
        print(f"[ERROR] File not found: {pcap_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        "tshark", "-r", pcap_path, "-T", "fields",
        "-E", "separator=|", "-E", "header=n", "-E", "quote=n",
    ] + _build_field_args()

    print(f"[READER] Parsing PCAP: {pcap_path} ({os.path.getsize(pcap_path)/1024/1024:.1f} MB)")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        count = 0
        for line in proc.stdout:
            pkt = _parse_line(line)
            if pkt and (not quic_only or pkt.is_quic):
                count += 1
                yield pkt
                if count % 10000 == 0:
                    print(f"  [READER] Processed {count} QUIC packets...", end="\r")
        proc.wait()
        print(f"\n[READER] Finished: {count} QUIC packets extracted")
    except FileNotFoundError:
        print("[ERROR] tshark not found. Install wireshark-cli.", file=sys.stderr)
        sys.exit(1)


@dataclass
class FlowRecord:
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
    packet_directions: list[int] = field(default_factory=list)
    sni_list: list[str] = field(default_factory=list)

    @property
    def n_packets(self) -> int:
        return len(self.packet_sizes)

    @property
    def duration(self) -> float:
        return max(self.last_seen - self.first_seen, 1e-6)

    @property
    def primary_sni(self) -> str:
        snis = [s for s in self.sni_list if s]
        if not snis:
            return ""
        from collections import Counter
        return Counter(snis).most_common(1)[0][0]


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
    n = len(directions)
    if n < 2:
        return {"markov_fwd_fwd": 0.0, "markov_fwd_bwd": 0.0,
                "markov_bwd_fwd": 0.0, "markov_bwd_bwd": 0.0}

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

    f.update(_markov_stats(flow.packet_directions))

    f.update({
        "iat_cv": min(iat_std / iat_mean, 10.0) if iat_mean > 1e-8 else 0.0,
        "size_iat_ratio": min(size_mean / iat_mean, 1e6) if iat_mean > 1e-8 else 0.0,
        "burst_score": float(np.clip((q75 - q25) / max(size_median, 1), -5, 5)),
        "direction_proxy": float(np.clip(size_skew * iat_skew, -10, 10)),
    })

    total_bytes = size_mean * n
    bwd_bytes = max(total_bytes - total_fwd, 0.0)
    p1 = max(f["markov_fwd_fwd"], 1e-5)
    p2 = max(f["markov_fwd_bwd"], 1e-5)
    p3 = max(f["markov_bwd_fwd"], 1e-5)
    p4 = max(f["markov_bwd_bwd"], 1e-5)
    directional_entropy = -(p1*np.log(p1) + p2*np.log(p2) + p3*np.log(p3) + p4*np.log(p4))
    req_resp_cadence = f["markov_fwd_bwd"] + f["markov_bwd_fwd"]

    f.update({
        "up_down_ratio": float(min(total_fwd / (bwd_bytes + 1), 100.0)),
        "directional_entropy": float(directional_entropy),
        "req_resp_cadence": float(req_resp_cadence),
        "alt_burst_score": float(min(req_resp_cadence * float(sizes.std() if n > 1 else 0), 5000.0)),
        "fwd_bytes_per_sec": float(min(total_fwd / max(flow.duration, 0.1), 1e8)),
        "bwd_bytes_per_sec": float(min(bwd_bytes / max(flow.duration, 0.1), 1e8)),
        "upload_ratio": float(min(total_fwd / max(total_bytes, 1.0), 1.0)),
    })

    for k in f:
        v = f[k]
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            f[k] = 0.0
    return f


STREAMING_DOMAINS = [
    "youtube", "googlevideo", "ytimg", "yt3.ggpht",
    "netflix", "nflxvideo", "twitch", "ttvnw",
    "spotify", "scdn", "soundcloud", "vimeo",
    "dailymotion", "bitmovin", "akamaized", "cloudfront",
    "fbcdn", "instagram",
]

VIDEO_CALL_DOMAINS = [
    "meet.google", "zoom", "teams.microsoft", "skype",
    "discord", "discordapp", "discordcdn",
    "webrtc", "stun", "turn",
]

CLOUD_SYNC_DOMAINS = [
    "drive.google", "dropbox", "onedrive", "icloud",
    "box.com", "sync", "mega", "gdrive",
    "github", "gitlab", "bitbucket",
]

BROWSING_DOMAINS = [
    "google.com", "googleapis", "gstatic",
    "facebook", "twitter", "reddit", "wikipedia",
    "amazon", "microsoft", "apple",
    "cloudflare", "fastly",
]


def classify_benign_subtype(flow: FlowRecord, features: dict) -> str:
    sni = flow.primary_sni.lower()
    dur = features["duration"]
    pkts = features["total_packets"]
    bps = features["bytes_per_sec"]
    pps = features["pkts_per_sec"]
    large_ratio = features["large_pkt_ratio"]
    small_ratio = features["small_pkt_ratio"]
    timing_reg = features["timing_regularity"]

    if sni:
        for domain in STREAMING_DOMAINS:
            if domain in sni:
                return "benign_streaming_real"
        for domain in VIDEO_CALL_DOMAINS:
            if domain in sni:
                return "benign_video_call"
        for domain in CLOUD_SYNC_DOMAINS:
            if domain in sni:
                return "benign_cloud_sync"
        for domain in BROWSING_DOMAINS:
            if domain in sni:
                return "benign_http3_browsing"

    if dur > 5.0 and bps > 50000 and large_ratio > 0.3:
        return "benign_streaming_real"

    if dur > 3.0 and pps > 20 and 0.3 < large_ratio < 0.7:
        fwd_bwd_balance = abs(features.get("markov_fwd_fwd", 0.5) - 0.5)
        if fwd_bwd_balance < 0.3:  
            return "benign_video_call"

    if dur > 1.0 and bps > 10000 and features.get("burst_score", 0) > 0.5:
        return "benign_cloud_sync"

    return "benign_http3_browsing"


def extract_from_pcap(
    pcap_path: str,
    timeout: float = 60.0,
    min_packets: int = 10,
    max_packets: int = 5000,
    quic_only: bool = True,
) -> pd.DataFrame:

    print(f"\n{'#'*60}")
    print(f"  REAL-WORLD FEATURE EXTRACTION")
    print(f"{'#'*60}")
    print(f"  PCAP: {pcap_path}")
    print(f"  Timeout: {timeout}s  Min packets: {min_packets}")
    print(f"  Expected features: {N_FEATURES}")
    print()

    active_flows: dict[str, FlowRecord] = {}
    completed_flows: list[tuple[FlowRecord, dict]] = []
    flow_counter = 0
    pkt_count = 0
    sni_map: dict[str, str] = {} 

    t0 = time.time()

    for pkt in read_pcap_with_sni(pcap_path, quic_only=quic_only):
        pkt_count += 1

        eps = sorted([(pkt.src_ip, pkt.src_port), (pkt.dst_ip, pkt.dst_port)])
        key = f"{eps[0][0]}:{eps[0][1]}-{eps[1][0]}:{eps[1][1]}"

        if key not in active_flows:
            flow_counter += 1
            active_flows[key] = FlowRecord(
                flow_id=f"real_{flow_counter:06d}",
                src_ip=pkt.src_ip, dst_ip=pkt.dst_ip,
                src_port=pkt.src_port, dst_port=pkt.dst_port,
                protocol=pkt.protocol,
                first_seen=pkt.timestamp, last_seen=pkt.timestamp,
            )

        flow = active_flows[key]
        flow.packet_sizes.append(pkt.length)
        flow.packet_timestamps.append(pkt.timestamp)
        flow.last_seen = pkt.timestamp

        is_fwd = (pkt.src_ip == flow.src_ip and pkt.src_port == flow.src_port)
        direction = 1 if is_fwd else -1
        flow.packet_directions.append(direction)
        if is_fwd:
            flow.fwd_sizes.append(pkt.length)

        if pkt.sni:
            flow.sni_list.append(pkt.sni)

        if flow.n_packets >= max_packets:
            if flow.n_packets >= min_packets:
                feats = extract_features(flow)
                completed_flows.append((flow, feats))
            del active_flows[key]

        if pkt_count % 5000 == 0:
            now = pkt.timestamp
            expired_keys = [k for k, f in active_flows.items()
                            if now - f.last_seen > timeout]
            for k in expired_keys:
                f = active_flows.pop(k)
                if f.n_packets >= min_packets:
                    feats = extract_features(f)
                    completed_flows.append((f, feats))

    for key, flow in list(active_flows.items()):
        if flow.n_packets >= min_packets:
            feats = extract_features(flow)
            completed_flows.append((flow, feats))

    elapsed = time.time() - t0
    print(f"\n  Extraction complete in {elapsed:.1f}s")
    print(f"  Total packets: {pkt_count}")
    print(f"  Total flows started: {flow_counter}")
    print(f"  Completed flows (>={min_packets} pkts): {len(completed_flows)}")

    if not completed_flows:
        print("[WARNING] No flows extracted! Check PCAP content and filters.")
        return pd.DataFrame()

    rows = []
    for flow, features in completed_flows:
        label = classify_benign_subtype(flow, features)
        row = features.copy()
        row["label"] = label
        row["flow_id"] = flow.flow_id
        row["source"] = "real_world"
        row["sni"] = flow.primary_sni
        row["src_ip"] = flow.src_ip
        row["dst_ip"] = flow.dst_ip
        row["n_packets_raw"] = flow.n_packets
        rows.append(row)

    df = pd.DataFrame(rows)

    feature_cols = [c for c in df.columns if c in ALL_FEATURES]
    print(f"\n  Feature columns matched: {len(feature_cols)}/{N_FEATURES}")
    if len(feature_cols) != N_FEATURES:
        missing = set(ALL_FEATURES) - set(feature_cols)
        extra = set(feature_cols) - set(ALL_FEATURES)
        if missing:
            print(f"  [WARNING] Missing features: {missing}")
        if extra:
            print(f"  [WARNING] Extra features: {extra}")

    df[ALL_FEATURES] = df[ALL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"\n  Label distribution:")
    for label, count in df["label"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {label:30s} {count:5d}  ({pct:5.1f}%)")

    sni_counts = df[df["sni"] != ""]["sni"].value_counts()
    if len(sni_counts) > 0:
        print(f"\n  Top SNI domains:")
        for sni, count in sni_counts.head(10).items():
            print(f"    {sni:40s} {count:5d}")

    print(f"\n  Feature Statistics (real-world):")
    print(f"  {'Feature':25s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s}")
    print(f"  {'-'*73}")
    for col in ALL_FEATURES[:12]:
        mean = df[col].mean()
        std = df[col].std()
        mn = df[col].min()
        mx = df[col].max()
        print(f"  {col:25s} {mean:12.4f} {std:12.4f} {mn:12.4f} {mx:12.4f}")
    print(f"  ... ({N_FEATURES - 12} more features)")

    return df


def export_dataset(df: pd.DataFrame, output_prefix: str) -> dict:
    output_dir = os.path.dirname(output_prefix) or "output"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = f"{output_prefix}.csv"
    parquet_path = f"{output_prefix}.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    print(f"\n[EXPORT] CSV     → {csv_path}  ({len(df)} rows, {os.path.getsize(csv_path)/1024:.1f} KB)")
    print(f"[EXPORT] Parquet → {parquet_path}  ({os.path.getsize(parquet_path)/1024:.1f} KB)")
    return {"csv": csv_path, "parquet": parquet_path}


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from real-world QUIC PCAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pcap", default="benign_real_world.pcap",
                        help="Path to PCAP file (default: benign_real_world.pcap)")
    parser.add_argument("--output", default="output/real_world_dataset",
                        help="Output prefix (default: output/real_world_dataset)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Flow timeout in seconds (default: 60)")
    parser.add_argument("--min-packets", type=int, default=10,
                        help="Min packets per flow (default: 10)")
    parser.add_argument("--max-packets", type=int, default=5000,
                        help="Max packets per flow (default: 5000)")
    parser.add_argument("--no-quic-filter", action="store_true",
                        help="Don't filter to QUIC-only packets")
    args = parser.parse_args()

    pcap_path = args.pcap
    if not os.path.isabs(pcap_path):
        pcap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pcap_path)

    df = extract_from_pcap(
        pcap_path=pcap_path,
        timeout=args.timeout,
        min_packets=args.min_packets,
        max_packets=args.max_packets,
        quic_only=not args.no_quic_filter,
    )

    if len(df) > 0:
        export_dataset(df, args.output)
        print(f"\n{'#'*60}")
        print(f"  EXTRACTION COMPLETE: {len(df)} flows")
        print(f"{'#'*60}\n")
    else:
        print("\n[ERROR] No flows extracted. The PCAP may not contain QUIC traffic.")
        print("  Try running with --no-quic-filter to check all traffic.")


if __name__ == "__main__":
    main()
