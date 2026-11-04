"""
Diverse benign traffic generators for anomaly detector training.

Each generator produces realistic packet-level flows using simulate_flow_features()
from training.utils. These cover the full spectrum of real-world QUIC behavior:
unidirectional streaming, uploads, downloads, bidirectional browsing, idle connections.

This module is the direct fix for the proven benign diversity gap that caused
the anomaly detector to flag all live traffic as anomalous.
"""

import numpy as np
import pandas as pd

# Reuse the same feature extraction function used for all training data
from training.utils import simulate_flow_features

_rng = np.random.default_rng(42)


def _set_seed(seed: int):
    global _rng
    _rng = np.random.default_rng(seed)


# ─── 1. YouTube-style adaptive streaming ────────────────────────────
def gen_youtube_streaming(n_samples: int = 200, rng=None) -> pd.DataFrame:
    """95% inbound. Large bursts with buffer refill gaps. Adaptive bitrate."""
    r = rng or _rng
    rows = []
    for _ in range(n_samples):
        n_pkts = r.integers(100, 600)
        sizes, iats, dirs = [], [], []
        mtu = r.integers(1200, 1450)
        for i in range(n_pkts):
            # 95% of packets are inbound (server → client)
            if r.random() < 0.95:
                sizes.append(mtu + r.integers(-100, 50))
                dirs.append(-1)
            else:
                # Client ACKs / range requests
                sizes.append(r.integers(40, 120))
                dirs.append(1)
            if i > 0:
                # Adaptive: occasional buffer refill gaps
                if r.random() < 0.08:
                    iats.append(r.uniform(0.2, 2.0))  # Buffer refill
                else:
                    iats.append(r.uniform(0.001, 0.02))  # Burst
        rows.append({**simulate_flow_features(sizes, iats, dirs), "label": "benign_youtube_stream"})
    return pd.DataFrame(rows)


# ─── 2. Adaptive video (quality switches) ───────────────────────────
def gen_adaptive_video(n_samples: int = 200, rng=None) -> pd.DataFrame:
    """90% inbound. Variable chunk sizes simulating bitrate adaptation."""
    r = rng or _rng
    rows = []
    for _ in range(n_samples):
        n_pkts = r.integers(80, 400)
        sizes, iats, dirs = [], [], []
        quality = r.choice([480, 720, 1080])  # Starting quality
        for i in range(n_pkts):
            # Quality switch every ~50 packets
            if i > 0 and i % r.integers(30, 70) == 0:
                quality = r.choice([480, 720, 1080])
            if r.random() < 0.90:
                base = {480: 600, 720: 900, 1080: 1350}[quality]
                sizes.append(base + r.integers(-100, 100))
                dirs.append(-1)
            else:
                sizes.append(r.integers(40, 100))
                dirs.append(1)
            if i > 0:
                if r.random() < 0.05:
                    iats.append(r.uniform(0.5, 3.0))  # Rebuffer
                else:
                    iats.append(r.uniform(0.002, 0.03))
        rows.append({**simulate_flow_features(sizes, iats, dirs), "label": "benign_adaptive_video"})
    return pd.DataFrame(rows)


# ─── 3. Discord voice ───────────────────────────────────────────────
def gen_discord_voice(n_samples: int = 200, rng=None) -> pd.DataFrame:
    """Balanced bidirectional. Small constant-rate packets, low jitter."""
    r = rng or _rng
    rows = []
    for _ in range(n_samples):
        duration = r.uniform(10, 120)  # 10s to 2min call
        pps = r.uniform(40, 60)  # ~50 packets/sec (Opus codec)
        n_pkts = int(duration * pps)
        n_pkts = min(n_pkts, 1500)
        sizes, iats, dirs = [], [], []
        iat_base = 1.0 / pps
        for i in range(n_pkts):
            # Voice packets are small and bidirectional
            sizes.append(r.integers(80, 200))
            dirs.append(1 if r.random() < 0.5 else -1)
            if i > 0:
                iats.append(iat_base * r.uniform(0.8, 1.2))  # Low jitter
        rows.append({**simulate_flow_features(sizes, iats, dirs), "label": "benign_discord_voice"})
    return pd.DataFrame(rows)


# ─── 4. Discord chat ────────────────────────────────────────────────
def gen_discord_chat(n_samples: int = 200, rng=None) -> pd.DataFrame:
    """Bidirectional bursty. Irregular bursts, variable sizes."""
    r = rng or _rng
    rows = []
    for _ in range(n_samples):
        n_msgs = r.integers(5, 40)
        sizes, iats, dirs = [], [], []
        for m in range(n_msgs):
            # Outgoing message
            msg_size = r.integers(100, 800)
            sizes.append(msg_size)
            dirs.append(1)
            if m > 0:
                iats.append(r.uniform(0.5, 15.0))  # Human typing delay
            # Server ACK + delivery confirmation
            sizes.append(r.integers(40, 200))
            dirs.append(-1)
            iats.append(r.uniform(0.02, 0.2))
            # Possible response message
            if r.random() < 0.4:
                sizes.append(r.integers(100, 600))
                dirs.append(-1)
                iats.append(r.uniform(0.5, 10.0))
        rows.append({**simulate_flow_features(sizes, iats, dirs), "label": "benign_discord_chat"})
    return pd.DataFrame(rows)


# ─── 5. Telegram messaging ──────────────────────────────────────────
def gen_telegram_messaging(n_samples: int = 200, rng=None) -> pd.DataFrame:
    """Bidirectional. Small packets, irregular cadence."""
    r = rng or _rng
    rows = []
    for _ in range(n_samples):
        n_msgs = r.integers(3, 30)
        sizes, iats, dirs = [], [], []
        for m in range(n_msgs):
            sizes.append(r.integers(60, 500))
            dirs.append(1)
            if m > 0: iats.append(r.uniform(1.0, 30.0))
            # Server response
            sizes.append(r.integers(40, 300))
            dirs.append(-1)
            iats.append(r.uniform(0.05, 0.5))
        rows.append({**simulate_flow_features(sizes, iats, dirs), "label": "benign_telegram"})
    return pd.DataFrame(rows)


# ─── 6. Google Drive upload ─────────────────────────────────────────
def gen_gdrive_upload(n_samples: int = 200, rng=None) -> pd.DataFrame:
    """95% outbound. Large MTU chunks, sparse ACKs. Pure upload pattern."""
    r = rng or _rng
    rows = []
    for _ in range(n_samples):
        file_size = r.integers(100_000, 5_000_000)
        sizes, iats, dirs = [], [], []
        mtu = r.integers(1200, 1450)
        sent = 0
        while sent < file_size:
            chunk = min(mtu, file_size - sent)
            sizes.append(chunk)
            dirs.append(1)
            if len(sizes) > 1: iats.append(r.uniform(0.0005, 0.008))
            sent += chunk
            # Sparse ACK from server
            if r.random() < 0.05:
                sizes.append(r.integers(40, 80))
                dirs.append(-1)
                iats.append(r.uniform(0.01, 0.1))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_gdrive_upload"})
    return pd.DataFrame(rows)


# ─── 7. Google Drive download ───────────────────────────────────────
def gen_gdrive_download(n_samples: int = 200, rng=None) -> pd.DataFrame:
    """95% inbound. Large MTU chunks, sparse ACKs. Pure download pattern."""
    r = rng or _rng
    rows = []
    for _ in range(n_samples):
        file_size = r.integers(100_000, 5_000_000)
        sizes, iats, dirs = [], [], []
        mtu = r.integers(1200, 1450)
        received = 0
        while received < file_size:
            chunk = min(mtu, file_size - received)
            sizes.append(chunk)
            dirs.append(-1)
            if len(sizes) > 1: iats.append(r.uniform(0.0005, 0.008))
            received += chunk
            if r.random() < 0.05:
                sizes.append(r.integers(40, 80))
                dirs.append(1)
                iats.append(r.uniform(0.01, 0.1))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_gdrive_download"})
    return pd.DataFrame(rows)


# ─── 8. Git clone ───────────────────────────────────────────────────
def gen_git_clone(n_samples: int = 200, rng=None) -> pd.DataFrame:
    """90% inbound. Initial negotiation then steady download stream."""
    r = rng or _rng
    rows = []
    for _ in range(n_samples):
        sizes, iats, dirs = [], [], []
        # Negotiation phase: bidirectional
        for i in range(r.integers(5, 20)):
            sizes.append(r.integers(100, 500))
            dirs.append(1 if i % 2 == 0 else -1)
            if i > 0: iats.append(r.uniform(0.01, 0.1))
        # Pack download: mostly inbound
        n_data = r.integers(50, 400)
        for i in range(n_data):
            sizes.append(r.integers(800, 1450))
            dirs.append(-1)
            iats.append(r.uniform(0.001, 0.01))
            if r.random() < 0.08:
                sizes.append(r.integers(40, 80))
                dirs.append(1)
                iats.append(r.uniform(0.005, 0.05))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_git_clone"})
    return pd.DataFrame(rows)


# ─── 9. Browser HTTP/3 browsing ─────────────────────────────────────
def gen_browser_http3(n_samples: int = 200, rng=None) -> pd.DataFrame:
    """Bidirectional. Multiple request-response cycles, variable objects."""
    r = rng or _rng
    rows = []
    for _ in range(n_samples):
        n_objects = r.integers(5, 30)
        sizes, iats, dirs = [], [], []
        for obj in range(n_objects):
            # Request
            sizes.append(r.integers(100, 600))
            dirs.append(1)
            if obj > 0: iats.append(r.uniform(0.01, 0.5))
            # Response (1-10 packets)
            resp_pkts = r.integers(1, 10)
            for p in range(resp_pkts):
                sizes.append(r.integers(200, 1450))
                dirs.append(-1)
                iats.append(r.uniform(0.001, 0.02))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_browser_http3"})
    return pd.DataFrame(rows)


# ─── 10. Software update download ───────────────────────────────────
def gen_software_update(n_samples: int = 200, rng=None) -> pd.DataFrame:
    """98% inbound. Long steady download with rare ACKs."""
    r = rng or _rng
    rows = []
    for _ in range(n_samples):
        n_pkts = r.integers(200, 800)
        sizes, iats, dirs = [], [], []
        mtu = r.integers(1300, 1450)
        for i in range(n_pkts):
            if r.random() < 0.98:
                sizes.append(mtu + r.integers(-30, 10))
                dirs.append(-1)
            else:
                sizes.append(r.integers(40, 60))
                dirs.append(1)
            if i > 0:
                iats.append(r.uniform(0.0005, 0.005))
        rows.append({**simulate_flow_features(sizes, iats, dirs), "label": "benign_sw_update"})
    return pd.DataFrame(rows)


# ─── 11. Idle keepalive ─────────────────────────────────────────────
def gen_idle_keepalive(n_samples: int = 200, rng=None) -> pd.DataFrame:
    """Bidirectional sparse. Tiny packets, very long IATs. Idle connections."""
    r = rng or _rng
    rows = []
    for _ in range(n_samples):
        n_pings = r.integers(5, 30)
        sizes, iats, dirs = [], [], []
        for i in range(n_pings):
            sizes.append(r.integers(20, 80))
            dirs.append(1)
            if i > 0: iats.append(r.uniform(5.0, 60.0))  # Long idle gaps
            # Response
            sizes.append(r.integers(20, 80))
            dirs.append(-1)
            iats.append(r.uniform(0.01, 0.2))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_idle_keepalive"})
    return pd.DataFrame(rows)


# ─── 12. Mixed browsing session ─────────────────────────────────────
def gen_mixed_browsing(n_samples: int = 200, rng=None) -> pd.DataFrame:
    """Bidirectional variable. Alternating active/idle, multi-page navigation."""
    r = rng or _rng
    rows = []
    for _ in range(n_samples):
        n_pages = r.integers(2, 8)
        sizes, iats, dirs = [], [], []
        for page in range(n_pages):
            # Page load burst
            n_objs = r.integers(3, 15)
            for obj in range(n_objs):
                sizes.append(r.integers(100, 400))
                dirs.append(1)
                if len(sizes) > 1: iats.append(r.uniform(0.001, 0.05))
                for _ in range(r.integers(1, 5)):
                    sizes.append(r.integers(300, 1450))
                    dirs.append(-1)
                    iats.append(r.uniform(0.001, 0.015))
            # Inter-page idle (reading time)
            if page < n_pages - 1:
                sizes.append(r.integers(20, 60))
                dirs.append(1)
                iats.append(r.uniform(2.0, 20.0))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_mixed_browse"})
    return pd.DataFrame(rows)


# ─── Master generator ───────────────────────────────────────────────
ALL_BENIGN_GENERATORS = [
    gen_youtube_streaming,
    gen_adaptive_video,
    gen_discord_voice,
    gen_discord_chat,
    gen_telegram_messaging,
    gen_gdrive_upload,
    gen_gdrive_download,
    gen_git_clone,
    gen_browser_http3,
    gen_software_update,
    gen_idle_keepalive,
    gen_mixed_browsing,
]


def generate_all_benign(n_per_class: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate diverse benign traffic from all generators."""
    rng = np.random.default_rng(seed)
    frames = []
    for gen_fn in ALL_BENIGN_GENERATORS:
        df = gen_fn(n_per_class, rng=rng)
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    print(f"  Generated {len(result)} diverse benign flows across {len(ALL_BENIGN_GENERATORS)} classes:")
    for label, count in result["label"].value_counts().items():
        print(f"    {label}: {count}")
    return result


if __name__ == "__main__":
    df = generate_all_benign(200)
    print(f"\n  Total: {len(df)} flows, {df['label'].nunique()} classes")
