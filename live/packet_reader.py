"""
Packet reader: ingests from tshark (live) or PCAP files.

Outputs a stream of RawPacket dataclass objects.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np


@dataclass
class RawPacket:
    """Single parsed packet."""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    length: int
    protocol: str
    is_quic: bool


TSHARK_FIELDS = [
    "frame.time_epoch",
    "ip.src",
    "ip.dst",
    "udp.srcport",
    "udp.dstport",
    "frame.len",
    "quic",
    "_ws.col.Protocol",
]


def _build_field_args() -> list[str]:
    args = []
    for f in TSHARK_FIELDS:
        args.extend(["-e", f])
    return args


def _parse_line(line: str) -> Optional[RawPacket]:
    """Parse a tshark pipe-delimited line."""
    parts = line.strip().split("|")
    if len(parts) < 6:
        return None
    try:
        ts = float(parts[0]) if parts[0] else 0.0
        # Handle comma-separated IPs from loopback (e.g., "127.0.0.1,127.0.0.1")
        src_ip = (parts[1] or "0.0.0.0").split(",")[0].strip()
        dst_ip = (parts[2] or "0.0.0.0").split(",")[0].strip()
        src_port = int(parts[3].split(",")[0]) if parts[3] else 0
        dst_port = int(parts[4].split(",")[0]) if parts[4] else 0
        length = int(parts[5]) if parts[5] else 0
        # Skip packets with no port info (non-UDP garbage on loopback)
        if src_port == 0 and dst_port == 0:
            return None
        if src_ip == "0.0.0.0" and dst_ip == "0.0.0.0":
            return None
        quic_present = bool(parts[6].strip()) if len(parts) > 6 else False
        protocol = parts[7].strip() if len(parts) > 7 else ""
        is_quic = quic_present or "QUIC" in protocol.upper()
        return RawPacket(ts, src_ip, dst_ip, src_port, dst_port, length, protocol, is_quic)
    except (ValueError, IndexError):
        return None


import queue
import threading

def read_live(interface: str = "any", quic_only: bool = True) -> Generator[Optional[RawPacket], None, None]:
    """Read live packets via tshark. Yields RawPacket objects or None if idle."""
    cmd = [
        "tshark", "-l", "-T", "fields",
        "-E", "separator=|", "-E", "header=n", "-E", "quote=n",
    ] + _build_field_args()
    if quic_only:
        cmd.extend(["-f", "udp port 443"])
    cmd.extend(["-i", interface])

    print(f"[READER] Starting tshark on '{interface}'")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        
        q = queue.Queue()
        def reader_thread():
            try:
                for line in proc.stdout:
                    q.put(line)
            except: pass
            
        t = threading.Thread(target=reader_thread, daemon=True)
        t.start()
        
        while True:
            try:
                line = q.get(timeout=0.5)
                pkt = _parse_line(line)
                if pkt and (not quic_only or pkt.is_quic):
                    yield pkt
            except queue.Empty:
                if proc.poll() is not None:
                    break  # tshark died
                yield None
                
    except FileNotFoundError:
        print("[ERROR] tshark not found. Install wireshark-cli.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[READER] Stopped.")
    finally:
        if "proc" in locals() and proc.poll() is None:
            proc.terminate()


def read_pcap(pcap_path: str, quic_only: bool = True) -> Generator[RawPacket, None, None]:
    """Read packets from a PCAP file."""
    if not os.path.exists(pcap_path):
        print(f"[ERROR] File not found: {pcap_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        "tshark", "-r", pcap_path, "-T", "fields",
        "-E", "separator=|", "-E", "header=n", "-E", "quote=n",
    ] + _build_field_args()

    print(f"[READER] Parsing PCAP: {pcap_path}")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        count = 0
        for line in proc.stdout:
            pkt = _parse_line(line)
            if pkt and (not quic_only or pkt.is_quic):
                count += 1
                yield pkt
        proc.wait()
        print(f"[READER] Finished: {count} QUIC packets extracted")
    except FileNotFoundError:
        print("[ERROR] tshark not found. Install wireshark-cli.", file=sys.stderr)
        sys.exit(1)


def read_simulated(n_flows: int = 20, seed: int = 42) -> Generator[RawPacket, None, None]:
    """Generate simulated QUIC-like packet stream for testing without tshark."""
    rng = np.random.default_rng(seed)
    print(f"[READER] Simulating {n_flows} QUIC flows")

    events = []
    t_base = time.time()

    for i in range(n_flows):
        src_ip = f"192.168.1.{rng.integers(1, 254)}"
        dst_ip = f"10.0.{rng.integers(0, 10)}.{rng.integers(1, 254)}"
        src_port = int(rng.integers(10000, 65535))
        n_pkts = int(rng.integers(15, 200))
        base_iat = rng.exponential(0.05)
        base_size = int(rng.integers(100, 1400))

        ts = t_base + rng.exponential(0.5)
        for _ in range(n_pkts):
            ts += base_iat * rng.exponential(1.0)
            size = max(40, int(base_size + rng.normal(0, base_size * 0.3)))
            events.append((ts, src_ip, dst_ip, src_port, 443, size))

    events.sort(key=lambda x: x[0])
    for ts, sip, dip, sp, dp, sz in events:
        yield RawPacket(ts, sip, dip, sp, dp, sz, "QUIC", True)

    print(f"[READER] Finished: {len(events)} simulated packets")
