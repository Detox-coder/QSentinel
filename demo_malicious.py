#!/usr/bin/env python3
"""
v6 Malicious QUIC Traffic Simulator.

Generates realistic attack traffic on loopback for IDS testing.
Improvements over v5:
  - Randomized timing with jitter distributions
  - Background benign-like packet mixing
  - Variable beacon cadence with drift and missed intervals
  - Realistic packet sizing (MTU-aware)
  - Longer sessions for better flow analysis
  - Configurable flow expiration wait time
"""

import argparse
import os
import socket
import struct
import time
import random
import threading


TARGET_IP = "127.0.0.1"


def make_quic_header() -> bytes:
    first_byte = 0xC0
    version = 0x00000001
    dcid = os.urandom(8)
    scid = os.urandom(8)
    header = struct.pack("!B I", first_byte, version)
    header += struct.pack("!B", len(dcid)) + dcid
    header += struct.pack("!B", len(scid)) + scid
    return header


HEADER = make_quic_header()


def send_attack(name, description, attack_fn, wait_time=10):
    print(f"\n  [ATTACK] {name}")
    print(f"           {description}")

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_port = random.randint(40000, 49999)
    client_sock.bind(("127.0.0.1", client_port))

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_port = 443
    try:
        server_sock.bind(("127.0.0.1", server_port))
    except (PermissionError, OSError):
        server_port = 8443
        try:
            server_sock.bind(("127.0.0.1", server_port))
        except OSError:
            server_port = random.randint(8000, 9000)
            server_sock.bind(("127.0.0.1", server_port))
        print(f"           [!] Using alternate port {server_port}")

    count = attack_fn(client_sock, client_port, server_sock, server_port)

    client_sock.close()
    server_sock.close()

    print(f"\n           Done: {count} total packets exchanged")
    print(f"           Waiting {wait_time}s for flow expiration...")
    time.sleep(wait_time)
    return count


def _send(sock, payload, dst_port):
    sock.sendto(payload, (TARGET_IP, dst_port))


def attack_c2_beacon(c_sock, c_port, s_sock, s_port) -> int:
    """C2 beacon with realistic drift, jitter, and missed intervals."""
    count = 0
    n_beacons = random.randint(20, 60)
    base_interval = random.uniform(0.3, 2.0)
    jitter_pct = random.uniform(0.05, 0.25)
    drift = random.uniform(-0.005, 0.005)  # Interval slowly changes

    for i in range(n_beacons):
        # Occasionally miss a beacon (real malware does this)
        if random.random() < 0.08:
            time.sleep(base_interval * random.uniform(1.5, 3.0))
            continue

        # Beacon packet (small, outbound)
        size = random.randint(20, 150)
        _send(c_sock, HEADER + os.urandom(size), s_port)
        count += 1

        # Response from C2 server (sometimes)
        if random.random() < 0.35:
            time.sleep(random.uniform(0.01, 0.1))
            resp_size = random.randint(40, 800)
            _send(s_sock, HEADER + os.urandom(resp_size), c_port)
            count += 1

        # Apply interval with drift and jitter
        cur_interval = max(0.1, base_interval + drift * i)
        actual_wait = cur_interval * (1 + random.uniform(-jitter_pct, jitter_pct))
        print(f"\r           Beacons: {i+1}/{n_beacons}  pkts: {count}", end="", flush=True)
        time.sleep(actual_wait)

    return count


def attack_exfil_burst(c_sock, c_port, s_sock, s_port) -> int:
    """Data exfiltration burst with handshake and variable chunk sizes."""
    count = 0
    total_bytes = random.randint(200_000, 1_500_000)

    # Handshake phase (looks normal)
    for _ in range(random.randint(3, 8)):
        _send(c_sock, HEADER + os.urandom(random.randint(40, 120)), s_port)
        count += 1
        time.sleep(random.uniform(0.1, 0.5))
        _send(s_sock, HEADER + os.urandom(random.randint(40, 120)), c_port)
        count += 1
        time.sleep(random.uniform(0.1, 0.5))

    # Quiet period before burst (stealth)
    time.sleep(random.uniform(0.5, 3.0))

    # Burst phase
    sent = 0
    while sent < total_bytes:
        chunk = random.randint(400, 1400)  # Variable chunk sizes
        _send(c_sock, HEADER + os.urandom(chunk), s_port)
        sent += chunk
        count += 1
        time.sleep(random.uniform(0.0002, 0.003))

        # Periodic ACKs from server
        if count % random.randint(15, 30) == 0:
            _send(s_sock, HEADER + os.urandom(random.randint(30, 80)), c_port)
            count += 1

        if count % 50 == 0:
            print(f"\r           Exfil: {sent//1000}/{total_bytes//1000} KB  pkts: {count}", end="", flush=True)

    return count


def attack_slow_exfil(c_sock, c_port, s_sock, s_port) -> int:
    """Low-and-slow exfiltration over extended period."""
    count = 0
    n_chunks = random.randint(25, 70)

    for i in range(n_chunks):
        # Variable chunk sizes (small to medium)
        chunk = random.randint(100, 800)
        _send(c_sock, HEADER + os.urandom(chunk), s_port)
        count += 1

        # Server response (mimics normal interaction)
        time.sleep(random.uniform(0.02, 0.1))
        if random.random() < 0.5:
            _send(s_sock, HEADER + os.urandom(random.randint(30, 150)), c_port)
            count += 1

        print(f"\r           Leaked: {i+1}/{n_chunks}  pkts: {count}", end="", flush=True)
        # Long variable pauses between chunks (stealth)
        time.sleep(random.uniform(0.3, 2.5))

    return count


def attack_tunnel(c_sock, c_port, s_sock, s_port) -> int:
    """Bidirectional tunnel (SSH/VPN-over-QUIC pattern)."""
    count = 0
    n_exchanges = random.randint(40, 100)

    for i in range(n_exchanges):
        # Client → Server (variable sizes)
        c_sz = random.choice([
            random.randint(40, 100),    # Small control
            random.randint(200, 600),   # Medium data
            random.randint(800, 1400),  # Large data
        ])
        _send(c_sock, HEADER + os.urandom(c_sz), s_port)
        count += 1
        time.sleep(random.uniform(0.003, 0.03))

        # Server → Client (variable sizes)
        s_sz = random.choice([
            random.randint(40, 100),
            random.randint(400, 1000),
            random.randint(1000, 1400),
        ])
        _send(s_sock, HEADER + os.urandom(s_sz), c_port)
        count += 1
        time.sleep(random.uniform(0.005, 0.05))

        # Occasional keepalive pause
        if random.random() < 0.05:
            time.sleep(random.uniform(0.5, 2.0))

        if count % 10 == 0:
            print(f"\r           Exchanges: {i+1}/{n_exchanges}  pkts: {count}", end="", flush=True)

    return count


def attack_stealth_c2(c_sock, c_port, s_sock, s_port) -> int:
    """Stealthy C2 that mimics legitimate HTTPS browsing patterns."""
    count = 0

    # Phase 1: Looks like a normal page load
    for _ in range(random.randint(5, 15)):
        _send(c_sock, HEADER + os.urandom(random.randint(100, 400)), s_port)
        count += 1
        time.sleep(random.uniform(0.01, 0.05))
        _send(s_sock, HEADER + os.urandom(random.randint(800, 1400)), c_port)
        count += 1
        time.sleep(random.uniform(0.005, 0.02))

    # Phase 2: Quiet period (simulates user reading)
    time.sleep(random.uniform(2.0, 5.0))

    # Phase 3: Hidden exfil in "keepalive" packets
    for i in range(random.randint(15, 40)):
        # Sends data disguised as small keepalives
        _send(c_sock, HEADER + os.urandom(random.randint(60, 250)), s_port)
        count += 1
        time.sleep(random.uniform(0.5, 3.0))

        # Occasional normal-looking response
        if random.random() < 0.3:
            _send(s_sock, HEADER + os.urandom(random.randint(40, 100)), c_port)
            count += 1

        print(f"\r           Stealth C2: {i+1} pkts: {count}", end="", flush=True)

    return count


def main():
    parser = argparse.ArgumentParser(description="v6 Malicious QUIC Traffic Simulator")
    parser.add_argument("--attack", default="all",
                        choices=["c2", "exfil", "slow_exfil", "tunnel", "stealth_c2", "all"])
    parser.add_argument("--wait", type=int, default=10,
                        help="Seconds to wait between attacks for flow expiration (default: 10)")
    args = parser.parse_args()

    print(f"""
+{'='*58}+
|  QSentinel v6 MALICIOUS TRAFFIC SIMULATOR                  |
|  Sends attack patterns on loopback (127.0.0.1)             |
+{'='*58}+
|  In another terminal run:                                   |
|  sudo python3 detect.py --live -i lo --no-quic-filter \\     |
|       --verbose --alert-level SAFE --timeout 5              |
|       --min-packets 5                                       |
+{'='*58}+""")

    attacks = {
        "c2": ("C2 BEACON ATTACK",
               "Periodic beacons with drift and missed intervals",
               attack_c2_beacon),
        "exfil": ("EXFIL BURST ATTACK",
                  "Handshake then rapid variable-size data burst",
                  attack_exfil_burst),
        "slow_exfil": ("SLOW EXFIL ATTACK",
                       "Low-and-slow data leakage with benign mixing",
                       attack_slow_exfil),
        "tunnel": ("TUNNEL ATTACK",
                   "Bidirectional exchanges (SSH/VPN-over-QUIC)",
                   attack_tunnel),
        "stealth_c2": ("STEALTH C2 ATTACK",
                       "C2 mimicking legitimate HTTPS browsing",
                       attack_stealth_c2),
    }

    if args.attack == "all":
        for name, (title, desc, fn) in attacks.items():
            send_attack(title, desc, fn, args.wait)
    else:
        title, desc, fn = attacks[args.attack]
        send_attack(title, desc, fn, args.wait)

    print(f"\n{'='*58}")
    print(f"  All attacks complete. Check detect.py terminal.")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
