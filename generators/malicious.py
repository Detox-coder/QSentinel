"""
Malicious QUIC traffic generators.

1. SlowExfilGenerator   — low-and-slow data exfiltration
2. BurstExfilGenerator  — rapid bulk exfiltration
3. C2BeaconGenerator    — command-and-control beaconing
4. TunnelGenerator      — protocol tunneling simulation
"""

import asyncio
import os
import random
import struct

import config
from generators.base import BaseGenerator


# ─────────────────────────────────────────────────────────────
#  Data Exfiltration — Slow Drip
# ─────────────────────────────────────────────────────────────

class SlowExfilGenerator(BaseGenerator):
    """
    Simulates slow data exfiltration.

    Pattern:
      • Long-lived connection (30–90 s)
      • Small packets (64–512 bytes) sent at regular intervals (1–5 s)
      • Mimics a trojan slowly leaking data
      • Very little server response (just ACKs)
      • High forward/backward byte ratio
    """

    traffic_type = config.TRAFFIC_EXFIL_SLOW
    label = "malicious_exfil_slow"
    description = "Slow data exfiltration"

    async def _traffic_pattern(self, protocol, flow_id: str):
        reader, writer = await protocol.create_stream()

        duration = random.uniform(15.0, 45.0)
        interval = random.uniform(1.0, 4.0)
        chunk_size_range = (64, 512)

        elapsed = 0.0
        while elapsed < duration:
            # Send a small chunk of "stolen" data
            chunk_size = random.randint(*chunk_size_range)
            payload = bytes([self.traffic_type]) + os.urandom(chunk_size)
            writer.write(payload)

            # Read server ACK (tiny)
            try:
                await asyncio.wait_for(reader.read(16), timeout=3.0)
            except asyncio.TimeoutError:
                pass

            jitter = random.uniform(-0.3, 0.3)
            sleep_time = max(0.5, interval + jitter)
            await asyncio.sleep(sleep_time)
            elapsed += sleep_time

        writer.write_eof()


# ─────────────────────────────────────────────────────────────
#  Data Exfiltration — Burst
# ─────────────────────────────────────────────────────────────

class BurstExfilGenerator(BaseGenerator):
    """
    Simulates burst data exfiltration.

    Pattern:
      • Short connection (2–8 s)
      • Initial quiet period
      • Sudden large upload (50 KB–500 KB in rapid small writes)
      • Minimal server response
      • Asymmetric traffic (mostly client→server)
    """

    traffic_type = config.TRAFFIC_EXFIL_BURST
    label = "malicious_exfil_burst"
    description = "Burst data exfiltration"

    async def _traffic_pattern(self, protocol, flow_id: str):
        reader, writer = await protocol.create_stream()

        # Initial quiet period (looks like idle connection)
        await asyncio.sleep(random.uniform(0.5, 2.0))

        # Burst: send a large amount of data in rapid succession
        total_exfil = random.randint(50_000, 500_000)
        sent = 0
        while sent < total_exfil:
            chunk_size = min(random.randint(1_000, 8_000), total_exfil - sent)
            payload = bytes([self.traffic_type]) + os.urandom(chunk_size)
            writer.write(payload)
            sent += chunk_size
            # Very short delay between chunks (rapid fire)
            await asyncio.sleep(random.uniform(0.001, 0.02))

        writer.write_eof()

        # Read final ACK
        try:
            await asyncio.wait_for(reader.read(64), timeout=3.0)
        except asyncio.TimeoutError:
            pass


# ─────────────────────────────────────────────────────────────
#  Command & Control Beaconing
# ─────────────────────────────────────────────────────────────

class C2BeaconGenerator(BaseGenerator):
    """
    Simulates C2 beaconing over QUIC.

    Pattern:
      • Periodic check-ins (beacon interval 2–10 s with jitter)
      • Each beacon: small request → small command response
      • Occasional larger response (command with payload)
      • Regular timing with slight jitter (characteristic of malware)
    """

    traffic_type = config.TRAFFIC_C2_BEACON
    label = "malicious_c2_beacon"
    description = "C2 beaconing"

    async def _traffic_pattern(self, protocol, flow_id: str):
        beacon_interval = random.uniform(2.0, 8.0)
        jitter_pct = random.uniform(0.05, 0.25)
        num_beacons = random.randint(5, 20)

        for i in range(num_beacons):
            reader, writer = await protocol.create_stream()

            # Beacon check-in (small)
            beacon_size = random.randint(16, 128)
            beacon = bytes([self.traffic_type]) + os.urandom(beacon_size)
            writer.write(beacon)
            writer.write_eof()

            # Receive command from C2
            try:
                response = await asyncio.wait_for(reader.read(), timeout=5.0)
            except asyncio.TimeoutError:
                response = b""

            # Occasionally send task result back on a new stream
            if random.random() < 0.3 and response:
                r2, w2 = await protocol.create_stream()
                result_size = random.randint(256, 4_096)
                result = bytes([self.traffic_type]) + os.urandom(result_size)
                w2.write(result)
                w2.write_eof()
                try:
                    await asyncio.wait_for(r2.read(), timeout=3.0)
                except asyncio.TimeoutError:
                    pass

            # Wait for next beacon (interval + jitter)
            jitter = beacon_interval * random.uniform(-jitter_pct, jitter_pct)
            await asyncio.sleep(beacon_interval + jitter)


# ─────────────────────────────────────────────────────────────
#  QUIC Tunneling
# ─────────────────────────────────────────────────────────────

class TunnelGenerator(BaseGenerator):
    """
    Simulates protocol tunneling over QUIC.

    Pattern:
      • Bidirectional chatty traffic (like SSH/RDP inside QUIC)
      • Multiple concurrent streams
      • Varied packet sizes in both directions
      • Persistent connection
      • Higher direction-change frequency than normal HTTPS
    """

    traffic_type = config.TRAFFIC_TUNNEL
    label = "malicious_tunnel"
    description = "Protocol tunneling over QUIC"

    async def _traffic_pattern(self, protocol, flow_id: str):
        num_exchanges = random.randint(15, 50)
        concurrent_streams = random.randint(2, 4)

        tasks = [
            self._tunnel_stream(protocol, num_exchanges // concurrent_streams)
            for _ in range(concurrent_streams)
        ]
        await asyncio.gather(*tasks)

    async def _tunnel_stream(self, protocol, exchanges: int):
        """Simulate a single tunnel stream with bidirectional traffic."""
        reader, writer = await protocol.create_stream()

        for _ in range(exchanges):
            # Client → Server (tunneled request)
            req_size = random.randint(32, 2_048)
            writer.write(bytes([self.traffic_type]) + os.urandom(req_size))

            # Read tunneled response
            try:
                resp = await asyncio.wait_for(reader.read(65536), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            # Short delay (interactive protocol pace)
            await asyncio.sleep(random.uniform(0.05, 0.5))

        writer.write_eof()

# ─────────────────────────────────────────────────────────────
#  Stealth C2 (Disguised)
# ─────────────────────────────────────────────────────────────

class StealthC2Generator(BaseGenerator):
    """
    Simulates a highly evasive C2 that mimics benign streaming.
    """
    traffic_type = config.TRAFFIC_STEALTH_C2
    label = "malicious_stealth_c2"
    description = "Stealth C2 (mimics streaming)"

    async def _traffic_pattern(self, protocol, flow_id: str):
        reader, writer = await protocol.create_stream()
        num_chunks = random.randint(10, 30)
        request = bytes([self.traffic_type]) + struct.pack("!I", num_chunks)
        writer.write(request)
        writer.write_eof()
        for _ in range(num_chunks):
            try:
                await asyncio.wait_for(reader.read(65536), timeout=5.0)
            except asyncio.TimeoutError:
                break
            await asyncio.sleep(random.uniform(0.1, 1.5))


# ─────────────────────────────────────────────────────────────
#  Hybrid Attack
# ─────────────────────────────────────────────────────────────

class HybridAttackGenerator(BaseGenerator):
    """
    Simulates a hybrid attack: benign browsing followed by sudden exfil.
    """
    traffic_type = config.TRAFFIC_HYBRID
    label = "malicious_hybrid"
    description = "Hybrid Attack (Browsing -> Burst Exfil)"

    async def _traffic_pattern(self, protocol, flow_id: str):
        # Benign phase
        num_requests = random.randint(1, 3)
        for _ in range(num_requests):
            reader, writer = await protocol.create_stream()
            resp_size = random.randint(1_024, 50_000)
            request = bytes([config.TRAFFIC_WEB_BROWSE]) + struct.pack("!I", resp_size)
            writer.write(request)
            writer.write_eof()
            try:
                await asyncio.wait_for(reader.read(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # Malicious phase (Exfil)
        r2, w2 = await protocol.create_stream()
        total_exfil = random.randint(50_000, 200_000)
        sent = 0
        while sent < total_exfil:
            chunk_size = min(random.randint(2_000, 10_000), total_exfil - sent)
            payload = bytes([self.traffic_type]) + os.urandom(chunk_size)
            w2.write(payload)
            sent += chunk_size
            await asyncio.sleep(random.uniform(0.005, 0.05))
        w2.write_eof()
