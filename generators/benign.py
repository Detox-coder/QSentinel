"""
Benign QUIC traffic generators.

1. WebBrowsingGenerator  — simulates HTTP/3 page loads
2. StreamingGenerator    — simulates video/audio streaming
"""

import asyncio
import os
import random
import struct

import config
from generators.base import BaseGenerator


class WebBrowsingGenerator(BaseGenerator):
    """
    Simulates a user browsing web pages over HTTP/3.

    Pattern:
      • Open connection
      • Send 2–8 small requests (each on a separate stream)
      • Each request triggers a variable-size response (1 KB–200 KB)
      • Random think-time between requests (0.1–2 s)
      • Close connection
    """

    traffic_type = config.TRAFFIC_WEB_BROWSE
    label = "benign_web"
    description = "Web browsing (HTTP/3 page loads)"

    async def _traffic_pattern(self, protocol, flow_id: str):
        num_requests = random.randint(2, 8)

        for _ in range(num_requests):
            reader, writer = await protocol.create_stream()

            # Request: traffic_type byte + 4 bytes desired response size
            resp_size = random.randint(1_024, 200_000)
            request = bytes([self.traffic_type]) + struct.pack("!I", resp_size)

            # Simulate request payload (headers + small body)
            request += os.urandom(random.randint(64, 512))
            writer.write(request)
            writer.write_eof()

            # Read full response
            response = await asyncio.wait_for(reader.read(), timeout=10.0)

            # Think time between requests
            await asyncio.sleep(random.uniform(0.1, 2.0))


class StreamingGenerator(BaseGenerator):
    """
    Simulates video/audio streaming over QUIC.

    Pattern:
      • Single long-lived connection
      • Small initial request
      • Server sends many chunks (8–64 KB each) with short intervals
      • Client occasionally sends small keep-alive / range requests
    """

    traffic_type = config.TRAFFIC_STREAMING
    label = "benign_streaming"
    description = "Video/audio streaming"

    async def _traffic_pattern(self, protocol, flow_id: str):
        reader, writer = await protocol.create_stream()

        # Initial request: traffic_type + number of chunks
        num_chunks = random.randint(10, 60)
        request = bytes([self.traffic_type]) + struct.pack("!I", num_chunks)
        writer.write(request)
        writer.write_eof()

        # Read streaming data (server sends chunks with delays)
        total_received = 0
        while True:
            try:
                chunk = await asyncio.wait_for(reader.read(65536), timeout=15.0)
                if not chunk:
                    break
                total_received += len(chunk)
            except asyncio.TimeoutError:
                break

        # Optionally open a second stream for "quality change" request
        if random.random() < 0.4:
            r2, w2 = await protocol.create_stream()
            w2.write(bytes([self.traffic_type]) + struct.pack("!I", 5))
            w2.write_eof()
            try:
                await asyncio.wait_for(r2.read(), timeout=10.0)
            except asyncio.TimeoutError:
                pass
