"""
Base class for QUIC traffic generators.

Every generator:
  1. Opens a QUIC connection to the lab server
  2. Produces a specific traffic pattern
  3. Logs every datagram via the instrumented protocol
"""

import asyncio
import ssl
import uuid
from abc import ABC, abstractmethod

from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration

import config
from packet_logger import PacketLogger
from protocol import InstrumentedClientProtocol


class BaseGenerator(ABC):
    """Abstract traffic generator."""

    traffic_type: int = 0          # override in subclass
    label: str = "unknown"         # override in subclass
    description: str = ""          # human-readable

    def __init__(self, logger: PacketLogger):
        self.logger = logger

    # ── Public API ───────────────────────────────────────

    async def generate(self, count: int = 1) -> list[str]:
        """
        Generate ``count`` independent flows and return their flow IDs.
        """
        flow_ids = []
        for i in range(count):
            flow_id = f"{self.label}_{uuid.uuid4().hex[:8]}"
            self.logger.register_flow(flow_id, self.label, self.traffic_type)

            print(f"  [{self.label}] flow {i+1}/{count}  id={flow_id}")
            try:
                await self._run_single_flow(flow_id)
            except Exception as exc:
                print(f"  [!] Flow {flow_id} failed: {exc}")
            flow_ids.append(flow_id)
        return flow_ids

    # ── Internal ─────────────────────────────────────────

    async def _run_single_flow(self, flow_id: str):
        """Connect to the server, run the traffic pattern, disconnect."""
        client_config = QuicConfiguration(
            is_client=True,
            alpn_protocols=["quic-ids-lab"],
            max_datagram_frame_size=65536,
        )
        # Trust self-signed cert
        client_config.verify_mode = ssl.CERT_NONE

        # Bind the class-level state for the instrumented protocol
        InstrumentedClientProtocol.packet_logger = self.logger
        InstrumentedClientProtocol.current_flow_id = flow_id

        async with connect(
            config.SERVER_HOST,
            config.SERVER_PORT,
            configuration=client_config,
            create_protocol=InstrumentedClientProtocol,
        ) as protocol:
            await self._traffic_pattern(protocol, flow_id)

    @abstractmethod
    async def _traffic_pattern(self, protocol, flow_id: str):
        """
        Implement the actual traffic pattern using ``protocol``.

        Use ``protocol.create_stream()`` to open QUIC streams and
        exchange data with the server.
        """
        ...
