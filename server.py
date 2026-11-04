import asyncio
import os
import random

from aioquic.asyncio import serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import StreamDataReceived, HandshakeCompleted
from aioquic.asyncio.protocol import QuicConnectionProtocol

import config


class DatasetServerProtocol(QuicConnectionProtocol):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stream_types: dict[int, int] = {}

    def quic_event_received(self, event):
        if isinstance(event, StreamDataReceived):
            self._handle_stream_data(event)

    def _safe_send(self, stream_id: int, data: bytes, end_stream: bool = False):
        try:
            self._quic.send_stream_data(stream_id, data, end_stream=end_stream)
            self.transmit()
        except Exception:
            pass

    def _handle_stream_data(self, event: StreamDataReceived):
        stream_id = event.stream_id
        data = event.data

        if not data:
            return

        if stream_id not in self._stream_types:
            traffic_type = data[0]
            self._stream_types[stream_id] = traffic_type
            payload = data[1:]
        else:
            traffic_type = self._stream_types[stream_id]
            payload = data

        if traffic_type == config.TRAFFIC_WEB_BROWSE:
            self._handle_web(stream_id, payload)
        elif traffic_type == config.TRAFFIC_STREAMING:
            self._handle_streaming(stream_id, payload)
        elif traffic_type in (config.TRAFFIC_EXFIL_SLOW, config.TRAFFIC_EXFIL_BURST):
            self._handle_exfil_ack(stream_id, payload)
        elif traffic_type == config.TRAFFIC_C2_BEACON:
            self._handle_c2(stream_id, payload)
        elif traffic_type == config.TRAFFIC_TUNNEL:
            self._handle_tunnel(stream_id, payload)
        else:
            self._safe_send(stream_id, data, end_stream=True)


    def _handle_web(self, stream_id: int, payload: bytes):
        if len(payload) >= 4:
            resp_size = int.from_bytes(payload[:4], "big")
        else:
            resp_size = 4096
        resp_size = min(resp_size, 500_000)

        response = os.urandom(resp_size)
        self._safe_send(stream_id, response, end_stream=True)

    def _handle_streaming(self, stream_id: int, payload: bytes):
        if len(payload) >= 4:
            num_chunks = int.from_bytes(payload[:4], "big")
        else:
            num_chunks = 20
        num_chunks = min(num_chunks, 200)
        asyncio.ensure_future(self._stream_chunks(stream_id, num_chunks))

    async def _stream_chunks(self, stream_id: int, num_chunks: int):
        for i in range(num_chunks):
            chunk_size = random.randint(8_000, 64_000)
            chunk = os.urandom(chunk_size)
            end = (i == num_chunks - 1)
            self._safe_send(stream_id, chunk, end_stream=end)
            await asyncio.sleep(random.uniform(0.02, 0.15))

    def _handle_exfil_ack(self, stream_id: int, payload: bytes):
        self._safe_send(stream_id, b"\x00", end_stream=False)

    def _handle_c2(self, stream_id: int, payload: bytes):
        cmd_size = random.randint(16, 256)
        command = os.urandom(cmd_size)
        self._safe_send(stream_id, command, end_stream=True)

    def _handle_tunnel(self, stream_id: int, payload: bytes):
        resp_size = max(len(payload), 64)
        response = os.urandom(resp_size)
        self._safe_send(stream_id, response, end_stream=False)


async def run_server(ready_event: asyncio.Event = None):
    server_config = QuicConfiguration(
        is_client=False,
        alpn_protocols=["quic-ids-lab"],
        max_datagram_frame_size=65536,
    )
    server_config.load_cert_chain(config.CERT_FILE, config.KEY_FILE)

    server = await serve(
        config.SERVER_HOST,
        config.SERVER_PORT,
        configuration=server_config,
        create_protocol=DatasetServerProtocol,
    )

    print(f"[SERVER] Listening on {config.SERVER_HOST}:{config.SERVER_PORT}")

    if ready_event is not None:
        ready_event.set()

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    from generate_certs import generate_self_signed_cert

    if not os.path.exists(config.CERT_FILE):
        generate_self_signed_cert()

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\n[SERVER] Stopped.")
