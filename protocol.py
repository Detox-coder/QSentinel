import time
from aioquic.asyncio.protocol import QuicConnectionProtocol

from packet_logger import PacketLogger



class _LoggingTransport:

    def __init__(self, transport, logger: PacketLogger, flow_id: str):
        object.__setattr__(self, "_transport", transport)
        object.__setattr__(self, "_logger", logger)
        object.__setattr__(self, "_flow_id", flow_id)

    def sendto(self, data, addr=None):
        self._logger.record(self._flow_id, time.monotonic(), len(data), 1)
        if addr is not None:
            self._transport.sendto(data, addr)
        else:
            self._transport.sendto(data)

    def close(self):
        self._transport.close()

    def is_closing(self):
        return self._transport.is_closing()

    def get_extra_info(self, name, default=None):
        return self._transport.get_extra_info(name, default)

    def __getattr__(self, name):
        return getattr(self._transport, name)



class InstrumentedClientProtocol(QuicConnectionProtocol):
    packet_logger: PacketLogger = None
    current_flow_id: str = ""

    def connection_made(self, transport):
        wrapped = _LoggingTransport(
            transport,
            self.__class__.packet_logger,
            self.__class__.current_flow_id,
        )
        super().connection_made(wrapped)

    def datagram_received(self, data, addr):
        if self.__class__.packet_logger is not None:
            self.__class__.packet_logger.record(
                self.__class__.current_flow_id,
                time.monotonic(),
                len(data),
                -1,
            )
        super().datagram_received(data, addr)


class InstrumentedServerProtocol(QuicConnectionProtocol):
    packet_logger: PacketLogger = None
    _instance_counter: int = 0

    def connection_made(self, transport):
        InstrumentedServerProtocol._instance_counter += 1
        self._server_flow_id = f"srv_{InstrumentedServerProtocol._instance_counter}"
        super().connection_made(transport)

    def datagram_received(self, data, addr):
        super().datagram_received(data, addr)
