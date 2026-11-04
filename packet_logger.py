import time
import threading
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List


@dataclass
class PacketRecord:
    timestamp: float         
    size: int               
    direction: int            
    flow_id: str = ""
    label: str = ""
    traffic_type: int = 0


class PacketLogger:

    def __init__(self):
        self._lock = threading.Lock()
        self._flows: Dict[str, List[PacketRecord]] = defaultdict(list)
        self._labels: Dict[str, str] = {}
        self._traffic_types: Dict[str, int] = {}


    def register_flow(self, flow_id: str, label: str, traffic_type: int) -> None:
        with self._lock:
            self._labels[flow_id] = label
            self._traffic_types[flow_id] = traffic_type

    def record(
        self,
        flow_id: str,
        timestamp: float,
        size: int,
        direction: int,
    ) -> None:
        rec = PacketRecord(
            timestamp=timestamp,
            size=size,
            direction=direction,
            flow_id=flow_id,
            label=self._labels.get(flow_id, "unknown"),
            traffic_type=self._traffic_types.get(flow_id, 0),
        )
        with self._lock:
            self._flows[flow_id].append(rec)


    @property
    def flow_ids(self) -> list:
        with self._lock:
            return list(self._flows.keys())

    def get_flow(self, flow_id: str) -> List[PacketRecord]:
        with self._lock:
            return list(self._flows.get(flow_id, []))

    def all_flows(self) -> Dict[str, List[PacketRecord]]:
        with self._lock:
            return {fid: list(recs) for fid, recs in self._flows.items()}

    def total_packets(self) -> int:
        with self._lock:
            return sum(len(recs) for recs in self._flows.values())

    def total_flows(self) -> int:
        with self._lock:
            return len(self._flows)

    def clear(self) -> None:
        with self._lock:
            self._flows.clear()
            self._labels.clear()
            self._traffic_types.clear()

    def summary(self) -> str:
        with self._lock:
            lines = [f"PacketLogger: {len(self._flows)} flows, "
                     f"{sum(len(r) for r in self._flows.values())} packets"]
            label_counts: Dict[str, int] = defaultdict(int)
            for fid in self._flows:
                lbl = self._labels.get(fid, "unknown")
                label_counts[lbl] += 1
            for lbl, cnt in sorted(label_counts.items()):
                lines.append(f"  {lbl}: {cnt} flows")
            return "\n".join(lines)
