"""
Alert system: terminal display + file logging.

Color-coded terminal output with structured JSON log files.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import TextIO, Optional

from live.inference import FlowVerdict


# ── ANSI colors ───────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BG_RED  = "\033[41m"
    BG_YEL  = "\033[43m"
    BG_GRN  = "\033[42m"


VERDICT_STYLE = {
    "SAFE":       (C.GREEN,  "✓"),
    "SUSPICIOUS": (C.YELLOW, "⚠"),
    "MALICIOUS":  (C.RED,    "✗"),
    "UNKNOWN":    (C.MAGENTA, "?"),
}

VERDICT_PRIORITY = {"SAFE": 0, "SUSPICIOUS": 1, "UNKNOWN": 2, "MALICIOUS": 3}


class AlertSystem:
    """
    Displays alerts in terminal and logs to JSON file.
    """

    def __init__(self, log_dir: str = "ids_logs", verbose: bool = False,
                 min_alert_level: str = "SUSPICIOUS"):
        self.log_dir = log_dir
        self.verbose = verbose
        self.min_priority = VERDICT_PRIORITY.get(min_alert_level, 1)
        self._alert_count = 0
        self._total = 0

        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = os.path.join(log_dir, f"ids_log_{ts}.jsonl")
        self._log_file: Optional[TextIO] = open(self._log_path, "a")

        self._printed_header = False

    def process(self, verdict: FlowVerdict):
        """Handle a verdict: log + optionally display."""
        self._total += 1

        # Always log to file
        self._log(verdict)

        priority = VERDICT_PRIORITY.get(verdict.verdict, 0)

        if priority >= self.min_priority:
            self._alert_count += 1
            self._display_alert(verdict)
        elif self.verbose:
            self._display_safe(verdict)

    def _display_alert(self, v: FlowVerdict):
        """Display color-coded alert in terminal."""
        color, icon = VERDICT_STYLE.get(v.verdict, (C.WHITE, "•"))

        # Header bar
        if v.verdict == "MALICIOUS":
            header_bg = C.BG_RED
        elif v.verdict == "SUSPICIOUS":
            header_bg = C.BG_YEL
        else:
            header_bg = ""

        print()
        print(f"{header_bg}{C.BOLD}{color}  {icon} {v.verdict:10s}{C.RESET}"
              f"  {C.BOLD}{v.flow_id}{C.RESET}"
              f"  {C.DIM}{v.src_ip}:{v.src_port} → {v.dst_ip}:{v.dst_port}{C.RESET}")

        print(f"  {C.DIM}├─{C.RESET} Prediction: {color}{v.prediction}{C.RESET}"
              f"  Confidence: {self._conf_color(v.confidence)}{v.confidence:.1%}{C.RESET}")

        print(f"  {C.DIM}├─{C.RESET} Risk Score: {self._risk_color(v.risk_score)}{v.risk_score:.3f}{C.RESET}"
              f"  Anomaly: {v.anomaly_score:.1f}"
              f"  Packets: {v.n_packets}  Duration: {v.duration:.2f}s")

        for exp in v.explanation:
            print(f"  {C.DIM}├─{C.RESET} {C.CYAN}{exp}{C.RESET}")

        if v.top_features:
            feats = ", ".join(f"{f['feature']}({f['importance']:.3f})" for f in v.top_features[:3])
            print(f"  {C.DIM}└─{C.RESET} Top features: {C.DIM}{feats}{C.RESET}")

    def _display_safe(self, v: FlowVerdict):
        """Compact display for safe flows."""
        print(f"  {C.GREEN}✓{C.RESET} {v.flow_id:15s}"
              f"  {v.prediction:25s}"
              f"  conf={v.confidence:.3f}"
              f"  risk={v.risk_score:.3f}"
              f"  {v.src_ip}→{v.dst_ip}")

    def _log(self, v: FlowVerdict):
        """Write JSON log line."""
        if self._log_file and not self._log_file.closed:
            entry = v.to_dict()
            entry["logged_at"] = datetime.now().isoformat()
            self._log_file.write(json.dumps(entry, default=str) + "\n")
            self._log_file.flush()

    @staticmethod
    def _conf_color(conf: float) -> str:
        if conf >= 0.80:
            return C.GREEN
        elif conf >= 0.50:
            return C.YELLOW
        return C.RED

    @staticmethod
    def _risk_color(risk: float) -> str:
        if risk < 0.20:
            return C.GREEN
        elif risk < 0.45:
            return C.YELLOW
        elif risk < 0.70:
            return C.RED
        return C.BG_RED + C.WHITE

    def print_summary(self):
        """Print session summary."""
        print(f"\n{C.BOLD}{'═'*60}{C.RESET}")
        print(f"{C.BOLD}  SESSION SUMMARY{C.RESET}")
        print(f"{'═'*60}")
        print(f"  Total flows analyzed:  {self._total}")
        print(f"  Alerts raised:         {self._alert_count}")
        print(f"  Log file:              {self._log_path}")
        print(f"{'═'*60}\n")

    def close(self):
        if self._log_file and not self._log_file.closed:
            self._log_file.close()

    @property
    def stats(self) -> dict:
        return {"total": self._total, "alerts": self._alert_count, "log": self._log_path}
