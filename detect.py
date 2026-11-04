#!/usr/bin/env python3

import argparse
import os
import sys
import time
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live.packet_reader import read_live, read_pcap, read_simulated
from live.flow_tracker import FlowTracker
from live.inference import InferenceEngine
from live.alert_system import AlertSystem, C


BANNER = f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   ██████╗ ██╗   ██╗██╗ ██████╗     ██╗██████╗ ███████╗   ║
║  ██╔═══██╗██║   ██║██║██╔════╝     ██║██╔══██╗██╔════╝   ║ 
║  ██║   ██║██║   ██║██║██║          ██║██║  ██║███████╗   ║
║  ██║▄▄ ██║██║   ██║██║██║          ██║██║  ██║╚════██║   ║
║  ╚██████╔╝╚██████╔╝██║╚██████╗     ██║██████╔╝███████║   ║
║   ╚══▀▀═╝  ╚═════╝ ╚═╝ ╚═════╝     ╚═╝╚═════╝ ╚══════╝   ║
║                                                          ║
║  Intrusion Detection System v5.0                         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝{C.RESET}
"""


def run_detection(packet_source, tracker: FlowTracker,
                  engine: InferenceEngine, alerts: AlertSystem,
                  check_interval: float = 2.0):

    last_timeout_check = time.time()
    last_status_print = time.time()
    pkt_count = 0

    try:
        for pkt in packet_source:
            if pkt is not None:
                pkt_count += 1

                result = tracker.ingest(pkt)
                if result:
                    flow, features = result
                    verdict = engine.analyze(flow, features)
                    alerts.process(verdict)

            now = time.time()
            if now - last_timeout_check > check_interval:
                expired = tracker.check_timeouts(now)
                for flow, features in expired:
                    verdict = engine.analyze(flow, features)
                    alerts.process(verdict)
                last_timeout_check = now

            if now - last_status_print > 2.0:
                v = engine.stats.get("verdicts", {})
                safe = v.get("SAFE", 0)
                sus = v.get("SUSPICIOUS", 0)
                mal = v.get("MALICIOUS", 0)
                active = tracker.stats.get("active", 0)
                done = tracker.stats.get("completed", 0)
                print(
                    f"\r  {C.DIM}⟳ pkts={pkt_count}  "
                    f"active={active}  done={done}  "
                    f"{C.GREEN}SAFE={safe}{C.DIM}  "
                    f"{C.YELLOW}SUS={sus}{C.DIM}  "
                    f"{C.RED}MAL={mal}{C.RESET}",
                    end="", flush=True,
                )
                last_status_print = now

        print()
        remaining = tracker.flush_all()
        for flow, features in remaining:
            verdict = engine.analyze(flow, features)
            alerts.process(verdict)

    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}[INTERRUPTED]{C.RESET} Flushing active flows...")
        remaining = tracker.flush_all()
        for flow, features in remaining:
            verdict = engine.analyze(flow, features)
            alerts.process(verdict)

    print(f"\n{C.DIM}  Packets processed: {pkt_count}{C.RESET}")
    print(f"{C.DIM}  Tracker: {tracker.stats}{C.RESET}")
    print(f"{C.DIM}  Engine:  {engine.stats}{C.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="QUIC IDS — Real-time Intrusion Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Live capture via tshark")
    mode.add_argument("--pcap", type=str, help="Analyze a PCAP file")
    mode.add_argument("--simulate", type=int, metavar="N", help="Simulate N flows (no tshark)")

    parser.add_argument("--interface", "-i", default="any", help="Network interface (default: any)")
    parser.add_argument("--models-dir", default="models_v6", help="Model directory (default: models_v5)")
    parser.add_argument("--timeout", type=float, default=30.0, help="Flow timeout seconds (default: 30)")
    parser.add_argument("--min-packets", type=int, default=10, help="Min packets per flow (default: 10)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show SAFE flows too")
    parser.add_argument("--alert-level", default="SUSPICIOUS",
                        choices=["SAFE", "SUSPICIOUS", "MALICIOUS", "UNKNOWN"],
                        help="Minimum alert level (default: SUSPICIOUS)")
    parser.add_argument("--log-dir", default="ids_logs", help="Log directory (default: ids_logs)")
    parser.add_argument("--no-quic-filter", action="store_true", help="Don't filter QUIC-only")
    parser.add_argument("--debug-ml", action="store_true", help="Print advanced ML feature analysis")

    args = parser.parse_args()

    print(BANNER)

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.models_dir)
    required = ["classifier_v5.pt", "iforest_v5.joblib", "scaler_v5.joblib", "label_encoder_v5.joblib"]
    missing = [name for name in required if not os.path.exists(os.path.join(models_dir, name))]
    if missing:
        print(f"{C.RED}[ERROR]{C.RESET} Models not found in {models_dir}/")
        print(f"  Missing: {', '.join(missing)}")
        print(f"  Run 'python retrain_v6.py' first to train the models.")
        sys.exit(1)

    print(f"{C.BOLD}[INIT]{C.RESET} Loading models from {models_dir}/")
    engine = InferenceEngine(models_dir, debug=args.debug_ml)

    tracker = FlowTracker(
        timeout=args.timeout,
        min_packets=args.min_packets,
    )

    alert_system = AlertSystem(
        log_dir=args.log_dir,
        verbose=args.verbose,
        min_alert_level=args.alert_level,
    )

    quic_only = not args.no_quic_filter

    if args.live:
        print(f"\n{C.BOLD}[MODE]{C.RESET} Live capture on interface '{args.interface}'")
        print(f"{C.DIM}  Press Ctrl+C to stop{C.RESET}\n")
        source = read_live(interface=args.interface, quic_only=quic_only)

    elif args.pcap:
        pcap_path = args.pcap
        if not os.path.isabs(pcap_path):
            pcap_path = os.path.join(os.getcwd(), pcap_path)
        print(f"\n{C.BOLD}[MODE]{C.RESET} Analyzing PCAP: {pcap_path}\n")
        source = read_pcap(pcap_path, quic_only=quic_only)

    elif args.simulate:
        print(f"\n{C.BOLD}[MODE]{C.RESET} Simulating {args.simulate} flows\n")
        source = read_simulated(n_flows=args.simulate)

    t0 = time.time()
    run_detection(source, tracker, engine, alert_system)
    elapsed = time.time() - t0

    alert_system.print_summary()
    print(f"  {C.DIM}Total time: {elapsed:.1f}s{C.RESET}")

    alert_system.close()


if __name__ == "__main__":
    main()
