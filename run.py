#!/usr/bin/env python3
"""
QSentinel

Usage:
    python run.py                                            # Generate 50 flows per class (default)
    python run.py --flows-per-class 100                      # Generate 100 flows per class
    python run.py --classes benign_web malicious_c2_beacon   # Specific classes only
"""

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from generate_certs import generate_self_signed_cert
from packet_logger import PacketLogger
from server import run_server
from features.extractor import FeatureExtractor
from generators import ALL_GENERATORS


GENERATOR_MAP = {gen.label: gen for gen in ALL_GENERATORS}


async def run_generation(
    flows_per_class: int,
    selected_classes: list[str] | None = None,
    output_prefix: str = "quic_dataset",
):
    """Run the full pipeline: server → generators → features → export."""

    if not os.path.exists(config.CERT_FILE):
        print("[SETUP] Generating TLS certificates...")
        generate_self_signed_cert()

    logger = PacketLogger()

    print("[SETUP] Starting QUIC server...")
    server_ready = asyncio.Event()
    server_task = asyncio.create_task(run_server(ready_event=server_ready))
    await server_ready.wait()
    print("[SETUP] Server ready.\n")

    if selected_classes:
        generators = []
        for cls_name in selected_classes:
            if cls_name not in GENERATOR_MAP:
                print(f"[!] Unknown class '{cls_name}'. Available: {list(GENERATOR_MAP.keys())}")
                continue
            generators.append(GENERATOR_MAP[cls_name])
    else:
        generators = ALL_GENERATORS

    t_start = time.time()
    for gen_cls in generators:
        gen = gen_cls(logger)
        print(f"\n{'='*60}")
        print(f"  Generating: {gen.description}  ({gen.label})")
        print(f"  Flows: {flows_per_class}")
        print(f"{'='*60}")
        await gen.generate(count=flows_per_class)

    elapsed = time.time() - t_start
    print(f"\n[DONE] Traffic generation completed in {elapsed:.1f}s")
    print(logger.summary())

    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    print("\n[FEATURES] Extracting features from packet logs...")
    extractor = FeatureExtractor()
    df = extractor.build_dataset(logger.all_flows())

    if df.empty:
        print("[!] No valid flows captured. Check for errors above.")
        return

    paths = extractor.export(df, prefix=output_prefix)

    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total flows:    {len(df)}")
    print(f"  Total features: {len([c for c in df.columns if c not in ('label', 'traffic_type', 'flow_id')])}")
    print(f"  Classes:")
    for label, count in df["label"].value_counts().items():
        print(f"    {label:30s} {count:5d} flows")
    print(f"\n  Files:")
    for fmt, path in paths.items():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"    {fmt.upper():10s} {path}  ({size_mb:.2f} MB)")
    print(f"{'='*60}\n")

    print("[SAMPLE] First 3 rows (transposed):")
    numeric_cols = [c for c in df.columns if c not in ("label", "traffic_type", "flow_id")]
    print(df[["flow_id", "label"] + numeric_cols[:10]].head(3).to_string(index=False))
    print("  ... (see full dataset in output files)\n")


def main():
    parser = argparse.ArgumentParser(
        description="QUIC IDS Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available classes:\n  " + "\n  ".join(GENERATOR_MAP.keys()),
    )
    parser.add_argument(
        "--flows-per-class", "-n",
        type=int,
        default=config.DEFAULT_FLOWS_PER_CLASS,
        help=f"Number of flows to generate per class (default: {config.DEFAULT_FLOWS_PER_CLASS})",
    )
    parser.add_argument(
        "--classes", "-c",
        nargs="+",
        default=None,
        help="Specific traffic classes to generate (default: all)",
    )
    parser.add_argument(
        "--output-prefix", "-o",
        default="quic_dataset",
        help="Output filename prefix (default: quic_dataset)",
    )

    args = parser.parse_args()

    asyncio.run(
        run_generation(
            flows_per_class=args.flows_per_class,
            selected_classes=args.classes,
            output_prefix=args.output_prefix,
        )
    )


if __name__ == "__main__":
    main()
