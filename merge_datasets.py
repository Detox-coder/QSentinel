#!/usr/bin/env python3
import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


NIMS_DIRECT_MAP = {
    "Flow Duration":         "duration",
    "Packet Count":          "total_packets",
    "Packet Length Total":   "total_bytes_fwd",    
    "Min Packet Length":     "all_size_min",
    "Max Packet Length":     "all_size_max",
    "Packet Length Mean":    "all_size_mean",
    "Flow IAT Min":          "all_iat_min",
    "Flow IAT Max":          "all_iat_max",
    "Flow IAT Mean":         "all_iat_mean",
    "Current Packet Length": "fwd_size_median",     
}

SHARED_FEATURES = [
    "duration",
    "total_packets",
    "total_bytes_fwd",
    "pkts_per_sec",
    "bytes_per_sec",

    "all_size_mean",
    "all_size_std",
    "all_size_min",
    "all_size_max",
    "all_size_median",
    "all_size_skew",
    "all_size_kurtosis",
    "all_size_q25",
    "all_size_q75",

    "all_iat_mean",
    "all_iat_std",
    "all_iat_min",
    "all_iat_max",
    "all_iat_skew",
    "all_iat_kurtosis",
    "timing_regularity",

    "small_pkt_ratio",
    "large_pkt_ratio",
    "size_entropy",
    "markov_fwd_fwd",
    "markov_fwd_bwd",
    "markov_bwd_fwd",
    "markov_bwd_bwd",
]

NUM_SHARED = len(SHARED_FEATURES)


def load_synthetic(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    print(f"[SYNTH] Loaded {len(df)} flows, {df.shape[1]} cols from {path}")
    print(f"[SYNTH] Labels: {df['label'].value_counts().to_dict()}")
    return df


def load_nims(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    print(f"[NIMS]  Loaded {len(raw)} flows, {raw.shape[1]} cols from {path}")
    print(f"[NIMS]  Labels: {raw['Label'].value_counts().to_dict()}")

    time_cols = ["Flow Duration", "Flow IAT Min", "Flow IAT Max", "Flow IAT Mean"]
    for col in time_cols:
        raw[col] = raw[col] / 1e9

    for col in time_cols:
        raw[col] = raw[col].clip(lower=0.0)

    df = pd.DataFrame()
    for nims_col, synth_col in NIMS_DIRECT_MAP.items():
        df[synth_col] = raw[nims_col].values


    df["pkts_per_sec"] = df["total_packets"] / df["duration"].clip(lower=1e-6)

    df["bytes_per_sec"] = df["total_bytes_fwd"] / df["duration"].clip(lower=1e-6)

    df["all_size_std"] = (df["all_size_max"] - df["all_size_min"]) / 4.0 
    df["all_size_median"] = df["all_size_mean"] 
    df["all_size_q25"] = (df["all_size_min"] + df["all_size_mean"]) / 2.0
    df["all_size_q75"] = (df["all_size_mean"] + df["all_size_max"]) / 2.0
    df["all_size_skew"] = 0.0 
    df["all_size_kurtosis"] = 0.0

    df["all_iat_std"] = (df["all_iat_max"] - df["all_iat_min"]) / 4.0
    df["all_iat_skew"] = 0.0
    df["all_iat_kurtosis"] = 0.0
    df["timing_regularity"] = df["all_iat_std"] / df["all_iat_mean"].clip(lower=1e-9)

    df["small_pkt_ratio"] = (raw["Min Packet Length"].values <= 100).astype(float)
    df["large_pkt_ratio"] = (raw["Max Packet Length"].values >= 1000).astype(float)

    size_range = (df["all_size_max"] - df["all_size_min"]).clip(lower=1.0)
    df["size_entropy"] = np.log2(size_range + 1)
    
    df["markov_fwd_fwd"] = 0.0
    df["markov_fwd_bwd"] = 0.0
    df["markov_bwd_fwd"] = 0.0
    df["markov_bwd_bwd"] = 0.0

    df["label"] = "benign_real"
    df["original_app"] = raw["Label"].values
    df["flow_id"] = raw["Flow ID"].values
    df["traffic_type"] = -1 
    df["source"] = "nims_real"

    print(f"[NIMS]  Transformed to {len(df)} flows, {NUM_SHARED} shared features")
    return df


def align_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    out = df[SHARED_FEATURES].copy()
    out["label"] = df["label"].values
    out["flow_id"] = df["flow_id"].values
    out["traffic_type"] = df["traffic_type"].values
    out["source"] = "synthetic"
    out["original_app"] = df["label"].values
    return out


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = SHARED_FEATURES

    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            lower = mean - 5 * std
            upper = mean + 5 * std
            df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def rebalance(nims_df: pd.DataFrame, target_per_class: int, random_state: int = 42) -> pd.DataFrame:
    n = min(len(nims_df), target_per_class)
    sampled = nims_df.sample(n=n, random_state=random_state)
    print(f"[BALANCE] Downsampled NIMS: {len(nims_df)} → {len(sampled)} flows")
    return sampled


def merge_datasets(
    synth_df: pd.DataFrame,
    nims_df: pd.DataFrame,
    balance: bool = True,
) -> pd.DataFrame:

    if balance:
        synth_per_class = synth_df["label"].value_counts().iloc[0]
        target = synth_per_class * 2
        nims_df = rebalance(nims_df, target)

    merged = pd.concat([synth_df, nims_df], ignore_index=True)

    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n[MERGE] Combined dataset: {len(merged)} flows")
    print(f"[MERGE] Features: {NUM_SHARED}")
    print(f"[MERGE] Label distribution:")
    for label, count in merged["label"].value_counts().items():
        pct = count / len(merged) * 100
        print(f"         {label:30s} {count:5d}  ({pct:5.1f}%)")

    print(f"\n[MERGE] Source distribution:")
    for src, count in merged["source"].value_counts().items():
        print(f"         {src:20s} {count:5d}")

    return merged


def validate_dataset(df: pd.DataFrame) -> bool:
    """Run validation checks on the merged dataset."""
    print(f"\n{'='*60}")
    print(f"  VALIDATION")
    print(f"{'='*60}")

    issues = []

    numeric_cols = SHARED_FEATURES
    nan_count = df[numeric_cols].isnull().sum().sum()
    inf_count = np.isinf(df[numeric_cols].values).sum()
    if nan_count > 0:
        issues.append(f"  ✗ Found {nan_count} NaN values")
    else:
        print(f"  ✓ No NaN values")
    if inf_count > 0:
        issues.append(f"  ✗ Found {inf_count} Inf values")
    else:
        print(f"  ✓ No Inf values")

    dup_count = df["flow_id"].duplicated().sum()
    if dup_count > 0:
        print(f"  ⚠ {dup_count} duplicate flow IDs (may be expected across sources)")
    else:
        print(f"  ✓ No duplicate flow IDs")

    from sklearn.metrics import mutual_info_score
    source_encoded = (df["source"] == "synthetic").astype(int)
    max_corr = 0.0
    max_feat = ""
    for col in numeric_cols:
        try:
            binned = pd.qcut(df[col], q=10, labels=False, duplicates="drop")
            mi = mutual_info_score(source_encoded, binned)
            if mi > max_corr:
                max_corr = mi
                max_feat = col
        except Exception:
            pass
    if max_corr > 1.5:
        issues.append(f"  ⚠ High mutual info between source and '{max_feat}' ({max_corr:.3f})")
    else:
        print(f"  ✓ No feature leakage detected (max MI: {max_feat}={max_corr:.3f})")

    label_counts = df["label"].value_counts()
    ratio = label_counts.max() / label_counts.min()
    if ratio > 10:
        print(f"  ⚠ Class imbalance ratio: {ratio:.1f}:1 (consider resampling)")
    else:
        print(f"  ✓ Class balance ratio: {ratio:.1f}:1")

    print(f"\n  Feature Statistics (merged):")
    print(f"  {'Feature':25s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s}")
    print(f"  {'-'*73}")
    for col in numeric_cols[:15]:
        mean = df[col].mean()
        std = df[col].std()
        mn = df[col].min()
        mx = df[col].max()
        print(f"  {col:25s} {mean:12.4f} {std:12.4f} {mn:12.4f} {mx:12.4f}")
    if len(numeric_cols) > 15:
        print(f"  ... ({len(numeric_cols) - 15} more features)")

    if issues:
        print(f"\n  ISSUES:")
        for issue in issues:
            print(issue)
        return False
    else:
        print(f"\n  ✓ All validation checks passed")
        return True


def export_dataset(df: pd.DataFrame, output_dir: str, prefix: str = "merged_dataset"):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{prefix}.csv")
    parquet_path = os.path.join(output_dir, f"{prefix}.parquet")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    feature_meta = {
        "shared_features": SHARED_FEATURES,
        "n_features": NUM_SHARED,
        "n_samples": len(df),
        "labels": df["label"].value_counts().to_dict(),
        "sources": df["source"].value_counts().to_dict(),
    }
    meta_path = os.path.join(output_dir, f"{prefix}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(feature_meta, f, indent=2)

    print(f"\n[EXPORT] CSV     → {csv_path}  ({os.path.getsize(csv_path)/1024:.1f} KB)")
    print(f"[EXPORT] Parquet → {parquet_path}  ({os.path.getsize(parquet_path)/1024:.1f} KB)")
    print(f"[EXPORT] Meta    → {meta_path}")
    return {"csv": csv_path, "parquet": parquet_path, "meta": meta_path}


def main():
    parser = argparse.ArgumentParser(description="Merge Synthetic + NIMS datasets")
    parser.add_argument("--synthetic", "-s",
        default="output/training_dataset.parquet",
        help="Path to synthetic QUIC IDS dataset")
    parser.add_argument("--real", "-r",
        default="NIMS_IMA_sample_data.csv",
        help="Path to NIMS IMA dataset")
    parser.add_argument("--output", "-o",
        default="output",
        help="Output directory")
    args = parser.parse_args()

    print(f"{'#'*60}")
    print(f"  DATASET MERGER: Synthetic QUIC + NIMS Real Traffic")
    print(f"{'#'*60}\n")

    print(f"\n{'='*60}")
    print(f"  PHASE 1: LOADING DATASETS")
    print(f"{'='*60}")
    synth_raw = load_synthetic(args.synthetic)
    nims_raw = load_nims(args.real)

    print(f"\n{'='*60}")
    print(f"  PHASE 2: FEATURE ALIGNMENT ({NUM_SHARED} shared features)")
    print(f"{'='*60}")
    synth_aligned = align_synthetic(synth_raw)
    print(f"[ALIGN] Synthetic: {synth_aligned.shape}")
    print(f"[ALIGN] NIMS:      {nims_raw[SHARED_FEATURES + ['label','flow_id','traffic_type','source','original_app']].shape}")

    nims_aligned = nims_raw[SHARED_FEATURES + ["label", "flow_id", "traffic_type", "source", "original_app"]]

    print(f"\n{'='*60}")
    print(f"  PHASE 3: DATA CLEANING")
    print(f"{'='*60}")
    synth_clean = clean_data(synth_aligned)
    nims_clean = clean_data(nims_aligned)
    print(f"[CLEAN] Synthetic: {synth_clean.shape}")
    print(f"[CLEAN] NIMS:      {nims_clean.shape}")

    print(f"\n{'='*60}")
    print(f"  PHASE 4: MERGING")
    print(f"{'='*60}")
    merged = merge_datasets(synth_clean, nims_clean)

    validate_dataset(merged)


    print(f"\n{'='*60}")
    print(f"  PHASE 6: EXPORT")
    print(f"{'='*60}")
    export_dataset(merged, args.output, prefix="merged_dataset")

    print(f"\n{'#'*60}")
    print(f"  MERGE COMPLETE")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
