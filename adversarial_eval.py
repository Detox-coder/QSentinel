#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
import copy
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from merge_datasets import SHARED_FEATURES, NUM_SHARED
from training.classifier import QUICClassifier
from training.augment import augment_gaussian, augment_scale

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models_v5")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "adversarial_results")


def load_v2_model(device):
    import joblib
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_v5.joblib"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder_v5.joblib"))
    class_names = list(le.classes_)

    cls_ckpt = torch.load(
        os.path.join(MODELS_DIR, "classifier_v5.pt"),
        map_location=device, weights_only=False,
    )
    classifier = QUICClassifier(
        n_features=cls_ckpt["n_features"],
        n_classes=cls_ckpt["n_classes"],
        hidden_dims=(256, 128, 64), dropout=0.45,
    ).to(device)
    classifier.load_state_dict(cls_ckpt["model_state"])
    classifier.eval()

    # Load Isolation Forest anomaly detector
    anomaly_detector = joblib.load(os.path.join(MODELS_DIR, "iforest_v5.joblib"))
    threshold = cls_ckpt.get("anomaly_threshold", 0.5)

    return classifier, anomaly_detector, scaler, le, class_names, threshold


def load_dataset():
    df = pd.read_parquet("output/scaled_merged_dataset.parquet")
    return df


def quick_eval(model, X, y, class_names, device, title=""):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_t).argmax(1).cpu().numpy()
    acc = accuracy_score(y, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        y, preds, average="weighted", zero_division=0,
    )
    present = sorted(set(y) | set(preds))
    present_names = [class_names[i] for i in present]

    _, _, per_f1, _ = precision_recall_fscore_support(
        y, preds, labels=present, average=None, zero_division=0,
    )
    pclass_f1 = {name: f for name, f in zip(present_names, per_f1)}

    fpr = {}
    for idx, name in zip(present, present_names):
        fp = int(((preds == idx) & (y != idx)).sum())
        tn = int(((preds != idx) & (y != idx)).sum())
        fpr[name] = fp / max(fp + tn, 1)

    return {
        "accuracy": acc, "precision": p, "recall": r, "f1": f1,
        "per_class_f1": pclass_f1, "fpr": fpr, "preds": preds,
    }


def print_comparison(baseline: dict, test: dict, title: str):
    print(f"\n  {'Metric':<20s} {'Baseline':>10s} {'Test':>10s} {'Δ':>10s}")
    print(f"  {'-'*52}")
    for key in ["accuracy", "precision", "recall", "f1"]:
        b = baseline[key]
        t = test[key]
        delta = t - b
        arrow = "↓" if delta < -0.01 else ("↑" if delta > 0.01 else "=")
        print(f"  {key:<20s} {b:10.4f} {t:10.4f} {delta:+10.4f} {arrow}")


def train_quick_classifier(X_train, y_train, X_val, y_val, n_features, n_classes,
                           device, epochs=80, dropout=0.4, lr=1e-3):
    model = QUICClassifier(
        n_features=n_features, n_classes=n_classes,
        hidden_dims=(256, 128, 64), dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_v = torch.tensor(y_val, dtype=torch.long).to(device)
            v_loss = criterion(model(X_v), y_v).item()
        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()
    return model

def test_feature_ablation(classifier, scaler, le, class_names, device):
    print(f"\n{'#'*60}")
    print(f"  TEST 1: FEATURE ABLATION STUDY")
    print(f"{'#'*60}")

    df = load_dataset()
    X_raw = np.array(df[SHARED_FEATURES].values, dtype=np.float32)
    y = np.array(le.transform(df["label"]))
    source = np.array(df["source"].to_numpy(), dtype=str)

    X_scaled = scaler.transform(X_raw)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y,
    )

    print(f"\n  Computing permutation importance...")

    print(f"\n  Computing permutation importance (10 repeats)...")
    baseline_acc = accuracy_score(
        y_test,
        classifier(
            torch.tensor(X_test, dtype=torch.float32).to(device)
        ).argmax(1).cpu().numpy(),
    )

    importances = np.zeros(NUM_SHARED)
    importances_std = np.zeros(NUM_SHARED)
    rng_perm = np.random.default_rng(42)

    for feat_idx in range(NUM_SHARED):
        scores = []
        for _ in range(10):
            X_perm = X_test.copy()
            X_perm[:, feat_idx] = rng_perm.permutation(X_perm[:, feat_idx])
            with torch.no_grad():
                preds = classifier(
                    torch.tensor(X_perm, dtype=torch.float32).to(device)
                ).argmax(1).cpu().numpy()
            scores.append(baseline_acc - accuracy_score(y_test, preds))
        importances[feat_idx] = np.mean(scores)
        importances_std[feat_idx] = np.std(scores)
    top_k = 5
    top_indices = np.argsort(importances)[::-1][:top_k]
    top_features = [SHARED_FEATURES[i] for i in top_indices]

    print(f"\n  Top {top_k} Most Important Features:")
    print(f"  {'Rank':<6} {'Feature':30s} {'Importance':>10s} {'Std':>10s}")
    print(f"  {'-'*58}")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:<6d} {SHARED_FEATURES[idx]:30s}"
              f" {importances[idx]:10.4f} {importances_std[idx]:10.4f}")

    print(f"\n  Baseline (all {NUM_SHARED} features):")
    baseline = quick_eval(classifier, X_test, y_test, class_names, device)
    print(f"    Acc={baseline['accuracy']:.4f}  F1={baseline['f1']:.4f}")

    ablation_results = {}
    print(f"\n  Ablation results (retrain without feature):")
    print(f"  {'Removed':30s} {'Acc':>8s} {'F1':>8s} {'ΔAcc':>8s} {'ΔF1':>8s}")
    print(f"  {'-'*58}")

    for feat_name in top_features:
        feat_idx = SHARED_FEATURES.index(feat_name)
        X_train_abl = X_train.copy()
        X_test_abl = X_test.copy()
        X_train_abl[:, feat_idx] = 0.0
        X_test_abl[:, feat_idx] = 0.0

        X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
            X_train_abl, y_train, test_size=0.15, random_state=42, stratify=y_train,
        )
        model_abl = train_quick_classifier(
            X_tr2, y_tr2, X_val2, y_val2,
            n_features=NUM_SHARED, n_classes=len(class_names),
            device=device, epochs=60,
        )
        res = quick_eval(model_abl, X_test_abl, y_test, class_names, device)
        d_acc = res["accuracy"] - baseline["accuracy"]
        d_f1 = res["f1"] - baseline["f1"]
        ablation_results[feat_name] = res
        print(f"  {feat_name:30s} {res['accuracy']:8.4f} {res['f1']:8.4f}"
              f" {d_acc:+8.4f} {d_f1:+8.4f}")

    X_train_abl5 = X_train.copy()
    X_test_abl5 = X_test.copy()
    for feat_name in top_features:
        idx = SHARED_FEATURES.index(feat_name)
        X_train_abl5[:, idx] = 0.0
        X_test_abl5[:, idx] = 0.0

    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
        X_train_abl5, y_train, test_size=0.15, random_state=42, stratify=y_train,
    )
    model_abl5 = train_quick_classifier(
        X_tr2, y_tr2, X_val2, y_val2,
        n_features=NUM_SHARED, n_classes=len(class_names),
        device=device, epochs=80,
    )
    res5 = quick_eval(model_abl5, X_test_abl5, y_test, class_names, device)
    d_acc = res5["accuracy"] - baseline["accuracy"]
    d_f1 = res5["f1"] - baseline["f1"]
    print(f"\n  {'ALL TOP-5 REMOVED':30s} {res5['accuracy']:8.4f} {res5['f1']:8.4f}"
          f" {d_acc:+8.4f} {d_f1:+8.4f}")

    return {
        "top_features": top_features,
        "importances": {SHARED_FEATURES[i]: float(importances[i]) for i in range(len(SHARED_FEATURES))},
        "baseline": {"accuracy": baseline["accuracy"], "f1": baseline["f1"]},
        "ablation_single": {k: {"accuracy": v["accuracy"], "f1": v["f1"]}
                           for k, v in ablation_results.items()},
        "ablation_all5": {"accuracy": res5["accuracy"], "f1": res5["f1"]},
    }


def generate_adversarial_samples(scaler, le, class_names):
    rng = np.random.default_rng(42)
    df = load_dataset()
    X_raw = np.array(df[SHARED_FEATURES].values, dtype=np.float32)
    y = np.array(le.transform(df["label"]))
    labels_str = np.array(df["label"].to_numpy(), dtype=str)

    benign_mask = np.array([l.startswith("benign") for l in labels_str])
    malicious_mask = ~benign_mask
    X_benign = X_raw[benign_mask]
    X_malicious = X_raw[malicious_mask]
    y_benign = y[benign_mask]
    y_malicious = y[malicious_mask]

    adversarial_sets = {}

    n_camo = 200
    camo_X = X_malicious[rng.choice(len(X_malicious), n_camo)].copy()
    benign_stats = X_benign.mean(axis=0)
    benign_stds = X_benign.std(axis=0).clip(min=1e-6)

    timing_feats = [SHARED_FEATURES.index(f) for f in SHARED_FEATURES
                    if "iat" in f or "timing" in f or f == "duration"]
    for idx in timing_feats:
        camo_X[:, idx] = benign_stats[idx] + rng.normal(0, benign_stds[idx] * 0.5, n_camo)

    camo_y = le.transform(["malicious_tunnel"] * n_camo)
    adversarial_sets["camouflage"] = {
        "X": scaler.transform(camo_X), "y": camo_y,
        "desc": "Malicious with benign-like timing features",
    }

    n_noisy = 200
    noisy_X = augment_gaussian(X_malicious, n_noisy, noise_scale=0.30, rng=rng)
    noisy_y = y_malicious[rng.choice(len(y_malicious), n_noisy)]
    adversarial_sets["noisy_attack"] = {
        "X": scaler.transform(noisy_X), "y": noisy_y,
        "desc": "Malicious with 30% Gaussian noise",
    }

    n_mixed = 200
    blend_ratios = rng.uniform(0.3, 0.7, size=(n_mixed, 1))
    idx_b = rng.choice(len(X_benign), n_mixed)
    idx_m = rng.choice(len(X_malicious), n_mixed)
    mixed_X = X_benign[idx_b] * blend_ratios + X_malicious[idx_m] * (1 - blend_ratios)
    mixed_y = y_malicious[rng.choice(len(y_malicious), n_mixed)]
    adversarial_sets["mixed_blend"] = {
        "X": scaler.transform(mixed_X), "y": mixed_y,
        "desc": "30-70% blend of benign + malicious features",
    }

    n_evasion = 200
    evasion_X = X_malicious[rng.choice(len(X_malicious), n_evasion)].copy()
    benign_centroid = X_benign.mean(axis=0)
    evasion_X = evasion_X * 0.6 + benign_centroid * 0.4
    evasion_y = y_malicious[rng.choice(len(y_malicious), n_evasion)]
    adversarial_sets["evasion_shift"] = {
        "X": scaler.transform(evasion_X), "y": evasion_y,
        "desc": "Malicious shifted 40% towards benign centroid",
    }

    n_stealth = 200
    stealth_X = X_malicious[rng.choice(len(X_malicious), n_stealth)].copy()
    benign_min = X_benign.min(axis=0)
    benign_max = X_benign.max(axis=0)
    stealth_X = np.clip(stealth_X, benign_min, benign_max)
    stealth_y = y_malicious[rng.choice(len(y_malicious), n_stealth)]
    adversarial_sets["stealth_clamp"] = {
        "X": scaler.transform(stealth_X), "y": stealth_y,
        "desc": "Malicious clamped to benign feature range",
    }

    return adversarial_sets


def test_adversarial(classifier, anomaly_detector, scaler, le, class_names, threshold, device):
    print(f"\n{'#'*60}")
    print(f"  TEST 2: ADVERSARIAL DATA EVALUATION")
    print(f"{'#'*60}")

    df = load_dataset()
    X_raw = np.array(df[SHARED_FEATURES].values, dtype=np.float32)
    y = np.array(le.transform(df["label"]))
    from sklearn.model_selection import train_test_split
    _, X_test_raw, _, y_test = train_test_split(
        X_raw, y, test_size=0.3, random_state=42, stratify=y,
    )
    X_test = scaler.transform(X_test_raw)
    baseline = quick_eval(classifier, X_test, y_test, class_names, device)
    print(f"\n  Baseline (clean test): Acc={baseline['accuracy']:.4f}  F1={baseline['f1']:.4f}")

    adv_sets = generate_adversarial_samples(scaler, le, class_names)
    results = {}

    print(f"\n  {'Attack Type':25s} {'Acc':>7s} {'F1':>7s} {'ΔAcc':>8s} {'ΔF1':>8s}"
          f" {'FPR↑':>7s} {'IF_Det':>8s}")
    print(f"  {'-'*70}")

    for name, data in adv_sets.items():
        X_adv = data["X"].astype(np.float32)
        y_adv = data["y"]

        res = quick_eval(classifier, X_adv, y_adv, class_names, device)
        d_acc = res["accuracy"] - baseline["accuracy"]
        d_f1 = res["f1"] - baseline["f1"]

        # IForest anomaly detection (numpy-based)
        scores = anomaly_detector.anomaly_score(X_adv)
        iforest_det = float(np.mean(scores > threshold))

        benign_class_ids = {i for i, n in enumerate(class_names) if n.startswith("benign")}
        preds = res["preds"]
        false_benign = np.array([p in benign_class_ids for p in preds]).mean()

        print(f"  {name:25s} {res['accuracy']:7.4f} {res['f1']:7.4f}"
              f" {d_acc:+8.4f} {d_f1:+8.4f}"
              f" {false_benign:7.4f} {iforest_det:8.4f}")

        results[name] = {
            "desc": data["desc"],
            "accuracy": res["accuracy"], "f1": res["f1"],
            "delta_acc": d_acc, "delta_f1": d_f1,
            "false_benign_rate": float(false_benign),
            "iforest_detection": iforest_det,
        }

    return results, baseline

def test_cross_domain(classifier, anomaly_detector, scaler, le, class_names, threshold, device):
    print(f"\n{'#'*60}")
    print(f"  TEST 3: CROSS-DOMAIN TESTING (Unseen Distribution)")
    print(f"{'#'*60}")

    rng = np.random.default_rng(99)
    df = load_dataset()
    X_raw = np.array(df[SHARED_FEATURES].values, dtype=np.float32)
    y = np.array(le.transform(df["label"]))
    labels_str = np.array(df["label"].to_numpy(), dtype=str)

    shifts = [
        ("temporal_drift",    0.15, "All features shifted by +15% (temporal drift)"),
        ("bandwidth_change",  0.25, "Packet sizes scaled by ±25% (different bandwidth)"),
        ("heavy_noise",       0.40, "Heavy random noise (40% std)"),
    ]

    results = {}
    n_unseen = 500

    from sklearn.model_selection import train_test_split
    _, X_test_raw, _, y_test = train_test_split(
        X_raw, y, test_size=0.3, random_state=42, stratify=y,
    )
    X_test = scaler.transform(X_test_raw)
    baseline = quick_eval(classifier, X_test, y_test, class_names, device)

    print(f"\n  Baseline (in-domain): Acc={baseline['accuracy']:.4f}  F1={baseline['f1']:.4f}")
    print(f"\n  {'Domain':25s} {'Acc':>7s} {'F1':>7s} {'ΔAcc':>8s} {'ΔF1':>8s}")
    print(f"  {'-'*55}")

    for shift_name, magnitude, desc in shifts:
        idx = rng.choice(len(X_test_raw), n_unseen)
        X_shifted = X_test_raw[idx].copy()
        y_shifted = y_test[idx]

        if shift_name == "temporal_drift":
            X_shifted *= (1 + magnitude)
        elif shift_name == "bandwidth_change":
            scales = rng.uniform(1 - magnitude, 1 + magnitude, size=X_shifted.shape)
            X_shifted *= scales
        elif shift_name == "heavy_noise":
            stds = X_test_raw.std(axis=0, keepdims=True).clip(min=1e-6)
            noise = rng.normal(0, magnitude, size=X_shifted.shape) * stds
            X_shifted += noise

        X_shifted_scaled = scaler.transform(X_shifted)
        res = quick_eval(classifier, X_shifted_scaled, y_shifted, class_names, device)
        d_acc = res["accuracy"] - baseline["accuracy"]
        d_f1 = res["f1"] - baseline["f1"]

        print(f"  {shift_name:25s} {res['accuracy']:7.4f} {res['f1']:7.4f}"
              f" {d_acc:+8.4f} {d_f1:+8.4f}")

        results[shift_name] = {
            "desc": desc, "accuracy": res["accuracy"], "f1": res["f1"],
            "delta_acc": d_acc, "delta_f1": d_f1,
            "per_class_f1": {k: float(v) for k, v in res["per_class_f1"].items()},
        }

    return results


def test_fgsm_attack(classifier, scaler, le, class_names, device):
    print(f"\n{'#'*60}")
    print(f"  TEST 4: FGSM GRADIENT-BASED ATTACK")
    print(f"{'#'*60}")

    df = load_dataset()
    X_raw = np.array(df[SHARED_FEATURES].values, dtype=np.float32)
    y = np.array(le.transform(df["label"]))

    from sklearn.model_selection import train_test_split
    _, X_test_raw, _, y_test = train_test_split(
        X_raw, y, test_size=0.3, random_state=42, stratify=y,
    )
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    baseline = quick_eval(classifier, X_test, y_test, class_names, device)
    print(f"\n  Baseline: Acc={baseline['accuracy']:.4f}  F1={baseline['f1']:.4f}")

    epsilons = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
    criterion = nn.CrossEntropyLoss()
    results = {}

    print(f"\n  {'ε':>8s} {'Acc':>8s} {'F1':>8s} {'ΔAcc':>8s} {'Success%':>10s}")
    print(f"  {'-'*46}")

    for eps in epsilons:
        X_t = torch.tensor(X_test, dtype=torch.float32, requires_grad=True).to(device)
        y_t = torch.tensor(y_test, dtype=torch.long).to(device)

        classifier.eval()
        logits = classifier(X_t)
        loss = criterion(logits, y_t)
        loss.backward()

        grad_sign = X_t.grad.data.sign()
        X_adv = X_t.data + eps * grad_sign
        X_adv_np = X_adv.detach().cpu().numpy()

        res = quick_eval(classifier, X_adv_np, y_test, class_names, device)
        d_acc = res["accuracy"] - baseline["accuracy"]
        success = 1.0 - res["accuracy"]

        print(f"  {eps:8.2f} {res['accuracy']:8.4f} {res['f1']:8.4f}"
              f" {d_acc:+8.4f} {success:9.1%}")

        results[str(eps)] = {
            "accuracy": res["accuracy"], "f1": res["f1"],
            "delta_acc": d_acc, "attack_success": success,
        }

    return results

def weakness_analysis(classifier, scaler, le, class_names, device):
    print(f"\n{'#'*60}")
    print(f"  TEST 5: WEAKNESS ANALYSIS")
    print(f"{'#'*60}")

    df = load_dataset()
    X_raw = np.array(df[SHARED_FEATURES].values, dtype=np.float32)
    labels_str = np.array(df["label"].to_numpy(), dtype=str)

    print(f"\n  5a. Feature Overlap (Bhattacharyya Distance between classes):")
    unique_labels = sorted(df["label"].unique())
    class_data = {}
    for label in unique_labels:
        mask = labels_str == label
        class_data[label] = X_raw[mask]

    print(f"\n  Pairwise distance (lower = more overlap → harder to distinguish):")
    pairs = []
    for i, l1 in enumerate(unique_labels):
        for l2 in unique_labels[i+1:]:
            d1, d2 = class_data[l1], class_data[l2]
            m1, m2 = d1.mean(axis=0), d2.mean(axis=0)
            s1, s2 = d1.std(axis=0).clip(min=1e-6), d2.std(axis=0).clip(min=1e-6)
            pooled_s = np.sqrt((s1**2 + s2**2) / 2)
            fisher = ((m1 - m2)**2 / (pooled_s**2 + 1e-8)).mean()
            pairs.append((l1, l2, fisher))

    pairs.sort(key=lambda x: x[2])
    print(f"  {'Class Pair':55s} {'Fisher Dist':>12s} {'Risk':>8s}")
    print(f"  {'-'*77}")
    for l1, l2, dist in pairs[:10]:
        risk = "HIGH" if dist < 1.0 else ("MEDIUM" if dist < 5.0 else "LOW")
        label = f"{l1[:25]:25s} ↔ {l2[:25]:25s}"
        print(f"  {label:55s} {dist:12.4f} {risk:>8s}")

    print(f"\n  5b. Per-class robustness under noise (30% Gaussian):")
    rng = np.random.default_rng(42)
    y = np.array(le.transform(df["label"]))

    stds = X_raw.std(axis=0, keepdims=True).clip(min=1e-6)
    X_noisy = X_raw + rng.normal(0, 0.30, size=X_raw.shape) * stds
    X_noisy_scaled = scaler.transform(X_noisy)

    classifier.eval()
    with torch.no_grad():
        preds = classifier(
            torch.tensor(X_noisy_scaled, dtype=torch.float32).to(device)
        ).argmax(1).cpu().numpy()

    print(f"  {'Class':30s} {'Acc':>7s} {'Top Confusion':>35s}")
    print(f"  {'-'*74}")

    weaknesses = {}
    for i, name in enumerate(class_names):
        mask = y == i
        if mask.sum() == 0:
            continue
        class_preds = preds[mask]
        class_acc = (class_preds == i).mean()

        confused_with = {}
        for p in class_preds[class_preds != i]:
            confused_with[class_names[p]] = confused_with.get(class_names[p], 0) + 1

        if confused_with:
            top_conf = max(confused_with, key=confused_with.get)
            top_conf_pct = confused_with[top_conf] / mask.sum()
            conf_str = f"→ {top_conf} ({top_conf_pct:.1%})"
        else:
            conf_str = "None"
            top_conf_pct = 0.0

        print(f"  {name:30s} {class_acc:7.4f} {conf_str:>35s}")

        weaknesses[name] = {
            "accuracy_under_noise": float(class_acc),
            "top_confusion": confused_with,
        }

    print(f"\n  5c. Most distinguishing feature per class:")
    for name in class_names:
        if name not in class_data:
            continue
        d = class_data[name]
        rest = X_raw[labels_str != name]

        best_feat, best_ratio = "", 0
        for j, feat in enumerate(SHARED_FEATURES):
            m_class = d[:, j].mean()
            m_rest = rest[:, j].mean()
            s_pool = np.sqrt((d[:, j].std()**2 + rest[:, j].std()**2) / 2 + 1e-8)
            ratio = abs(m_class - m_rest) / (s_pool + 1e-8)
            if ratio > best_ratio:
                best_ratio = ratio
                best_feat = feat

        print(f"  {name:30s} → {best_feat:25s} (Fisher={best_ratio:.2f})")

    return weaknesses, pairs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    print(f"\n{'='*60}")
    print(f"  ADVERSARIAL EVALUATION & ROBUSTNESS TESTING")
    print(f"  QUIC IDS — Breaking the Model")
    print(f"{'='*60}")

    classifier, anomaly_detector, scaler, le, class_names, threshold = load_v2_model(device)
    print(f"  Loaded models: {len(class_names)} classes, threshold={threshold:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    t0 = time.time()
    ablation = test_feature_ablation(classifier, scaler, le, class_names, device)

    adversarial, adv_baseline = test_adversarial(
        classifier, anomaly_detector, scaler, le, class_names, threshold, device,
    )

    cross_domain = test_cross_domain(
        classifier, anomaly_detector, scaler, le, class_names, threshold, device,
    )

    fgsm = test_fgsm_attack(classifier, scaler, le, class_names, device)

    weaknesses, overlap_pairs = weakness_analysis(
        classifier, scaler, le, class_names, device,
    )

    total_time = time.time() - t0

    all_results = {
        "ablation": ablation,
        "adversarial": adversarial,
        "cross_domain": cross_domain,
        "fgsm": fgsm,
        "weaknesses": {k: {"accuracy_under_noise": v["accuracy_under_noise"]}
                       for k, v in weaknesses.items()},
        "overlap_top5": [
            {"pair": f"{p[0]} ↔ {p[1]}", "fisher_dist": p[2]}
            for p in overlap_pairs[:5]
        ],
        "total_time_s": round(total_time, 1),
    }

    results_path = os.path.join(RESULTS_DIR, "adversarial_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'#'*60}")
    print(f"  FINAL VULNERABILITY REPORT")
    print(f"{'#'*60}")

    print(f"\n  1. FEATURE DEPENDENCY:")
    print(f"     Top features: {', '.join(ablation['top_features'])}")
    abl5 = ablation['ablation_all5']
    print(f"     Acc without top-5: {abl5['accuracy']:.4f}"
          f" (Δ={abl5['accuracy'] - ablation['baseline']['accuracy']:+.4f})")

    print(f"\n  2. ADVERSARIAL VULNERABILITY:")
    worst_adv = min(adversarial.items(), key=lambda x: x[1]["accuracy"])
    print(f"     Worst attack: {worst_adv[0]}")
    print(f"       Acc={worst_adv[1]['accuracy']:.4f}"
          f"  Evasion={worst_adv[1]['false_benign_rate']:.1%}")

    print(f"\n  3. DOMAIN SHIFT SENSITIVITY:")
    worst_shift = min(cross_domain.items(), key=lambda x: x[1]["accuracy"])
    print(f"     Worst shift: {worst_shift[0]}")
    print(f"       Acc={worst_shift[1]['accuracy']:.4f}"
          f"  (Δ={worst_shift[1]['delta_acc']:+.4f})")

    print(f"\n  4. GRADIENT ATTACK (FGSM):")
    for eps in ["0.1", "0.5", "1.0"]:
        if eps in fgsm:
            print(f"     ε={eps}: Acc={fgsm[eps]['accuracy']:.4f}"
                  f"  Attack success={fgsm[eps]['attack_success']:.1%}")

    print(f"\n  5. WEAKEST CLASSES (under noise):")
    sorted_weak = sorted(weaknesses.items(), key=lambda x: x[1]["accuracy_under_noise"])
    for name, data in sorted_weak[:3]:
        print(f"     {name:30s}  Acc={data['accuracy_under_noise']:.4f}")

    print(f"\n  6. HIGHEST CONFUSION RISK:")
    for p in overlap_pairs[:3]:
        print(f"     {p[0][:25]} ↔ {p[1][:25]}  (Fisher={p[2]:.4f})")

    print(f"\n  Results saved to: {results_path}")
    print(f"  Total evaluation time: {total_time:.1f}s")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
