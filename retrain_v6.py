#!/usr/bin/env python3
"""
QSentinel v6 Retraining — Expanded classifier + Isolation Forest anomaly detector.

Key changes from v5:
- Classifier trained on ALL benign classes (original 12 + 15 expanded = 27 benign)
- Classifier trained on ALL attack classes (original 4 + 10 expanded = 14 malicious)
- IForest retrained on expanded benign corpus
- Saves to models_v6/ for before/after comparison
"""
import os, sys, time, json
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.utils import (
    ALL_FEATURES, N_FEATURES, SHARED_FEATURES, BEHAVIORAL_FEATURES,
    simulate_flow_features, add_advanced_features, augment,
    gen_c2_beacon, gen_exfil_burst, gen_slow_exfil, gen_tunnel,
    gen_benign_upload, gen_benign_cloud_sync, gen_benign_video,
)
from merge_datasets import load_synthetic, load_nims
from training.classifier import QUICClassifier
from training.isolation_forest import QUICIsolationForest
from training.adversarial import adversarial_training_step, gaussian_noise, random_feature_mask
from training.benign_generators import generate_all_benign
from training.expanded_generators import (
    EXPANDED_BENIGN_GENERATORS, EXPANDED_ATTACK_GENERATORS, generate_expanded,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models_v6")
rng = np.random.default_rng(42)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42); np.random.seed(42)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: GENERATE ALL TRAINING DATA
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'#'*60}\n  PHASE 1: GENERATING EXPANDED TRAINING DATA\n{'#'*60}")

    # Original wire-realistic generators
    mal_c2 = gen_c2_beacon(200)
    mal_exfil = gen_exfil_burst(200)
    mal_slow = gen_slow_exfil(200)
    mal_tunnel = gen_tunnel(200)
    benign_up = gen_benign_upload(200)
    benign_sync = gen_benign_cloud_sync(200)
    benign_vid = gen_benign_video(200)
    wire_data = pd.concat([mal_c2, mal_exfil, mal_slow, mal_tunnel,
                           benign_up, benign_sync, benign_vid], ignore_index=True)
    wire_data["source"] = "wire_realistic"
    wire_data["flow_id"] = [f"wire_{i}" for i in range(len(wire_data))]

    # Original diverse benign generators (12 classes)
    diverse = generate_all_benign(200, seed=42)
    diverse["source"] = "diverse_generator"
    diverse["flow_id"] = [f"diverse_{i}" for i in range(len(diverse))]

    # NEW: Expanded generators (15 benign + 10 attack)
    expanded = generate_expanded(200, seed=42)
    expanded["source"] = "expanded_generator"
    expanded["flow_id"] = [f"expanded_{i}" for i in range(len(expanded))]

    # Existing datasets
    synth = load_synthetic("output/quic_dataset.csv")
    synth_aligned = synth[SHARED_FEATURES + ["label"]].copy()
    synth_aligned["source"] = "synthetic"
    synth_aligned["flow_id"] = [f"synth_{i}" for i in range(len(synth_aligned))]

    nims = load_nims("NIMS_IMA_sample_data.csv")
    nims_aligned = nims[SHARED_FEATURES + ["label", "flow_id", "source"]].copy()
    nims_sampled = nims_aligned.sample(n=min(800, len(nims_aligned)), random_state=42)

    real = pd.read_csv("output/real_world_dataset.csv")
    real_aligned = real[[c for c in SHARED_FEATURES + ["label", "flow_id", "source"] if c in real.columns]].copy()

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: MERGE ALL DATA
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'#'*60}\n  PHASE 2: MERGING EXPANDED DATASET\n{'#'*60}")

    # Get shared columns for concatenation
    shared_cols = SHARED_FEATURES + ["label", "source", "flow_id"]
    parts = []
    for df in [synth_aligned, nims_sampled, real_aligned,
               wire_data, diverse, expanded]:
        cols = [c for c in shared_cols if c in df.columns]
        parts.append(df[cols])

    merged = pd.concat(parts, ignore_index=True)
    merged[SHARED_FEATURES] = merged[SHARED_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    merged = add_advanced_features(merged)

    # ═══════════════════════════════════════════════════════════
    # PHASE 2b: CONSOLIDATE BENIGN LABELS
    # ═══════════════════════════════════════════════════════════
    # 35 benign classes → 18 distinct behavioral groups.
    # This keeps ALL training data but prevents class fragmentation
    # that caused attacks to be misclassified as similar benign types.
    LABEL_MAP = {
        # Streaming: all unidirectional inbound video patterns
        "benign_youtube_stream": "benign_streaming",
        "benign_adaptive_video": "benign_streaming",
        "benign_video_stream": "benign_streaming",
        "benign_streaming_real": "benign_streaming",
        # "benign_streaming" stays as is
        # Chat/messaging: bidirectional, bursty, small packets
        "benign_discord_chat": "benign_chat",
        "benign_discord_text": "benign_chat",
        "benign_telegram": "benign_chat",
        "benign_telegram_text": "benign_chat",
        # Git operations: negotiation + bulk transfer
        "benign_git_clone": "benign_git_download",
        "benign_github_clone": "benign_git_download",
        "benign_github_pull": "benign_git_download",
        "benign_github_push": "benign_git_upload",
        # Browsing: bidirectional request-response
        "benign_browser_http3": "benign_browsing",
        "benign_http3_browsing": "benign_browsing",
        "benign_web": "benign_browsing",
        "benign_mixed_browse": "benign_browsing",
        "benign_multitab_browse": "benign_browsing",
        "benign_mixed_session": "benign_browsing",
        # Bulk download: large inbound
        "benign_gdrive_download": "benign_bulk_download",
        "benign_sw_update": "benign_bulk_download",
        "benign_package_manager": "benign_bulk_download",
        "benign_docker_pull": "benign_bulk_download",
        "benign_vscode_ext": "benign_bulk_download",
        # Upload: large outbound
        "benign_large_upload": "benign_upload",
        "benign_gdrive_upload": "benign_upload",
        "benign_docker_push": "benign_upload",
        # Sync: bidirectional periodic
        "benign_cloud_sync": "benign_sync",
        "benign_background_sync": "benign_sync",
        # Keep distinct patterns as-is
        # benign_discord_voice — unique: constant-rate small bidir
        # benign_idle_keepalive — unique: very sparse tiny bidir
        # benign_developer — unique: multi-phase mixed
        # benign_cicd — unique: API + artifact transfer
        # benign_cloud_shell — unique: interactive terminal
        # benign_real — unique: real-world capture
    }
    before_labels = merged["label"].nunique()
    merged["label"] = merged["label"].map(lambda x: LABEL_MAP.get(x, x))
    after_labels = merged["label"].nunique()
    print(f"\n  Label consolidation: {before_labels} → {after_labels} classes")

    le = LabelEncoder()
    le.fit(merged["label"])
    class_names = list(le.classes_)
    n_classes = len(class_names)

    X_all = merged[ALL_FEATURES].values.astype(np.float32)
    y_all = le.transform(merged["label"])

    print(f"  Total samples: {len(X_all)}, Classes: {n_classes}")
    benign_classes = [n for n in class_names if n.startswith("benign")]
    mal_classes = [n for n in class_names if n.startswith("malicious")]
    print(f"  Benign classes: {len(benign_classes)}, Malicious classes: {len(mal_classes)}")
    for i, name in enumerate(class_names):
        print(f"    {name:40s} {(y_all==i).sum():5d}")

    # Augment to balanced
    X_aug, y_aug = augment(X_all, y_all, 400, rng, class_names)
    print(f"\n  After augmentation: {len(X_aug)} samples")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_aug, y_aug, test_size=0.25, random_state=42, stratify=y_aug)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Scaler with winsorization
    X_train_w = X_train.copy()
    for col in range(X_train_w.shape[1]):
        lo = np.percentile(X_train_w[:, col], 1)
        hi = np.percentile(X_train_w[:, col], 99)
        X_train_w[:, col] = np.clip(X_train_w[:, col], lo, hi)
    scaler = RobustScaler(quantile_range=(5, 95))
    X_train_s = np.clip(scaler.fit_transform(X_train_w), -5, 5).astype(np.float32)
    X_val_s = np.clip(scaler.transform(np.clip(X_val, -1e7, 1e7)), -5, 5).astype(np.float32)
    X_test_s = np.clip(scaler.transform(np.clip(X_test, -1e7, 1e7)), -5, 5).astype(np.float32)

    bs = 64
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train_s), torch.tensor(y_train, dtype=torch.long)), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val_s), torch.tensor(y_val, dtype=torch.long)), batch_size=bs)

    benign_ids = {i for i, n in enumerate(class_names) if n.startswith("benign")}
    benign_mask = np.array([y in benign_ids for y in y_train])
    X_benign = torch.tensor(X_train_s[benign_mask], dtype=torch.float32)
    benign_loader = DataLoader(TensorDataset(X_benign, X_benign), batch_size=bs, shuffle=True)

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: TRAIN CLASSIFIER
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'#'*60}\n  PHASE 3: TRAINING CLASSIFIER ({n_classes} classes)\n{'#'*60}")

    model = QUICClassifier(N_FEATURES, n_classes, (256, 128, 64), dropout=0.45).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=8e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
    best_vl, best_state, no_imp = float("inf"), None, 0

    for ep in range(1, 151):
        model.train()
        tl, tc, tt = 0, 0, 0
        eps_adv = 0.05 + 0.15 * min(ep/75, 1.0)
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            if ep > 10:
                feat_mask = torch.bernoulli(torch.ones_like(Xb) * 0.85)
                Xb = Xb * feat_mask
            if ep <= 20:
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                tl += loss.item()*len(Xb); tc += (model(Xb).argmax(1)==yb).sum().item()
            else:
                stats = adversarial_training_step(model, Xb, yb, criterion, optimizer,
                                                  epsilon=eps_adv, noise_std=0.05+0.1*(ep/150))
                tl += stats["total_loss"]*stats["total"]; tc += stats["correct"]
            tt += len(Xb)
        scheduler.step()
        model.eval()
        vl, vc, vt = 0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                vl += criterion(logits, yb).item()*len(Xb)
                vc += (logits.argmax(1)==yb).sum().item(); vt += len(Xb)
        vl /= max(vt, 1); va = vc/max(vt, 1)
        if ep % 25 == 0 or ep == 1:
            print(f"  {ep:4d} TrnA={tc/max(tt,1):.4f} ValA={va:.4f} ValL={vl:.4f}")
        if vl < best_vl:
            best_vl = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 30: print(f"  Early stop at {ep}"); break
    if best_state: model.load_state_dict(best_state); model.to(device)

    # Temperature calibration
    from training.calibration import TemperatureCalibrator
    calibrator = TemperatureCalibrator()
    temperature = calibrator.fit(model, val_loader, device)
    print(f"  Temperature: {temperature:.4f}")



    # ═══════════════════════════════════════════════════════════
    # PHASE 5: TRAIN ISOLATION FOREST
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'#'*60}\n  PHASE 4: TRAINING ISOLATION FOREST\n{'#'*60}")
    iforest = QUICIsolationForest(n_estimators=200, contamination=0.01, max_features=0.8)
    iforest.fit(X_train_s[benign_mask])
    print(f"  IForest trained on {benign_mask.sum()} benign samples")

    # Calibrate anomaly threshold on VALIDATION benign data
    benign_val_mask = np.array([y in benign_ids for y in y_val])
    iforest_val_scores = iforest.anomaly_score(X_val_s[benign_val_mask])
    threshold = float(np.percentile(iforest_val_scores, 97))
    print(f"  IForest Threshold (val P97): {threshold:.4f}")
    print(f"  Val benign scores: mean={np.mean(iforest_val_scores):.2f} std={np.std(iforest_val_scores):.2f} max={np.max(iforest_val_scores):.2f}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 6: EVALUATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'#'*60}\n  PHASE 5: EVALUATION\n{'#'*60}")
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test_s).to(device)).argmax(1).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    present = sorted(set(y_test) | set(preds))
    print(f"\n  Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, labels=present,
          target_names=[class_names[i] for i in present], digits=4, zero_division=0))

    ben_mask_test = np.array([y in benign_ids for y in y_test])
    ben_as_mal = sum(1 for i, p in enumerate(preds) if ben_mask_test[i] and p not in benign_ids)
    print(f"  Benign misclassified as malicious: {ben_as_mal}/{ben_mask_test.sum()}")

    # ═══════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════
    import joblib
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_v5.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder_v5.joblib"))
    joblib.dump(iforest, os.path.join(MODELS_DIR, "iforest_v5.joblib"))

    torch.save({"model_state": model.state_dict(), "n_features": N_FEATURES,
                "n_classes": n_classes, "class_names": class_names,
                "features": ALL_FEATURES, "arch": "ffnn", "temperature": temperature,
                "anomaly_threshold": threshold},
               os.path.join(MODELS_DIR, "classifier_v5.pt"))

    summary = {
        "accuracy": acc, "benign_fp": ben_as_mal, "threshold": threshold,
        "n_classes": n_classes, "classes": class_names, "features": ALL_FEATURES,
        "temperature": temperature, "ece_before": calibrator.ece_before,
        "ece_after": calibrator.ece_after, "scaler": "RobustScaler",
        "benign_classes": len(benign_classes), "malicious_classes": len(mal_classes),
        "total_training_samples": len(X_aug),
    }
    with open(os.path.join(MODELS_DIR, "training_summary_v5.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'#'*60}")
    print(f"  RETRAIN v6 COMPLETE")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Classes: {n_classes} ({len(benign_classes)} benign, {len(mal_classes)} malicious)")
    print(f"  Benign FP: {ben_as_mal}/{ben_mask_test.sum()}")
    print(f"  Models → {MODELS_DIR}/")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
