"""
Data loading, preprocessing, splitting, and DataLoader creation.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


class DataPipeline:
    """
    End-to-end data pipeline:
      Load → Clean → Encode labels → Normalize → Split → DataLoaders
    """

    META_COLS = ["flow_id", "label", "traffic_type"]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: list[str] = []
        self.class_names: list[str] = []
        self.n_features: int = 0
        self.n_classes: int = 0

    # ── Public API ───────────────────────────────────────

    def load_and_prepare(
        self,
        data_path: str,
        test_size: float = 0.15,
        val_size: float = 0.15,
        batch_size: int = 32,
    ) -> dict:
        """
        Full pipeline. Returns dict with DataLoaders and metadata.
        """
        # 1. Load
        df = self._load(data_path)
        print(f"[DATA] Loaded {len(df)} flows from {data_path}")
        print(f"[DATA] Label distribution:")
        for label, count in df["label"].value_counts().items():
            print(f"       {label:30s} {count:5d}")

        # 2. Separate features and labels
        feature_cols = [c for c in df.columns if c not in self.META_COLS]
        self.feature_names = feature_cols
        self.n_features = len(feature_cols)

        X = df[feature_cols].values.astype(np.float32)
        y_str = df["label"].values

        # 3. Handle NaN / Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. Encode labels
        self.label_encoder.fit(y_str)
        y = self.label_encoder.transform(y_str)
        self.class_names = list(self.label_encoder.classes_)
        self.n_classes = len(self.class_names)
        print(f"[DATA] Classes ({self.n_classes}): {self.class_names}")
        print(f"[DATA] Features: {self.n_features}")

        # 5. Train / temp split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(test_size + val_size),
            random_state=self.random_state,
            stratify=y,
        )

        # 6. Val / test split from temp
        relative_test = test_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=relative_test,
            random_state=self.random_state,
            stratify=y_temp,
        )

        print(f"[DATA] Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        # 7. Fit scaler on train, transform all
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        # 8. Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        # 9. DataLoaders
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size, shuffle=True, drop_last=False,
        )
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=batch_size, shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(X_test_t, y_test_t),
            batch_size=batch_size, shuffle=False,
        )

        # 10. Benign-only loader for VAE training
        benign_indices = np.isin(self.label_encoder.classes_[y_train],
                                  ["benign_web", "benign_streaming"])
        X_benign = X_train_t[benign_indices]
        benign_loader = DataLoader(
            TensorDataset(X_benign, X_benign),  # autoencoder: input=target
            batch_size=batch_size, shuffle=True,
        )

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "benign_loader": benign_loader,
            "X_train": X_train_t,
            "y_train": y_train_t,
            "X_val": X_val_t,
            "y_val": y_val_t,
            "X_test": X_test_t,
            "y_test": y_test_t,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "class_names": self.class_names,
            "feature_names": self.feature_names,
        }

    def save_artifacts(self, output_dir: str) -> None:
        """Save scaler, label encoder, and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.joblib"))
        joblib.dump(self.label_encoder, os.path.join(output_dir, "label_encoder.joblib"))

        meta = {
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "class_names": self.class_names,
            "feature_names": self.feature_names,
        }
        with open(os.path.join(output_dir, "pipeline_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[DATA] Artifacts saved to {output_dir}")

    # ── Internal ─────────────────────────────────────────

    @staticmethod
    def _load(path: str) -> pd.DataFrame:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        elif path.endswith(".csv"):
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported format: {path}")
