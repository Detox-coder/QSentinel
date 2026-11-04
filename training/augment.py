"""
Data augmentation for scaling up the synthetic QUIC IDS dataset.

Strategies:
  1. Gaussian noise injection (preserves statistical profile)
  2. Feature-wise scaling perturbation (realistic variation)
  3. SMOTE-like interpolation between same-class samples
"""

import numpy as np
import pandas as pd


def augment_gaussian(X: np.ndarray, n_aug: int, noise_scale: float = 0.05,
                     rng: np.random.Generator = None) -> np.ndarray:
    """Add Gaussian noise to randomly selected samples."""
    if rng is None:
        rng = np.random.default_rng(42)
    indices = rng.integers(0, len(X), size=n_aug)
    base = X[indices]
    stds = np.std(X, axis=0, keepdims=True).clip(min=1e-8)
    noise = rng.normal(0, noise_scale, size=base.shape) * stds
    augmented = base + noise
    return augmented


def augment_scale(X: np.ndarray, n_aug: int, scale_range: tuple = (0.9, 1.1),
                  rng: np.random.Generator = None) -> np.ndarray:
    """Multiply features by random scaling factors."""
    if rng is None:
        rng = np.random.default_rng(42)
    indices = rng.integers(0, len(X), size=n_aug)
    base = X[indices]
    scales = rng.uniform(scale_range[0], scale_range[1], size=base.shape)
    return base * scales


def augment_interpolate(X: np.ndarray, n_aug: int,
                        rng: np.random.Generator = None) -> np.ndarray:
    """SMOTE-like: interpolate between random pairs of same-class samples."""
    if rng is None:
        rng = np.random.default_rng(42)
    idx_a = rng.integers(0, len(X), size=n_aug)
    idx_b = rng.integers(0, len(X), size=n_aug)
    lam = rng.uniform(0.2, 0.8, size=(n_aug, 1))
    return X[idx_a] * lam + X[idx_b] * (1 - lam)


def augment_class(X: np.ndarray, target_count: int,
                  rng: np.random.Generator = None) -> np.ndarray:
    """
    Augment a single class to target_count using mixed strategies.
    Returns the FULL set (original + augmented).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    current = len(X)
    if current >= target_count:
        return X[:target_count]

    needed = target_count - current
    # Split augmentation across strategies: 40% noise, 30% scale, 30% interpolation
    n_noise = int(needed * 0.4)
    n_scale = int(needed * 0.3)
    n_interp = needed - n_noise - n_scale

    parts = [X]
    if n_noise > 0:
        parts.append(augment_gaussian(X, n_noise, noise_scale=0.05, rng=rng))
    if n_scale > 0:
        parts.append(augment_scale(X, n_scale, scale_range=(0.92, 1.08), rng=rng))
    if n_interp > 0:
        parts.append(augment_interpolate(X, n_interp, rng=rng))

    result = np.concatenate(parts, axis=0)
    # Clip to non-negative for features that should be positive
    result = np.clip(result, 0, None)
    return result[:target_count]


def augment_dataset(df: pd.DataFrame, feature_cols: list[str],
                    target_per_class: int = 600,
                    random_state: int = 42) -> pd.DataFrame:
    """
    Augment entire dataset to target_per_class per label.
    Returns a new DataFrame with augmented rows.
    """
    rng = np.random.default_rng(random_state)
    augmented_parts = []

    for label in sorted(df["label"].unique()):
        mask = df["label"] == label
        subset = df[mask]
        X = subset[feature_cols].values.astype(np.float64)

        X_aug = augment_class(X, target_per_class, rng=rng)

        aug_df = pd.DataFrame(X_aug, columns=feature_cols)
        aug_df["label"] = label

        # Copy non-feature columns from original for first N rows
        meta_cols = [c for c in df.columns if c not in feature_cols and c != "label"]
        for col in meta_cols:
            orig_vals = subset[col].values
            # Tile original metadata to fill augmented rows
            tiled = np.tile(orig_vals, (target_per_class // len(orig_vals) + 1))[:target_per_class]
            aug_df[col] = tiled

        # Mark augmented rows
        aug_df["is_augmented"] = False
        aug_df.loc[len(subset):, "is_augmented"] = True

        augmented_parts.append(aug_df)

        orig_count = len(subset)
        new_count = len(aug_df)
        print(f"  {label:30s} {orig_count:4d} → {new_count:4d}  "
              f"(+{new_count - orig_count} augmented)")

    result = pd.concat(augmented_parts, ignore_index=True)
    return result
