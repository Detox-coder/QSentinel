"""
Isolation Forest anomaly detector for QUIC traffic.

Wraps sklearn IsolationForest with a consistent anomaly_score() interface
for use by the risk engine, inference engine, and ensemble decision system.
"""

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIF


class QUICIsolationForest:
    """Isolation Forest anomaly detector trained on benign traffic."""

    def __init__(self, n_estimators: int = 200, contamination: float = 0.01,
                 max_features: float = 0.8, random_state: int = 42):
        self.model = SklearnIF(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X_benign: np.ndarray):
        """Train on benign-only data."""
        self.model.fit(X_benign)
        self._fitted = True
        # Compute baseline score statistics for normalization
        raw = -self.model.score_samples(X_benign)
        self._score_mean = float(raw.mean())
        self._score_std = float(raw.std())
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Per-sample anomaly score (higher = more anomalous).
        Normalized so benign training data has mean ~0, std ~1.
        Scaled to approximate VAE score range for compatibility.
        """
        raw = -self.model.score_samples(X)
        # Z-score normalize then scale to match VAE-like range
        normalized = (raw - self._score_mean) / max(self._score_std, 1e-8)
        # Shift so benign is ~3-5 range, anomalous is >15
        return np.clip(normalized * 5.0 + 5.0, 0, None)

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Returns 1 for anomalous, 0 for normal."""
        scores = self.anomaly_score(X)
        if threshold is None:
            return (self.model.predict(X) == -1).astype(int)
        return (scores > threshold).astype(int)

    def summary(self) -> str:
        return (
            f"QUICIsolationForest: {self.model.n_estimators} trees, "
            f"contamination={self.model.contamination}, "
            f"fitted={self._fitted}"
        )
