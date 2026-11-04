"""
Supervised feedforward neural network classifier for QUIC traffic.

Architecture:
  Input(90) → 256 → 128 → 64 → Output(6)
  With BatchNorm, ReLU, Dropout at each hidden layer.
"""

import torch
import torch.nn as nn


class QUICClassifier(nn.Module):
    """
    Multi-class feedforward classifier for QUIC flow classification.
    """

    def __init__(
        self,
        n_features: int = 90,
        n_classes: int = 6,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, n_classes)

        # Initialize weights
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (no softmax — use CrossEntropyLoss)."""
        features = self.backbone(x)
        return self.head(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices."""
        probs = self.predict_proba(x)
        return probs.argmax(dim=-1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"QUICClassifier: {self.n_features}→256→128→64→{self.n_classes}\n"
            f"  Total params:     {total:,}\n"
            f"  Trainable params: {trainable:,}"
        )


