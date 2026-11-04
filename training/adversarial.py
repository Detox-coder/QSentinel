import torch
import torch.nn as nn
import numpy as np


def fgsm_attack(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                criterion: nn.Module, epsilon: float = 0.1) -> torch.Tensor:
    X_adv = X.clone().detach().requires_grad_(True)
    logits = model(X_adv)
    loss = criterion(logits, y)
    loss.backward()
    perturbation = epsilon * X_adv.grad.data.sign()
    X_perturbed = X_adv.data + perturbation
    return X_perturbed.detach()


def pgd_attack(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
               criterion: nn.Module, epsilon: float = 0.1,
               alpha: float = 0.02, n_steps: int = 7) -> torch.Tensor:
    X_adv = X.clone().detach()
    X_orig = X.clone().detach()

    for _ in range(n_steps):
        X_adv.requires_grad_(True)
        logits = model(X_adv)
        loss = criterion(logits, y)
        loss.backward()
        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            delta = torch.clamp(X_adv - X_orig, min=-epsilon, max=epsilon)
            X_adv = X_orig + delta
        X_adv = X_adv.detach()

    return X_adv


def gaussian_noise(X: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
    noise = torch.randn_like(X) * noise_std
    return X + noise


def random_feature_mask(X: torch.Tensor, mask_prob: float = 0.15) -> torch.Tensor:
    mask = torch.bernoulli(torch.ones_like(X) * (1 - mask_prob))
    return X * mask


def adversarial_training_step(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epsilon: float = 0.1,
    noise_std: float = 0.05,
    mask_prob: float = 0.10,
    adv_ratio: float = 0.5,
) -> dict:
    model.train()
    batch_size = X.shape[0]

    # ── Clean forward pass ────────────────────────
    optimizer.zero_grad()
    logits_clean = model(X)
    loss_clean = criterion(logits_clean, y)
    correct_clean = (logits_clean.argmax(1) == y).sum().item()

    # ── FGSM adversarial examples ─────────────────
    X_adv = fgsm_attack(model, X, y, criterion, epsilon)
    logits_adv = model(X_adv)
    loss_adv = criterion(logits_adv, y)

    # ── Noise-augmented examples ──────────────────
    X_noisy = gaussian_noise(X, noise_std)
    X_noisy = random_feature_mask(X_noisy, mask_prob)
    logits_noisy = model(X_noisy)
    loss_noisy = criterion(logits_noisy, y)

    # ── Combined loss ─────────────────────────────
    # Weighted: 40% clean + 40% adversarial + 20% noisy
    total_loss = 0.4 * loss_clean + 0.4 * loss_adv + 0.2 * loss_noisy
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "clean_loss": loss_clean.item(),
        "adv_loss": loss_adv.item(),
        "noise_loss": loss_noisy.item(),
        "correct": correct_clean,
        "total": batch_size,
    }
