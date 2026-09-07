"""Held-out probe classifiers for the history-necessity check.

Predicts the opponent's 10-card hand (52-way multilabel) from state features.
Multilabel, because a per-(state, card) probe without the queried card's
identity cannot use card-indexed history at all. Split is by game (deal),
never by state: nearby states from one game share nearly identical hands.
"""

from __future__ import annotations

import numpy as np
import torch

OBS_DIM = 644


def mean_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    """Mean per-card AUC over cards held at least once and missed at least once."""
    aucs = []
    for j in range(labels.shape[1]):
        lab = labels[:, j]
        pos = float(lab.sum())
        neg = float(len(lab) - pos)
        if pos == 0 or neg == 0:
            continue
        order = np.argsort(scores[:, j])
        ranked = lab[order]
        ranks = np.arange(1, len(ranked) + 1)
        aucs.append((float(ranks[ranked == 1].sum()) - pos * (pos + 1) / 2) / (pos * neg))
    return (float(np.mean(aucs)), len(aucs)) if aucs else (float("nan"), 0)


def train_multilabel_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int = 0,
    iters: int = 300,
    lr: float = 0.05,
) -> np.ndarray:
    """Linear  D->52 probe, BCE loss. Returns test-split scores."""
    torch.manual_seed(seed)
    dim = x_train.shape[1]
    weight = torch.zeros(dim, 52, requires_grad=True)
    bias = torch.zeros(52, requires_grad=True)
    opt = torch.optim.Adam([weight, bias], lr=lr)
    inputs = torch.from_numpy(x_train.astype(np.float32))
    targets = torch.from_numpy(y_train.astype(np.float32))
    for _ in range(iters):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs @ weight + bias, targets)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (torch.from_numpy(x_test.astype(np.float32)) @ weight + bias).numpy()
