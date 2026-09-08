"""Shared-torso actor-critic for masked imperfect-information games.

One residual-MLP torso feeds three fixed heads: policy over the game's
action set, value, and an auxiliary opponent-card head (multilabel: which
hidden cards does the opponent hold). The three heads are fixed — the aux
head backs a tripwire and the Phase 6 inspectability story. The torso is a
default: judge alternatives on `gate-p4`, `belief/auc` and `perf/`.

Action masking is applied to logits before the softmax, never by
renormalising after sampling. Everything touching torch is float32
(landmine 8: MPS has no float64).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ginrl.utils.device import pick_device

LOGIT_FLOOR = -1e9  # illegal actions: finite (no -inf NaNs), ~zero probability


@dataclass(frozen=True)
class NetConfig:
    """Network shape. Defaults fit Kuhn/Leduc on CPU; resize per game."""

    obs_dim: int = 12
    n_actions: int = 2
    aux_dim: int = 3
    hidden: int = 128
    layers: int = 2


class ResidualBlock(nn.Module):
    """Pre-activation residual block over a flat feature vector."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MaskedActorCritic(nn.Module):
    """Residual-MLP torso + policy / value / opponent-card heads."""

    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else NetConfig()
        cfg = self.config
        self.stem = nn.Linear(cfg.obs_dim, cfg.hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(cfg.hidden) for _ in range(cfg.layers)])
        self.policy_head = nn.Linear(cfg.hidden, cfg.n_actions)
        self.value_head = nn.Linear(cfg.hidden, 1)
        self.aux_head = nn.Linear(cfg.hidden, cfg.aux_dim)

    def forward(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (masked logits [B, A], value [B], aux logits [B, aux]).

        `mask[b, a]` is True for legal actions. Every row must have at least
        one legal action; illegal logits sit at LOGIT_FLOOR.
        """
        obs = obs.to(torch.float32)
        h = self.blocks(self.stem(obs))
        logits = self.policy_head(h).masked_fill(~mask, LOGIT_FLOOR)
        return logits, self.value_head(h).squeeze(-1), self.aux_head(h)

    @staticmethod
    def log_probs(logits: torch.Tensor) -> torch.Tensor:
        """Masked log-softmax (masking already applied in forward)."""
        return torch.log_softmax(logits, dim=-1)

    @staticmethod
    def entropy(logits: torch.Tensor) -> torch.Tensor:
        """Shannon entropy of the masked policy, per row."""
        logp = torch.log_softmax(logits, dim=-1)
        return -(logp.exp() * logp).sum(dim=-1)

    def to_learner_device(self, prefer: str = "auto") -> MaskedActorCritic:
        """Move to the benchmarked learner device (AGENTS.md: no hardcoding)."""
        return self.to(pick_device(prefer))
