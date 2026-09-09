"""Shared config dataclasses and the single-seed RNG rule.

Every RNG in ginrl is seeded from a Seeds object; no bare random.random(),
no unseeded np.random. Experiment constants live here, not inline.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from pyspiel import gin_rummy as gr


@dataclass(frozen=True)
class Seeds:
    """One master seed; derive independent streams by name."""

    master: int = 0

    def spawn(self, stream: str) -> random.Random:
        digest = hashlib.sha256(f"{self.master}:{stream}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))


@dataclass(frozen=True)
class HandConfig:
    """Single-hand environment settings (engine game params)."""

    gin_bonus: int = 25
    undercut_bonus: int = 25
    knock_card: int = 10
    hand_size: int = 10
    num_ranks: int = 13
    num_suits: int = 4
    oklahoma: bool = False

    def __post_init__(self) -> None:
        minimum = 2 * self.hand_size + gr.WALL_STOCK_SIZE + 1
        if self.num_ranks * self.num_suits < minimum:
            raise ValueError(
                "deck too small for two hands, wall stock, and one upcard: "
                f"{self.num_ranks * self.num_suits} < {minimum}"
            )

    @property
    def deck_size(self) -> int:
        return self.num_ranks * self.num_suits

    def game_params(self) -> dict[str, object]:
        return {
            "gin_bonus": self.gin_bonus,
            "undercut_bonus": self.undercut_bonus,
            "knock_card": self.knock_card,
            "hand_size": self.hand_size,
            "num_ranks": self.num_ranks,
            "num_suits": self.num_suits,
            "oklahoma": self.oklahoma,
        }


# Phase 4 reduced game: deck 10, hand 3. The padded engine action and
# observation layouts stay 241 and 644; masking selects the legal subset.
REDUCED_HAND_CONFIG = HandConfig(num_ranks=5, num_suits=2, hand_size=3)


@dataclass(frozen=True)
class MatchConfig:
    """Match wrapper settings. Target 100 is the North American/EAAI standard."""

    hand: HandConfig = field(default_factory=HandConfig)
    target_score: int = 100
    max_hands: int = 1000


# Magnet modes for the regularised learner (IMPLEMENTATION_PLAN Phase 3).
# `uniform` is PPO with an entropy bonus (KL to uniform, up to a constant);
# `snapshot` is MMD proper; `ema` tracks the learner's own parameters.
MAGNET_UNIFORM = "uniform"
MAGNET_EMA = "ema"
MAGNET_SNAPSHOT = "snapshot"
MAGNET_MODES = (MAGNET_UNIFORM, MAGNET_EMA, MAGNET_SNAPSHOT)

# Environments the trainer supports. Kuhn/Leduc calibrate the learner;
# gin_reduced is the bridge to full gin (Phase 4+).
TRAIN_ENVS = ("kuhn", "leduc", "gin_reduced")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class TrainerConfig:
    """PPO learner settings.

    Defaults are sized for the 24GB M4 notebook (CPU learner): Kuhn/Leduc
    runs are tiny, and reduced-gin smoke runs fit comfortably. Re-benchmark
    and resize before any expensive run (AGENTS.md landmine 9); changing a
    default is fine, the Phase 3 gate is the contract.
    """

    env: str = "kuhn"
    total_steps: int = 200_000
    n_envs: int = 16
    rollout_len: int = 128
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    reg_coef: float = 0.0
    aux_coef: float = 0.1
    # Linear reward rescaling for the gin path (critic preconditioning).
    # Positive scaling preserves zero-sum, Nash set, and best responses;
    # it is not shaping. Eval scores are always unscaled. Default 1.0 keeps
    # every calibrated small-game recipe bit-identical.
    reward_scale: float = 1.0
    advantage: str = "gae"  # "gae" | "mc" (estimator guard decides)
    anneal: str = "none"  # "none" | "linear" (lr and reg_coef -> 0)
    magnet_mode: str = MAGNET_UNIFORM
    magnet_ema_decay: float = 0.999
    snapshot_every: int = 0
    eval_every: int = 0
    eval_games: int = 200
    log_every: int = 100
    seeds: Seeds = field(default_factory=Seeds)
    device: str = "auto"

    def __post_init__(self) -> None:
        _require(self.env in TRAIN_ENVS, f"env must be one of {TRAIN_ENVS}, got {self.env!r}")
        _require(
            self.magnet_mode in MAGNET_MODES,
            f"magnet_mode must be one of {MAGNET_MODES}, got {self.magnet_mode!r}",
        )
        for name in ("total_steps", "n_envs", "rollout_len", "epochs", "minibatches"):
            _require(
                getattr(self, name) > 0,
                f"{name} must be positive, got {getattr(self, name)}",
            )
        _require(
            self.minibatches <= self.n_envs * self.rollout_len,
            "minibatches exceeds rollout batch size",
        )
        _require(
            0.0 <= self.magnet_ema_decay <= 1.0,
            f"magnet_ema_decay must be in [0, 1], got {self.magnet_ema_decay}",
        )
        _require(self.reward_scale > 0.0, f"reward_scale must be positive, got {self.reward_scale}")
        _require(
            self.magnet_mode != MAGNET_SNAPSHOT or self.snapshot_every > 0,
            "snapshot magnet needs snapshot_every > 0",
        )
        _require(
            self.advantage in ("gae", "mc"),
            f"advantage must be 'gae' or 'mc', got {self.advantage!r}",
        )
        _require(
            self.anneal in ("none", "linear"),
            f"anneal must be 'none' or 'linear', got {self.anneal!r}",
        )

    @property
    def batch_size(self) -> int:
        """Transitions collected per rollout round."""
        return self.n_envs * self.rollout_len

    @property
    def minibatch_size(self) -> int:
        """Transitions per gradient minibatch."""
        return self.batch_size // self.minibatches
