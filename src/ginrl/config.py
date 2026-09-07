"""Shared config dataclasses and the single-seed RNG rule.

Every RNG in ginrl is seeded from a Seeds object; no bare random.random(),
no unseeded np.random. Experiment constants live here, not inline.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class MatchConfig:
    """Match wrapper settings. Target 100 is the North American/EAAI standard."""

    hand: HandConfig = field(default_factory=HandConfig)
    target_score: int = 100
    max_hands: int = 1000
