"""Common Agent interface.

Every agent — random, heuristic, engine bot, learned net — plays through
choose(): given the shared HandEnv positioned at a learned decision for
`seat`, return a learned action (0..55). Bots additionally need the full
transition stream (landmine 3): begin_game/inform route it to them.
"""

from __future__ import annotations

from typing import Protocol

import pyspiel

from ginrl.env.game import HandEnv


class Agent(Protocol):
    name: str
    # True when this agent drives its own Knock/Layoff phases natively
    # through choose_raw (engine bots). False uses the meld auto-policy.
    manual_phases: bool

    def begin_game(self, seat: int) -> None:
        """Fresh per-game state. Called once per game, before any choose/inform."""
        ...

    def choose(self, env: HandEnv, seat: int) -> int:
        """Return a learned action (0..55) for seat at its decision node."""
        ...

    def choose_raw(self, state: pyspiel.State) -> int:
        """Return any legal engine action at a Knock/Layoff node (manual only)."""
        ...

    def inform(self, state: pyspiel.State, player: int, action: int) -> None:
        """Observe another transition (opponent/chance/auto action). No-op default."""
        ...
