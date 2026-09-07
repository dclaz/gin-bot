"""Match play: successive hands to a target score (default 100).

The match is the episode; hand scores are intermediate rewards; the running
score is part of the state (via set_match_state on the BeliefTracker).
Duplicate matches replay the same seed sequence with the agents' seats
swapped — the driver swaps seats, the envs replay identical shuffles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ginrl.config import MatchConfig, Seeds
from ginrl.env.game import HandEnv, StepResult


@dataclass
class MatchResult:
    hand: StepResult  # underlying hand step (obs/mask live here until match end)
    scores: tuple[float, float]
    hand_index: int
    hand_returns: tuple[float, float] | None
    done: bool
    winner: int  # -1 while running


@dataclass
class MatchEnv:
    config: MatchConfig = field(default_factory=MatchConfig)
    seeds: Seeds = field(default_factory=Seeds)

    def __post_init__(self) -> None:
        self._hand = HandEnv(self.config.hand, self.seeds)
        self._scores = (0.0, 0.0)
        self._hand_index = 0
        self._base_seed = 0

    def reset(self, seed: int | None = None) -> MatchResult:
        self._scores = (0.0, 0.0)
        self._hand_index = 0
        self._base_seed = seed if seed is not None else 0
        return self._wrap(self._reset_hand())

    @property
    def scores(self) -> tuple[float, float]:
        return self._scores

    @property
    def hand(self) -> HandEnv:
        return self._hand

    def step(self, action: int) -> MatchResult:
        step = self._hand.step(action)
        if not step.done:
            return MatchResult(
                hand=step,
                scores=self._scores,
                hand_index=self._hand_index,
                hand_returns=None,
                done=False,
                winner=-1,
            )
        assert step.returns is not None
        # Match scoring: only the hand winner's points count; the loser
        # adds nothing (scores never go negative).
        self._scores = (
            self._scores[0] + max(step.returns[0], 0.0),
            self._scores[1] + max(step.returns[1], 0.0),
        )
        hand_returns = step.returns
        hand_index = self._hand_index
        winner = self._match_winner()
        done = winner >= 0 or self._hand_index + 1 >= self.config.max_hands
        if done:
            if winner < 0:  # max_hands guard: leader wins
                winner = 0 if self._scores[0] >= self._scores[1] else 1
            return MatchResult(
                hand=step,
                scores=self._scores,
                hand_index=hand_index,
                hand_returns=hand_returns,
                done=True,
                winner=winner,
            )
        self._hand_index += 1
        return self._wrap(self._reset_hand(), hand_returns=hand_returns)

    # -- internals -------------------------------------------------------

    def _match_winner(self) -> int:
        leaders = [i for i, s in enumerate(self._scores) if s >= self.config.target_score]
        if not leaders:
            return -1
        return max(leaders, key=lambda i: self._scores[i])

    def _reset_hand(self) -> StepResult:
        return self._hand.reset(seed=self._base_seed * 100003 + self._hand_index)

    def _wrap(
        self, step: StepResult, hand_returns: tuple[float, float] | None = None
    ) -> MatchResult:
        return MatchResult(
            hand=step,
            scores=self._scores,
            hand_index=self._hand_index,
            hand_returns=hand_returns,
            done=False,
            winner=-1,
        )
