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
    # True when the match ended on the max_hands cap rather than the target.
    # Every cap hit is reported here; the gate asserts zero unreported hits.
    capped: bool = False


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
            capped = winner < 0
            if capped:  # max_hands guard: leader wins
                winner = 0 if self._scores[0] >= self._scores[1] else 1
            return MatchResult(
                hand=step,
                scores=self._scores,
                hand_index=hand_index,
                hand_returns=hand_returns,
                done=True,
                winner=winner,
                capped=capped,
            )
        self._hand_index += 1
        return self._wrap(self._reset_hand(), hand_returns=hand_returns, hand_index=hand_index)

    # -- internals -------------------------------------------------------

    def _match_winner(self) -> int:
        leaders = [i for i, s in enumerate(self._scores) if s >= self.config.target_score]
        if not leaders:
            return -1
        return max(leaders, key=lambda i: self._scores[i])

    def _reset_hand(self) -> StepResult:
        result = self._hand.reset(seed=self._base_seed * 100003 + self._hand_index)
        self._push_scores()
        return result

    def _push_scores(self) -> None:
        """Publish the running score to both trackers (seat-relative).

        Called on every hand boundary, including match reset (0-0). Scores
        are public, so this is hygiene-safe; without it the match scalars
        sit at their defaults and the policy cannot learn score dependence.
        """
        for seat in (0, 1):
            self._hand.trackers[seat].set_match_state(
                my_score=self._scores[seat],
                opp_score=self._scores[1 - seat],
                target=float(self.config.target_score),
            )

    def _wrap(
        self,
        step: StepResult,
        hand_returns: tuple[float, float] | None = None,
        hand_index: int | None = None,
    ) -> MatchResult:
        return MatchResult(
            hand=step,
            scores=self._scores,
            hand_index=self._hand_index if hand_index is None else hand_index,
            hand_returns=hand_returns,
            done=False,
            winner=-1,
        )
