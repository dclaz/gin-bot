"""Random and heuristic agents over the 56-action learned interface."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pyspiel
from pyspiel import gin_rummy as gr

from ginrl.env import melds
from ginrl.env.game import N_LEARNED_ACTIONS, HandEnv


class RandomAgent:
    """Uniform over legal learned actions."""

    manual_phases = False

    def __init__(self, name: str = "random", seed: int = 0) -> None:
        self.name = name
        self._rng = random.Random(seed)

    def begin_game(self, seat: int) -> None:
        _ = seat

    def choose(self, env: HandEnv, seat: int) -> int:
        _ = seat
        legal = [a for a, m in enumerate(env.legal_mask()) if m]
        return self._rng.choice(legal)

    def choose_raw(self, state: pyspiel.State) -> int:
        raise AssertionError("random agent has no manual phases")

    def inform(self, state: pyspiel.State, player: int, action: int) -> None:
        _ = (state, player, action)


@dataclass(frozen=True)
class HeuristicParams:
    """Knobs of the heuristic family (each setting is a ladder member)."""

    knock_threshold: int = 7
    discard_danger_weight: float = 1.0
    pile_draw_aggression: float = 0.5


class HeuristicAgent:
    manual_phases = False

    """Deadwood minimiser with opponent-aware discards.

    - Draw: take the upcard iff it lowers post-discard deadwood by more
      than pile_draw_aggression (FirstUpcard pass costs nothing extra).
    - Discard: minimise deadwood plus danger_weight * danger, where danger
      counts how badly the opponent is known to want the card (adjacent in
      suit to known holdings, or matching a known rank).
    - Knock as soon as post-discard deadwood <= knock_threshold.
    """

    def __init__(self, params: HeuristicParams | None = None, name: str | None = None) -> None:
        self.params = params or HeuristicParams()
        p = self.params
        self.name = (
            name
            or f"heuristic(T{p.knock_threshold},D{p.discard_danger_weight},A{p.pile_draw_aggression})"
        )

    def begin_game(self, seat: int) -> None:
        _ = seat

    def inform(self, state: pyspiel.State, player: int, action: int) -> None:
        _ = (state, player, action)

    def choose_raw(self, state: pyspiel.State) -> int:
        raise AssertionError("heuristic agent has no manual phases")

    def choose(self, env: HandEnv, seat: int) -> int:
        state = env.state()
        info = state.to_dict()
        hand = melds.parse_hand(info["hands"][seat])
        mask = env.legal_mask()
        tracker = env.trackers[seat]
        phase = str(info["phase"])
        up = info["upcard"]
        if up and phase in ("FirstUpcard", "Draw") and mask[gr.DRAW_UPCARD_ACTION]:
            up_idx = melds.card_to_index(up)
            keep = melds.min_deadwood(hand)
            eleven = hand + [up_idx]
            # Simulate the discard this agent would actually choose (deadwood
            # + danger), not the deadwood-only knock discard: taking a card
            # the danger term then vetoes keeping is a wasted take, and
            # repeated take-and-rediscard ends the hand as an engine draw.
            planned = min(eleven, key=lambda c: self._discard_cost(eleven, c, tracker))
            take = melds.min_deadwood([h for h in eleven if h != planned])
            if keep - take > self.params.pile_draw_aggression:
                return gr.DRAW_UPCARD_ACTION
            if phase == "Draw" and mask[gr.DRAW_STOCK_ACTION]:
                # Failed take test means draw from the stock. Falling through
                # to `legal[0]` below would take the upcard — action 52 sorts
                # before 53 — making every draw a pile take.
                return gr.DRAW_STOCK_ACTION
        if phase == "Discard":
            if mask[gr.KNOCK_ACTION]:
                best = min(melds.min_deadwood([h for h in hand if h != c]) for c in hand)
                if best <= self.params.knock_threshold:
                    return gr.KNOCK_ACTION
            options = [c for c in hand if c < N_LEARNED_ACTIONS and mask[c]]
            assert options, "no legal discard"
            return min(options, key=lambda c: self._discard_cost(hand, c, tracker))
        if mask[gr.PASS_ACTION]:
            return gr.PASS_ACTION
        legal = [a for a, m in enumerate(mask) if m]
        assert legal, "no legal action"
        return legal[0]

    def _discard_cost(self, hand: list[int], card: int, tracker) -> float:
        rest = [h for h in hand if h != card]
        deadwood = melds.min_deadwood(rest)
        return deadwood + self.params.discard_danger_weight * _danger(card, tracker)


def _danger(card: int, tracker) -> float:
    """How badly the opponent is known to want `card` (0 = safe)."""
    known = tracker.known_cards()
    if not known:
        return 0.0
    suit, rank = divmod(card, melds.NUM_RANKS)
    score = 0.0
    for other in known:
        osuit, orank = divmod(other, melds.NUM_RANKS)
        if osuit == suit and abs(orank - rank) <= 2:
            score += 1.0
        elif orank == rank:
            score += 1.0
    return score
