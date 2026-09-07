"""BeliefTracker: per-seat, per-hand history accumulator.

Landmine 1: a single observation is not Markov — it does not record which
cards the opponent took from the pile or which upcards they declined. Every
policy consumes features from here, never raw observations alone.

Information hygiene contract: features derive ONLY from
(a) the perspective seat's own observation struct
    (state.to_observation_struct(p).to_dict()), and
(b) the public action stream (phases, acting seats, action ids, pile tops).
Anything else (opponent's hidden hand, stock order) is a leak and fails
tests/test_information_hygiene.py: features must be bit-identical across
resample_from_infostate resamples of the same infostate.

All internal card sets hold int indices (never strings: str hashing is
seed-dependent, int hashing is not), all sums are integers, and every float
is a per-element division — so the vector is bit-deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pyspiel import gin_rummy as gr

from ginrl.env import melds

DRAW_UPCARD = gr.DRAW_UPCARD_ACTION
DRAW_STOCK = gr.DRAW_STOCK_ACTION
PASS = gr.PASS_ACTION
KNOCK = gr.KNOCK_ACTION

N_PHASE_FEATURES = 3  # FirstUpcard, Draw, Discard
PHASE_INDEX = {"FirstUpcard": 0, "Draw": 1, "Discard": 2}

# 11 card-blocks of 52 (own_hand, own_melded, own_deadwood_cards,
# discard_cost, meld_adjacency, pile_top, pile_buried, pile_order, upcard,
# known_opp, declined_opp) + 14 scalars (phase x3, stock, turn,
# known_count, can_knock, dist_to_knock, own_deadwood, match x5).
FEATURE_DIM = 11 * 52 + 14


@dataclass
class BeliefTracker:
    """Tracks public history from the perspective of one seat, for one hand."""

    perspective: int
    knock_card: int = 10
    _taken: set[int] = field(default_factory=set)  # opp took from pile, still held
    _declined: set[int] = field(default_factory=set)  # opp declined these upcards
    _turn: int = field(default=0)
    _my_score: float = field(default=0.0)
    _opp_score: float = field(default=0.0)
    _target: float = field(default=100.0)

    # -- public event ingestion (called by the env after every engine action) --

    def observe_take(self, seat: int, card: int) -> None:
        if seat != self.perspective:
            self._taken.add(card)
        self._declined.discard(card)

    def observe_discard(self, seat: int, card: int) -> None:
        if seat != self.perspective:
            self._taken.discard(card)

    def observe_layoff(self, seat: int, card: int) -> None:
        if seat != self.perspective:
            self._taken.discard(card)

    def observe_decline(self, seat: int, card: int) -> None:
        if seat != self.perspective:
            self._declined.add(card)

    def observe_draw_stock(self, seat: int) -> None:
        _ = seat

    def observe_turn(self) -> None:
        self._turn += 1

    def set_match_state(self, my_score: float, opp_score: float, target: float) -> None:
        self._my_score = my_score
        self._opp_score = opp_score
        self._target = target

    # -- features ----------------------------------------------------------

    def features(self, obs: dict[str, object], mask: list[bool] | None = None) -> np.ndarray:
        """Feature vector (float32, FEATURE_DIM) from my observation struct."""
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)
        hands = obs["hands"]
        assert isinstance(hands, list)
        own = sorted(melds.card_to_index(c) for c in hands[self.perspective] if c != "XX")
        own_set = set(own)

        pile = obs.get("discard_pile")
        pile_cards = [melds.card_to_index(c) for c in pile] if isinstance(pile, list) else []
        upcard = obs.get("upcard")
        upcard_idx = melds.card_to_index(upcard) if isinstance(upcard, str) else -1
        stock = obs.get("stock_size")
        stock_size = int(stock) if isinstance(stock, int) else 0
        phase = obs.get("phase")
        phase_name = phase if isinstance(phase, str) else ""

        group = melds.best_meld_group(own)
        melded = {c for m in group for c in m}
        deadwood_cards = own_set - melded
        own_dw = sum(melds.card_value(c) for c in deadwood_cards)

        all_meld_options = melds.get_utils().all_melds(own)
        adjacency = [0] * melds.NUM_CARDS
        for option in all_meld_options:
            for c in option:
                adjacency[c] += 1

        discard_cost = [0] * melds.NUM_CARDS
        for c in own:
            rest = [h for h in own if h != c]
            discard_cost[c] = melds.min_deadwood(rest)

        dist_to_knock = own_dw
        if len(own) > 10 and own:
            dist_to_knock = min(discard_cost[c] for c in own)

        b = 0

        def block(values: list[float]) -> None:
            nonlocal b
            vec[b : b + 52] = values
            b += 52

        block([1.0 if c in own_set else 0.0 for c in range(52)])
        block([1.0 if c in melded else 0.0 for c in range(52)])
        block([1.0 if c in deadwood_cards else 0.0 for c in range(52)])
        block([discard_cost[c] / 50.0 if c in own_set else 0.0 for c in range(52)])
        block([min(adjacency[c], 10) / 10.0 for c in range(52)])
        # The takeable top is the upcard when present; discard_pile excludes it.
        top = upcard_idx if upcard_idx >= 0 else (pile_cards[-1] if pile_cards else -1)
        block([1.0 if c == top else 0.0 for c in range(52)])
        buried = set(pile_cards)
        if upcard_idx < 0 and pile_cards:
            buried.discard(pile_cards[-1])
        block([1.0 if c in buried else 0.0 for c in range(52)])
        order = [0.0] * 52
        for i, c in enumerate(pile_cards):
            order[c] = (i + 1) / 52.0
        block(order)
        block([1.0 if c == upcard_idx else 0.0 for c in range(52)])
        block([1.0 if c in self._taken else 0.0 for c in range(52)])
        block([1.0 if c in self._declined else 0.0 for c in range(52)])

        if phase_name in PHASE_INDEX:
            vec[b + PHASE_INDEX[phase_name]] = 1.0
        b += 3
        vec[b] = stock_size / 52.0
        b += 1
        vec[b] = min(self._turn, 100) / 100.0
        b += 1
        vec[b] = len(self._taken) / 10.0
        b += 1
        vec[b] = 1.0 if (mask is not None and len(mask) > KNOCK and mask[KNOCK]) else 0.0
        b += 1
        vec[b] = min(dist_to_knock, 50) / 50.0
        b += 1
        vec[b] = min(own_dw, 50) / 50.0
        b += 1
        vec[b] = self._my_score / 100.0
        b += 1
        vec[b] = self._opp_score / 100.0
        b += 1
        vec[b] = (self._my_score - self._opp_score) / 100.0
        b += 1
        vec[b] = max(self._target - self._my_score, 0.0) / 100.0
        b += 1
        vec[b] = max(self._target - self._opp_score, 0.0) / 100.0
        b += 1
        assert b == FEATURE_DIM, f"feature layout drift: {b} != {FEATURE_DIM}"
        return vec
