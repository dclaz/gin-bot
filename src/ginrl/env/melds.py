"""Thin adapter over pyspiel.gin_rummy.GinRummyUtils. No original meld logic.

Card index layout (engine): index = suit * 13 + rank with suits s=0, c=1,
d=2, h=3 and ranks A=0 .. K=12. Structs use strings ("As", "Td"); the utils
take int indices. This module converts and delegates; the engine computes.

Deliberately NOT wrapped: GinRummyUtils.all_layoffs. Its binding accepts
several argument shapes without erroring and none of them reproduce the
engine's legal layoff set (checked against legal_actions at Layoff nodes),
so its contract is unverifiable from Python. Layoff options come from the
engine's legal_actions; deadwood math comes from min_deadwood/card_value.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pyspiel import gin_rummy as gr

from ginrl import spiel_facts

RANKS = "A23456789TJQK"
SUITS = "scdh"
NUM_RANKS: int = spiel_facts.GAME_PARAMS["num_ranks"]
NUM_SUITS: int = spiel_facts.GAME_PARAMS["num_suits"]
NUM_CARDS = NUM_SUITS * NUM_RANKS


@dataclass(frozen=True)
class CardLayout:
    """Compact card codec for one gin deck size.

    The engine pads actions/observations to the full-deck layout, but reduced
    games use compact indices ``suit * num_ranks + rank`` over the leading
    ranks and suits. Codec and meld helpers must use the active layout.
    """

    num_ranks: int = NUM_RANKS
    num_suits: int = NUM_SUITS

    def __post_init__(self) -> None:
        if not 1 <= self.num_ranks <= len(RANKS):
            raise ValueError(f"num_ranks must be 1..{len(RANKS)}, got {self.num_ranks}")
        if not 1 <= self.num_suits <= len(SUITS):
            raise ValueError(f"num_suits must be 1..{len(SUITS)}, got {self.num_suits}")

    @property
    def deck_size(self) -> int:
        return self.num_ranks * self.num_suits

    @property
    def ranks(self) -> str:
        return RANKS[: self.num_ranks]

    @property
    def suits(self) -> str:
        return SUITS[: self.num_suits]

    def card_to_index(self, card: str) -> int:
        """'As' -> 0 in every layout; 'Ac' -> 13 full, 5 reduced."""
        return self.suits.index(card[1]) * self.num_ranks + self.ranks.index(card[0])

    def index_to_card(self, index: int) -> str:
        """Inverse codec within this layout's compact index range."""
        if not 0 <= index < self.deck_size:
            raise ValueError(f"card index {index} outside 0..{self.deck_size - 1}")
        suit, rank = divmod(index, self.num_ranks)
        return f"{self.ranks[rank]}{self.suits[suit]}"

    def parse_hand(self, cards: list[str]) -> list[int]:
        return [self.card_to_index(c) for c in cards]


DEFAULT_LAYOUT = CardLayout()


def layout_from_params(params: dict[str, object]) -> tuple[CardLayout, int]:
    """Active codec and hand size from engine game parameters."""
    layout = CardLayout(
        num_ranks=int(params.get("num_ranks", NUM_RANKS)),
        num_suits=int(params.get("num_suits", NUM_SUITS)),
    )
    hand_size = int(params.get("hand_size", spiel_facts.GAME_PARAMS["hand_size"]))
    return layout, hand_size


def card_to_index(card: str, layout: CardLayout | None = None) -> int:
    """'As' -> 0, 'Ks' -> 12, 'Ac' -> 13, ... (full deck by default)."""
    return (layout or DEFAULT_LAYOUT).card_to_index(card)


def index_to_card(index: int, layout: CardLayout | None = None) -> str:
    """0 -> 'As', 12 -> 'Ks', 13 -> 'Ac', ... (full deck by default)."""
    return (layout or DEFAULT_LAYOUT).index_to_card(index)


def parse_hand(cards: list[str], layout: CardLayout | None = None) -> list[int]:
    return (layout or DEFAULT_LAYOUT).parse_hand(cards)


@lru_cache(maxsize=8)
def get_utils(
    num_ranks: int = NUM_RANKS,
    num_suits: int = NUM_SUITS,
    hand_size: int = spiel_facts.GAME_PARAMS["hand_size"],
) -> gr.GinRummyUtils:
    return gr.GinRummyUtils(num_ranks, num_suits, hand_size)


@lru_cache(maxsize=8)
def layout_utils(layout: CardLayout, hand_size: int) -> gr.GinRummyUtils:
    return gr.GinRummyUtils(layout.num_ranks, layout.num_suits, hand_size)


def _utils_for(layout: CardLayout | None, hand_size: int | None) -> gr.GinRummyUtils:
    layout = layout or DEFAULT_LAYOUT
    size = spiel_facts.GAME_PARAMS["hand_size"] if hand_size is None else hand_size
    return layout_utils(layout, size)


def min_deadwood(
    hand: list[int], layout: CardLayout | None = None, hand_size: int | None = None
) -> int:
    """Optimal deadwood of a card set. The meld oracle."""
    return _utils_for(layout, hand_size).min_deadwood(hand)


def best_meld_group(
    hand: list[int], layout: CardLayout | None = None, hand_size: int | None = None
) -> list[list[int]]:
    """Meld set minimising deadwood. Laid verbatim at knock (meld_policy)."""
    return _utils_for(layout, hand_size).best_meld_group(hand)


def card_value(card: int, layout: CardLayout | None = None, hand_size: int | None = None) -> int:
    return _utils_for(layout, hand_size).card_value(card)


def meld_action_id(
    meld: list[int], layout: CardLayout | None = None, hand_size: int | None = None
) -> int:
    """Engine action id declaring this meld (MELD_ACTION_BASE + offset)."""
    return gr.MELD_ACTION_BASE + _utils_for(layout, hand_size).meld_to_int(meld)


def action_id_to_meld(
    action: int, layout: CardLayout | None = None, hand_size: int | None = None
) -> list[int]:
    return list(_utils_for(layout, hand_size).int_to_meld[action - gr.MELD_ACTION_BASE])


def deadwood_of_melded(
    hand: list[int],
    melds: list[list[int]],
    layout: CardLayout | None = None,
    hand_size: int | None = None,
) -> int:
    """Deadwood of a hand given the melds laid from it."""
    melded = {c for meld in melds for c in meld}
    helper = _utils_for(layout, hand_size)
    return sum(helper.card_value(c) for c in hand if c not in melded)


def reproduce_returns(
    terminal: dict[str, object],
    gin_bonus: int = spiel_facts.GAME_PARAMS["gin_bonus"],
    undercut_bonus: int = spiel_facts.GAME_PARAMS["undercut_bonus"],
    layout: CardLayout | None = None,
    hand_size: int | None = None,
) -> tuple[float, float]:
    """Recompute a finished hand's returns from our own deadwood numbers.

    Inputs are the engine terminal state's public record: which seat knocked,
    the laid melds, the remaining hands and the layoffs. Scoring:
    - no knock (wall/draw): 0-0;
    - gin (knocker deadwood 0): knocker scores defender deadwood + gin bonus;
    - undercut (defender deadwood <= knocker's): defender scores the
      difference + undercut bonus;
    - else the knocker scores the deadwood difference.
    Must equal state.returns() to the cent (gate-p1 meld oracle).
    """
    knocked = terminal["knocked"]
    assert isinstance(knocked, list)
    hands = terminal["hands"]
    assert isinstance(hands, list)
    layed = terminal["layed_melds"]
    assert isinstance(layed, list)
    layoffs = terminal["layoffs"]
    assert isinstance(layoffs, list)
    laid_off = {card_to_index(c, layout) for c in layoffs if isinstance(c, str)}

    if not any(knocked):
        return (0.0, 0.0)
    knocker = int(knocked.index(True))
    defender = 1 - knocker

    knocker_hand = parse_hand([c for c in hands[knocker] if c != "XX"], layout)
    knocker_melds = [parse_hand(m, layout) for m in layed[knocker]]
    knocker_dw = deadwood_of_melded(
        knocker_hand + [c for m in knocker_melds for c in m],
        knocker_melds,
        layout,
        hand_size,
    )

    defender_hand = parse_hand([c for c in hands[defender] if c != "XX"], layout)
    defender_dw = sum(card_value(c, layout, hand_size) for c in defender_hand if c not in laid_off)

    if knocker_dw == 0:
        score = defender_dw + gin_bonus
        return (score, -score) if knocker == 0 else (-score, score)
    # Verified against the engine (tie at 9-9 scores 0-0, not undercut+25):
    # undercut needs defender STRICTLY below the knocker; a tie scores the
    # (zero) difference for the knocker with no bonus.
    if defender_dw < knocker_dw:
        score = knocker_dw - defender_dw + undercut_bonus
        return (-score, score) if knocker == 0 else (score, -score)
    score = defender_dw - knocker_dw
    return (score, -score) if knocker == 0 else (-score, score)
