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

from functools import lru_cache

from pyspiel import gin_rummy as gr

from ginrl import spiel_facts

RANKS = "A23456789TJQK"
SUITS = "scdh"
NUM_RANKS: int = spiel_facts.GAME_PARAMS["num_ranks"]
NUM_SUITS: int = spiel_facts.GAME_PARAMS["num_suits"]
NUM_CARDS = NUM_SUITS * NUM_RANKS


def card_to_index(card: str) -> int:
    """'As' -> 0, 'Ks' -> 12, 'Ac' -> 13, ..."""
    return SUITS.index(card[1]) * NUM_RANKS + RANKS.index(card[0])


def index_to_card(index: int) -> str:
    """0 -> 'As', 12 -> 'Ks', 13 -> 'Ac', ..."""
    suit, rank = divmod(index, NUM_RANKS)
    return f"{RANKS[rank]}{SUITS[suit]}"


def parse_hand(cards: list[str]) -> list[int]:
    return [card_to_index(c) for c in cards]


@lru_cache(maxsize=8)
def get_utils(
    num_ranks: int = NUM_RANKS,
    num_suits: int = NUM_SUITS,
    hand_size: int = spiel_facts.GAME_PARAMS["hand_size"],
) -> gr.GinRummyUtils:
    return gr.GinRummyUtils(num_ranks, num_suits, hand_size)


def min_deadwood(hand: list[int]) -> int:
    """Optimal deadwood of a card set. The meld oracle."""
    return get_utils().min_deadwood(hand)


def best_meld_group(hand: list[int]) -> list[list[int]]:
    """Meld set minimising deadwood. Laid verbatim at knock (meld_policy)."""
    return get_utils().best_meld_group(hand)


def card_value(card: int) -> int:
    return get_utils().card_value(card)


def meld_action_id(meld: list[int]) -> int:
    """Engine action id declaring this meld (MELD_ACTION_BASE + offset)."""
    return gr.MELD_ACTION_BASE + get_utils().meld_to_int(meld)


def action_id_to_meld(action: int) -> list[int]:
    return list(get_utils().int_to_meld[action - gr.MELD_ACTION_BASE])


def deadwood_of_melded(hand: list[int], melds: list[list[int]]) -> int:
    """Deadwood of a hand given the melds laid from it."""
    melded = {c for meld in melds for c in meld}
    return sum(card_value(c) for c in hand if c not in melded)


def reproduce_returns(
    terminal: dict[str, object],
    gin_bonus: int = spiel_facts.GAME_PARAMS["gin_bonus"],
    undercut_bonus: int = spiel_facts.GAME_PARAMS["undercut_bonus"],
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
    laid_off = {card_to_index(c) for c in layoffs if isinstance(c, str)}

    if not any(knocked):
        return (0.0, 0.0)
    knocker = int(knocked.index(True))
    defender = 1 - knocker

    knocker_hand = parse_hand([c for c in hands[knocker] if c != "XX"])
    knocker_melds = [parse_hand(m) for m in layed[knocker]]
    knocker_dw = deadwood_of_melded(
        knocker_hand + [c for m in knocker_melds for c in m], knocker_melds
    )

    defender_hand = parse_hand([c for c in hands[defender] if c != "XX"])
    defender_dw = sum(card_value(c) for c in defender_hand if c not in laid_off)

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
