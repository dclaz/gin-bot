"""Rule-based policy for the meld and layoff phases.

When the game asks which melds to lay or which cards to lay off, the answer
is a deterministic optimisation, not a strategic choice:

- knock discard: the discard minimising resulting deadwood (exact-optimal:
  the remaining hand is never played again, so only its value matters);
- melds: the best meld group, laid verbatim;
- layoffs: every legal layoff (each strictly reduces deadwood; order is
  irrelevant, so lay the highest-value card first).

This removes 185 of 241 actions from the learned policy. The engine's
legal_actions is the source of truth throughout; nothing here reimplements
legality. Phase 6 ablates letting the network choose melds.
"""

from __future__ import annotations

import pyspiel
from pyspiel import gin_rummy as gr

from ginrl.env import melds

_PASS = gr.PASS_ACTION


def choose_knock_discard(hand: list[int]) -> int:
    """Discard minimising the deadwood of the remaining 10 cards.

    Deterministic tie-break: lowest deadwood, then lowest card value,
    then lowest index.
    """
    return min(
        hand,
        key=lambda c: (melds.min_deadwood([h for h in hand if h != c]), melds.card_value(c), c),
    )


def _hand_of(state: pyspiel.State, seat: int) -> list[int]:
    return melds.parse_hand(state.to_dict()["hands"][seat])


def _laid_meld_cards(state: pyspiel.State, seat: int) -> set[int]:
    laid = state.to_dict()["layed_melds"][seat]
    return {melds.card_to_index(c) for meld in laid for c in meld}


def declare_meld_action(state: pyspiel.State) -> int:
    """One action for the player to act in the Knock (meld declaration) phase.

    Lays the best meld group computed over the full 10-card knocking hand.
    Returns PASS when every group meld is laid.
    """
    seat = state.current_player()
    legal = set(state.legal_actions())
    # Remaining hand plus already-laid cards reconstructs the full knocking
    # hand at any point of the declaration sequence, at any hand size.
    knocker_hand = _hand_of(state, seat) + sorted(_laid_meld_cards(state, seat))
    target_groups = melds.best_meld_group(knocker_hand)
    laid = _laid_meld_cards(state, seat)
    for group in target_groups:
        action = melds.meld_action_id(group)
        if action in legal and not set(group) <= laid:
            return action
    # Target groups all laid (or uncomputable): declare anything left, else pass.
    for action in sorted(legal):
        if action >= gr.MELD_ACTION_BASE:
            return action
    return _PASS


def layoff_action(state: pyspiel.State) -> int:
    """One action for the defender in the Layoff phase: best layoff, else pass."""
    legal = [a for a in state.legal_actions() if a != _PASS]
    if not legal:
        return _PASS
    return max(legal, key=lambda a: (melds.card_value(a), -a))


def next_auto_action(state: pyspiel.State) -> int | None:
    """One automatic action, or None at terminal/chance/learned phases.

    The env applies the returned action and publishes it to the
    BeliefTrackers as a public event. Knock-declaration and Layoff and Wall
    phases are automatic; the learned KNOCK decision itself happens at the
    Discard phase (action KNOCK_ACTION) and is not automatic.
    """
    if state.is_terminal() or state.is_chance_node():
        return None
    phase = state.to_dict()["phase"]
    if phase == "Knock":
        legal = state.legal_actions()
        if any(a >= gr.MELD_ACTION_BASE for a in legal):
            return declare_meld_action(state)
        discards = [a for a in legal if a != _PASS]
        if discards:
            hand = _hand_of(state, state.current_player())
            choice = choose_knock_discard(hand)
            return choice if choice in discards else sorted(discards)[0]
        return _PASS
    if phase == "Layoff":
        return layoff_action(state)
    if phase == "Wall":
        return _PASS
    return None
