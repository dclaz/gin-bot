"""Meld oracle: GinRummyUtils adapter, card codec, and returns reproduction."""

from __future__ import annotations

import random

import pyspiel
import pytest
from pyspiel import gin_rummy as gr

from ginrl.config import HandConfig, Seeds
from ginrl.env import melds
from ginrl.env.game import HandEnv
from ginrl.env.melds import CardLayout


def test_card_codec_round_trips_all_52() -> None:
    for i in range(52):
        assert melds.card_to_index(melds.index_to_card(i)) == i
    assert melds.card_to_index("As") == 0
    assert melds.card_to_index("Ks") == 12
    assert melds.card_to_index("Ac") == 13
    assert melds.index_to_card(51) == "Kh"


def test_reduced_layout_codec_and_melds() -> None:
    layout = CardLayout(num_ranks=5, num_suits=2)
    assert layout.deck_size == 10
    assert layout.card_to_index("Ac") == 5
    assert layout.index_to_card(9) == "5c"
    assert layout.parse_hand(["As", "2s", "3s"]) == [0, 1, 2]
    assert melds.min_deadwood([0, 1, 2], layout, 3) == 0


def test_min_deadwood_known_hands() -> None:
    # 6c7c8c + 7d8d9dTd + 3c 2d 5h -> 3+2+5.
    hand = melds.parse_hand(["6c", "7c", "8c", "7d", "8d", "9d", "Td", "3c", "2d", "5h"])
    assert melds.min_deadwood(hand) == 10
    gin = melds.parse_hand(["6c", "7c", "8c", "7d", "8d", "9d", "Td", "Tc", "Th", "Ts"])
    assert melds.min_deadwood(gin) == 0


def test_best_meld_group_finds_both_melds() -> None:
    hand = melds.parse_hand(["6c", "7c", "8c", "7d", "8d", "9d", "Td", "3c", "2d", "5h"])
    groups = melds.best_meld_group(hand)
    flat = sorted(c for m in groups for c in m)
    assert flat == sorted(melds.parse_hand(["6c", "7c", "8c", "7d", "8d", "9d", "Td"]))
    assert melds.deadwood_of_melded(hand, groups) == 10


def test_meld_action_id_round_trips() -> None:
    meld = melds.parse_hand(["6c", "7c", "8c"])
    action = melds.meld_action_id(meld)
    assert gr.MELD_ACTION_BASE <= action < gr.MELD_ACTION_BASE + gr.NUM_MELD_ACTIONS
    assert sorted(melds.action_id_to_meld(action)) == sorted(meld)


def _terminal(
    knocked: list[bool],
    hands: list[list[str]],
    layed: list[list[list[str]]],
    layoffs: list[str],
) -> dict[str, object]:
    return {"knocked": knocked, "hands": hands, "layed_melds": layed, "layoffs": layoffs}


def test_reproduce_returns_wall() -> None:
    t = _terminal([False, False], [["As"], ["Ks"]], [[], []], [])
    assert melds.reproduce_returns(t) == (0.0, 0.0)


def test_reproduce_returns_knock() -> None:
    # Knocker (seat 1): melds laid, 3c 2d 5h left = 10. Defender: 44 - 9c layoff.
    t = _terminal(
        [False, True],
        [["5s", "6s", "Js", "Ac", "2c", "Tc", "5d", "Ah", "4h"], ["3c", "2d", "5h"]],
        [[], [["6c", "7c", "8c"], ["7d", "8d", "9d", "Td"]]],
        ["9c"],
    )
    assert melds.reproduce_returns(t) == (-34.0, 34.0)


def test_reproduce_returns_gin() -> None:
    # Knocker fully melded (deadwood 0): defender 39 + gin bonus 25.
    t = _terminal(
        [True, False],
        [[], ["Ks", "Qd", "Jh", "9s"]],
        [[["6c", "7c", "8c"], ["7d", "8d", "9d", "Td"], ["Tc", "Th", "Ts"]], []],
        [],
    )
    assert melds.reproduce_returns(t) == (64.0, -64.0)


def test_reproduce_returns_undercut() -> None:
    # Knocker deadwood 8, defender 5 after layoffs -> defender scores 3 + 25.
    t = _terminal(
        [True, False],
        [["5h", "3c"], ["2c", "3d"]],
        [[], []],
        [],
    )
    assert melds.reproduce_returns(t) == (-28.0, 28.0)


def test_reproduce_returns_tied_deadwood_scores_zero() -> None:
    # Engine-verified (greedy seed 60294): 9-9 tie is 0-0, not undercut+25.
    t = _terminal(
        [True, False],
        [["3c", "Ad", "2d", "3h"], ["2c", "3d", "4h"]],
        [[["5c", "5d", "5h"], ["6s", "6d", "6h"]], [["As", "2s", "3s"], ["8c", "8d", "8h"]]],
        ["5s"],
    )
    assert melds.reproduce_returns(t) == (0.0, 0.0)


def _knock_terminals(env: HandEnv, rng: random.Random, count: int, start_seed: int):
    """(terminal public record, engine returns) pairs with a knock."""
    found = []
    seed = start_seed
    while len(found) < count:
        r = env.reset(seed=seed)
        seed += 1
        while not r.done:
            legal = [a for a, m in enumerate(r.mask) if m]
            action = 55 if r.mask[55] else rng.choice(legal)
            r = env.step(action)
        assert r.returns is not None
        if r.returns != (0.0, 0.0):
            found.append((env.state().to_dict(), r.returns))
    return found


def test_reproduce_returns_matches_engine_on_knock_terminals() -> None:
    env = HandEnv(HandConfig(), Seeds(11))
    rng = random.Random(11)
    for terminal, engine_returns in _knock_terminals(env, rng, 15, 500):
        assert melds.reproduce_returns(terminal) == pytest.approx(engine_returns)


def test_action_round_trip_all_241() -> None:
    game = pyspiel.load_game("gin_rummy")
    state = game.new_initial_state()
    rng = random.Random(0)
    while state.is_chance_node():
        outs, probs = zip(*state.chance_outcomes(), strict=True)
        state.apply_action(rng.choices(outs, list(probs))[0])
    for action in range(game.num_distinct_actions()):
        assert state.action_to_string(action)  # every id stringifies
    # Reduced 56-action set is the identity on 0..55.
    from ginrl.env.game import N_LEARNED_ACTIONS

    assert N_LEARNED_ACTIONS == 56
