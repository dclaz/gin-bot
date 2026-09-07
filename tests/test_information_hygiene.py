"""Information hygiene: features must not leak hidden state.

For states sampled from real play, resample the hidden information 32 times
with resample_from_infostate (which provably varies the opponent's hand —
asserted below so the test cannot pass vacuously) and require the feature
vector to be bit-identical across resamples. Any variation is a leak of the
opponent's hand into our features.
"""

from __future__ import annotations

import random

import numpy as np

from ginrl.config import Seeds
from ginrl.env.game import HandEnv
from ginrl.env.melds import parse_hand


def test_features_bit_identical_across_resamples() -> None:
    env = HandEnv(seeds=Seeds(31))
    rng = random.Random(31)
    checked = 0
    for seed in range(4):
        r = env.reset(seed=seed)
        points = 0
        while not r.done and points < 8:
            # Test each point immediately: tracker history is exactly this prefix.
            state = env.state().clone()
            seat = state.current_player()
            base = env.trackers[seat].features(state.to_observation_struct(seat).to_dict(), None)
            opp_hands = set()
            for _ in range(8):
                rs = state.resample_from_infostate(seat, lambda: rng.random())
                d = rs.to_dict()
                opp_hands.add(tuple(sorted(parse_hand(d["hands"][1 - seat]))))
                obs = rs.to_observation_struct(seat).to_dict()
                assert np.array_equal(env.trackers[seat].features(obs, None), base), (
                    "feature leak: vector varies across resamples of one infostate"
                )
            # Non-vacuous: the opponent's hand actually varied.
            assert len(opp_hands) > 1, "resample never varied the hidden hand"
            checked += 1
            points += 1
            legal = [a for a, m in enumerate(r.mask) if m]
            r = env.step(rng.choice(legal))
    assert checked > 0


def test_tracker_history_matches_play_prefix() -> None:
    """The features used above come from the real play prefix, not thin air."""
    env = HandEnv(seeds=Seeds(32))
    rng = random.Random(32)
    r = env.reset(seed=0)
    turns = 0
    while not r.done and turns < 4:
        legal = [a for a, m in enumerate(r.mask) if m]
        r = env.step(rng.choice(legal))
        turns += 1
    for seat in (0, 1):
        assert env.trackers[seat]._turn == turns
