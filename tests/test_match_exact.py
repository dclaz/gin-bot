"""Match wrapper against an independent implementation (Phase 4 exact check).

The reference below reimplements match scoring, target termination, and cap
accounting in plain test code driving HandEnv directly — it shares no code
with MatchEnv. Agreement is bit-exact per seed (not statistical): any wrapper
distortion in totals, termination, winner, capped, or hand indexing fails.
Near-target edges (exact hit, overshoot) and cap edges (leader wins, tie to
seat 0) are covered by sweeping target and max_hands.
"""

from __future__ import annotations

import random
import sys

sys.path.insert(0, "src")

from ginrl.config import REDUCED_HAND_CONFIG, Seeds
from ginrl.env.game import HandEnv
from ginrl.env.match import MatchEnv


def _reference(
    hand_cfg, master: int, base_seed: int, target: int, max_hands: int, knock: bool, rng_seed: int
):
    """Independent match loop. Returns (hands, scores, done_hand, winner, capped)."""
    rng = random.Random(rng_seed)
    scores = [0.0, 0.0]
    hands = []
    for hi in range(max_hands):
        env = HandEnv(hand_cfg, Seeds(master))
        r = env.reset(seed=base_seed * 100003 + hi)
        while not r.done:
            legal = [a for a, m in enumerate(r.mask) if m]
            if knock and 55 in legal:
                a = 55
            elif knock:
                a = rng.choice(legal)
            else:
                a = legal[0]
            r = env.step(a)
        ret = (float(r.returns[0]), float(r.returns[1]))
        hands.append(ret)
        scores[0] += max(ret[0], 0.0)
        scores[1] += max(ret[1], 0.0)
        leaders = [i for i, s in enumerate(scores) if s >= target]
        if leaders:
            winner = max(leaders, key=lambda i: scores[i])
            return hands, tuple(scores), hi, winner, False
    winner = 0 if scores[0] >= scores[1] else 1
    return hands, tuple(scores), max_hands - 1, winner, True


def _wrapper(
    hand_cfg, master: int, base_seed: int, target: int, max_hands: int, knock: bool, rng_seed: int
):
    from ginrl.config import MatchConfig

    rng = random.Random(rng_seed)
    cfg = MatchConfig(hand=hand_cfg, target_score=target, max_hands=max_hands)
    env = MatchEnv(config=cfg, seeds=Seeds(master))
    r = env.reset(seed=base_seed)
    hands = []
    while True:
        legal = [i for i, m in enumerate(r.hand.mask) if m]
        if knock and 55 in legal:
            a = 55
        elif knock:
            a = rng.choice(legal)
        else:
            a = legal[0]
        r = env.step(a)
        if r.hand_returns is not None:
            hands.append((r.hand_index, tuple(r.hand_returns)))
        if r.done:
            return hands, r.scores, r.hand_index, r.winner, r.capped


def _check(master, base_seed, target, max_hands, knock):
    ref_hands, ref_scores, ref_hi, ref_winner, ref_capped = _reference(
        REDUCED_HAND_CONFIG, master, base_seed, target, max_hands, knock, 1234
    )
    got_hands, got_scores, got_hi, got_winner, got_capped = _wrapper(
        REDUCED_HAND_CONFIG, master, base_seed, target, max_hands, knock, 1234
    )
    assert [h for _, h in got_hands] == list(ref_hands), (target, max_hands, base_seed)
    assert [i for i, _ in got_hands] == list(range(len(ref_hands)))
    assert tuple(got_scores) == ref_scores
    assert got_hi == ref_hi
    assert got_winner == ref_winner
    assert got_capped == ref_capped
    # Invariants: scores never go negative; only the winner's points count.
    assert got_scores[0] >= 0 and got_scores[1] >= 0
    return ref_capped


def test_wrapper_matches_reference_across_edges() -> None:
    caps = 0
    n = 0
    for target in (1, 2, 30, 100, 10_000):
        for max_hands in (1, 2, 7):
            for knock in (False, True):
                for base_seed in (0, 3, 11):
                    caps += _check(7, base_seed, target, max_hands, knock)
                    n += 1
    assert n == 5 * 3 * 2 * 3
    assert caps > 0  # the cap path was actually exercised


def test_overshoot_ends_match_not_at_target() -> None:
    # Knock-first/rng policy ends the match over target on hand 1: the
    # recorded total is the true overshoot (35, 6), not clipped to 30.
    _, scores, hi, winner, capped = _wrapper(REDUCED_HAND_CONFIG, 7, 3, 30, 50, True, 1234)
    assert (tuple(scores), hi, winner, capped) == ((35.0, 6.0), 1, 0, False)
