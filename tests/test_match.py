"""Exact-fixture tests for the match wrapper (Phase 4).

Policies are scripted (first legal action; knock when legal), so every
number below is bit-exact given the seed. The fixtures pin: score
accumulation across hands, target-score termination with winner/capped,
the max_hands cap path (capped=True, leader wins, ties go to seat 0),
per-hand index reporting, and duplicate-seed replay.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from ginrl.config import REDUCED_HAND_CONFIG, MatchConfig, Seeds
from ginrl.env.match import MatchEnv


def _run(seed: int, knock: bool = False) -> list[tuple]:
    cfg = MatchConfig(hand=REDUCED_HAND_CONFIG, target_score=30, max_hands=50)
    env = MatchEnv(config=cfg, seeds=Seeds(master=7))
    r = env.reset(seed=seed)
    traj = []
    while True:
        legal = [i for i, m in enumerate(r.hand.mask) if m]
        a = 55 if (knock and 55 in legal) else legal[0]
        r = env.step(a)
        if r.hand_returns is not None:
            traj.append(
                (
                    r.hand_index,
                    r.hand_returns,
                    r.scores,
                    r.done,
                    r.winner,
                    r.capped,
                )
            )
        if r.done:
            break
    return traj


def test_knock_policy_exact_trajectory() -> None:
    traj = _run(3, knock=True)
    assert traj == [
        (0, (1.0, -1.0), (1.0, 0.0), False, -1, False),
        (1, (35.0, -35.0), (36.0, 0.0), True, 0, False),
    ]


def test_cap_path_reports_capped_and_tie_goes_to_seat_zero() -> None:
    # First-legal policy draws forever: every hand is (0,0), so the match
    # must end on the max_hands cap with the tie broken to seat 0.
    traj = _run(3, knock=False)
    assert len(traj) == 50
    assert all(t[1] == (0.0, 0.0) for t in traj)
    assert [t[0] for t in traj] == list(range(50))
    last = traj[-1]
    assert last[3] is True and last[4] == 0 and last[5] is True
    assert last[2] == (0.0, 0.0)


def test_duplicate_seed_replays_identical_hands() -> None:
    assert _run(3, knock=True) == _run(3, knock=True)
    assert _run(11, knock=True) == _run(11, knock=True)
    assert _run(3, knock=True) != _run(11, knock=True)
