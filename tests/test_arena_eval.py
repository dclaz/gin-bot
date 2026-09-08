"""Arena evaluation layer: summaries, refusal, replay, knock capture."""

from __future__ import annotations

import pytest

from ginrl.agents.baselines import HeuristicAgent, RandomAgent
from ginrl.agents.spiel_bots import SimpleGinRummyAgent
from ginrl.config import HandConfig, Seeds
from ginrl.env.game import HandEnv
from ginrl.eval.arena import Arena, replay
from ginrl.eval.stats import InsufficientDealsError

arena = Arena()


def test_pair_summary_points_per_hand_covers_truth() -> None:
    summary = arena.duplicate_summary(HeuristicAgent(), RandomAgent(), seed=4242, n_deals=60)
    assert summary.n_deals == 60 and summary.n_legs == 120
    assert len(summary.deal_seeds) == 60
    ppb = summary.points_per_hand()
    assert ppb.lo > 0, ppb  # heuristic is far stronger than random
    wins, losses, _ = summary.wins_losses_ties()
    assert wins > losses


def test_checked_refuses_below_resolved_count() -> None:
    summary = arena.duplicate_summary(HeuristicAgent(), RandomAgent(), seed=4243, n_deals=5)
    with pytest.raises(InsufficientDealsError):
        summary.checked(100)
    assert summary.checked(5) is summary


def test_replay_is_bit_exact() -> None:
    for seed in range(3000, 3005):
        result = arena.play_game(SimpleGinRummyAgent(), HeuristicAgent(), seed)
        env = HandEnv(HandConfig(), Seeds())
        assert replay(result, env) == result.returns


def test_knock_capture_is_consistent() -> None:
    knocks = walls = gins = 0
    for seed in range(3100, 3120):
        result = arena.play_game(SimpleGinRummyAgent(), SimpleGinRummyAgent(), seed)
        if result.wall:
            walls += 1
            assert result.knock_seat is None and result.knock_turns is None
            assert result.returns == (0.0, 0.0)
        else:
            knocks += 1
            assert result.knock_seat in (0, 1)
            assert result.knock_turns is not None and result.knock_turns >= 0
            assert result.deadwood is not None
            if result.gin_seat is not None:
                gins += 1
                assert result.deadwood[result.gin_seat] == 0
    assert knocks > 0
