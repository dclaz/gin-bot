"""Style profiler: rates are coherent and replay-derived metrics work."""

from __future__ import annotations

from ginrl.agents.baselines import HeuristicAgent
from ginrl.agents.spiel_bots import SimpleGinRummyAgent
from ginrl.config import HandConfig, Seeds
from ginrl.env.game import HandEnv
from ginrl.eval.arena import Arena
from ginrl.eval.styles import TABLE_COLUMNS, profile, table_row

arena = Arena()


def _results(n: int = 12):  # type: ignore[no-untyped-def]
    out = []
    for i in range(n):
        out.append(arena.play_game(SimpleGinRummyAgent(), HeuristicAgent(), 6000 + i))
        out.append(arena.play_game(HeuristicAgent(), SimpleGinRummyAgent(), 6000 + i))
    return out


def _make_env() -> HandEnv:
    return HandEnv(HandConfig(), Seeds())


def test_profile_rates_are_coherent() -> None:
    results = _results()
    make_env = _make_env
    for name in ("simple_bot", HeuristicAgent().name):
        p = profile(name, results, make_env)
        assert p.n_hands == 2 * 12
        for rate in (
            p.gin_rate,
            p.knock_rate,
            p.undercut_rate,
            p.wall_rate,
            p.pile_draw_rate,
            p.danger_discard_rate,
        ):
            assert 0.0 <= rate <= 1.0, (name, rate)
        # Every hand ends exactly one way: gin, knock, undercut-knock, or wall.
        assert p.gin_rate + p.knock_rate + p.wall_rate <= 1.0 + 1e-9
        assert p.n_knocks > 0
        assert p.mean_turns_to_knock > 0
        assert p.n_draws > 0 and p.n_discards > 0
        assert len(p.deadwood_by_turn) == len(p.deadwood_by_turn_n) > 0
        assert all(v >= 0 for v in p.deadwood_by_turn)
        row = table_row(p)
        assert len(row) == len(TABLE_COLUMNS)
