"""Game record: append-only JSONL round-trips into paired refit input."""

from __future__ import annotations

from pathlib import Path

from ginrl.agents.baselines import HeuristicAgent, RandomAgent
from ginrl.eval.arena import Arena
from ginrl.eval.ratings import fit_ratings
from ginrl.eval.record import append_summary, load_deals

arena = Arena()


def test_record_roundtrip_reproduces_ratings(tmp_path: Path) -> None:
    path = tmp_path / "game_record.jsonl"
    first = arena.duplicate_summary(HeuristicAgent(), RandomAgent(), seed=99, n_deals=8)
    second = arena.duplicate_summary(RandomAgent(), HeuristicAgent(), seed=100, n_deals=8)
    assert append_summary(path, first) == 16
    assert append_summary(path, second) == 16
    deals = load_deals(path)
    assert len(deals) == 16
    names = {HeuristicAgent().name, RandomAgent().name}
    assert all({d.a, d.b} == names for d in deals)
    assert all(len(d.legs) == 2 for d in deals)
    res = fit_ratings(deals, RandomAgent().name)
    assert res.n_deals == 16 and res.n_legs == 32
    assert res.rating[HeuristicAgent().name] > res.rating[RandomAgent().name]
