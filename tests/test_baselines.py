"""Heuristic baselines: sane draws, no pile-cycling engine draws."""

from __future__ import annotations

from pyspiel import gin_rummy as gr

from ginrl.agents.baselines import HeuristicAgent, HeuristicParams
from ginrl.eval.arena import Arena

arena = Arena()


def test_heuristic_draws_stock_and_knocks_against_itself() -> None:
    """Regression: the draw fallthrough used to take the pile unconditionally
    (action 52 sorts before 53), so heuristic mirrors pile-cycled into
    engine no-progress draws ~99% of the time."""
    summary = arena.duplicate_summary(
        HeuristicAgent(HeuristicParams(knock_threshold=7)),
        HeuristicAgent(HeuristicParams(knock_threshold=9)),
        seed=3,
        n_deals=20,
    )
    legs = [leg for p in summary.pairs for leg in (p.leg1, p.leg2)]
    walls = sum(1 for leg in legs if leg.wall)
    assert walls / len(legs) < 0.5, f"{walls}/{len(legs)} walls"
    stock = sum(
        1
        for leg in legs
        for rec in leg.actions
        if not rec.manual and rec.action == gr.DRAW_STOCK_ACTION
    )
    assert stock > 0, "heuristic never draws from the stock"
    knocks = sum(1 for leg in legs if leg.knock_seat is not None)
    assert knocks > len(legs) / 2, f"only {knocks}/{len(legs)} knocks"
