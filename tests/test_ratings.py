"""Anchored Bradley-Terry ratings: recovery, coverage, cycles, edge, prior."""

from __future__ import annotations

import numpy as np

from ginrl.eval.ratings import (
    ELO_SCALE,
    DealPair,
    LegRecord,
    elo_duel_prob,
    fit_ratings,
    synthetic_deals,
)

ANCHOR = "anchor"


def _rps_deals(q: float, n: int, seed: int) -> list[DealPair]:
    beats = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
    names = ["rock", "paper", "scissors"]
    rng = np.random.RandomState(seed)
    deals = []
    for x in range(3):
        for y in range(x + 1, 3):
            a, b = names[x], names[y]
            for _ in range(n):
                legs = []
                score = 0.0
                for leg in range(2):
                    a_first = leg == 0
                    first = a if a_first else b
                    fav = a if beats[a] == b else b
                    winner = fav if rng.rand() < q else (b if fav == a else a)
                    won_a = winner == a
                    legs.append(LegRecord(a=a, b=b, a_first=a_first, first_won=(winner == first)))
                    score += 1.0 if won_a else -1.0
                deals.append(DealPair(a=a, b=b, score_a=score, legs=(legs[0], legs[1])))
    return deals


def test_recovery_within_tolerance() -> None:
    truth = {ANCHOR: 0.0, "mid": 100.0, "strong": 200.0}
    deals = synthetic_deals(truth, 200, seed=7)
    res = fit_ratings(deals, ANCHOR)
    assert res.elo(ANCHOR) == 0.0
    for name, elo in truth.items():
        if name == ANCHOR:
            continue
        assert abs(res.elo(name) - elo) <= 25, (name, res.elo(name), elo)


def test_ci_covers_at_nominal_rate() -> None:
    truth = {ANCHOR: 0.0, "mid": 80.0, "strong": 160.0}
    hits = trials = 0
    for seed in range(40):
        res = fit_ratings(synthetic_deals(truth, 150, seed), ANCHOR)
        for name, elo in truth.items():
            if name == ANCHOR:
                continue
            trials += 1
            lo, hi = res.elo_ci(name)
            hits += lo <= elo <= hi
    assert hits / trials >= 0.90, f"coverage {hits}/{trials}"


def test_fit_is_order_invariant() -> None:
    deals = synthetic_deals({ANCHOR: 0.0, "mid": 100.0}, 100, seed=3)
    rng = np.random.RandomState(0)
    shuffled = list(deals)
    rng.shuffle(shuffled)
    first = fit_ratings(deals, ANCHOR)
    second = fit_ratings(shuffled, ANCHOR)
    assert first.rating == second.rating


def test_rps_is_cyclic_and_ratings_are_fiction() -> None:
    res = fit_ratings(_rps_deals(0.9, 500, seed=0), "rock")
    assert abs(res.cyclic_fraction - 0.9994) <= 0.01
    elos = sorted(res.elo(n) for n in res.names)
    assert max(elos) - min(elos) < 15  # Elo cannot see the cycle


def test_transitive_ladder_is_not_cyclic() -> None:
    truth = {"a": 0.0, "b": 100.0, "c": 200.0, "d": 300.0}
    res = fit_ratings(synthetic_deals(truth, 200, seed=11), "a")
    assert abs(res.cyclic_fraction - 0.0006) <= 0.01


def test_undefeated_agent_gets_finite_rating() -> None:
    deals = []
    for _ in range(30):
        legs = (
            LegRecord(a="champ", b=ANCHOR, a_first=True, first_won=True),
            LegRecord(a="champ", b=ANCHOR, a_first=False, first_won=False),
        )
        deals.append(DealPair(a="champ", b=ANCHOR, score_a=20.0, legs=legs))
    res = fit_ratings(deals, ANCHOR)
    elo = res.elo("champ")
    assert np.isfinite(elo)
    assert 400 < elo < 800  # L2 prior keeps 30-0 strong but sane


def test_self_play_fit_measures_only_the_edge() -> None:
    rng = np.random.RandomState(0)
    deals = []
    for _ in range(200):
        u = rng.rand()
        first_won = None if u < 0.1 else u < 0.55
        legs = (
            LegRecord(a="x", b="x", a_first=True, first_won=first_won),
            LegRecord(a="x", b="x", a_first=False, first_won=first_won),
        )
        deals.append(DealPair(a="x", b="x", score_a=0.0, legs=legs))
    res = fit_ratings(deals, "x")
    assert res.rating == {"x": 0.0} and res.cyclic_fraction == 0.0
    lo, hi = res.edge_elo_ci()
    assert lo <= 0 <= hi, (lo, hi)


def test_symmetric_play_has_no_edge() -> None:
    deals = synthetic_deals({ANCHOR: 0.0, "opp": 50.0}, 300, seed=5, edge_logit=0.0)
    res = fit_ratings(deals, ANCHOR)
    lo, hi = res.edge_elo_ci()
    assert lo <= 0 <= hi, (lo, hi)


def test_edge_recovery() -> None:
    deals = synthetic_deals({ANCHOR: 0.0, "opp": 0.0}, 400, seed=9, edge_logit=100.0 / ELO_SCALE)
    res = fit_ratings(deals, ANCHOR)
    assert abs(res.edge_elo - 100.0) <= 25, res.edge_elo


def test_win_prob_is_sigmoid_of_rating_gap() -> None:
    deals = synthetic_deals({ANCHOR: 0.0, "mid": 100.0}, 100, seed=3)
    res = fit_ratings(deals, ANCHOR)
    assert res.win_prob("mid", ANCHOR) > 0.5
    assert abs(res.win_prob(ANCHOR, "mid") + res.win_prob("mid", ANCHOR) - 1.0) < 1e-12
    assert 0.0 < res.win_prob("mid", ANCHOR) < 1.0


def test_elo_duel_prob_matches_logistic() -> None:
    assert abs(elo_duel_prob(0.0, 0.0) - 0.5) < 1e-12
    assert abs(elo_duel_prob(400.0, 0.0) - 10 / 11) < 1e-9


def test_large_star_record_fits() -> None:
    """Solver regression: a large star-graph record (every pair involves the
    champ, as our ladder records do) stalled the numeric-Hessian Newton in
    both stages — the deflected final step lands at gnorm ~= 1e-4, where the
    absolute Armijo decrease is below float resolution of the ~1e4-magnitude
    objective at every alpha. The closed-form Hessian converges quadratically
    past that band. This input raised 'stalled above tolerance' before it.
    """
    elos = {
        "anchor": 0.0,
        "champ": -510.0,
        "c0": -80.0,
        "c1": -190.0,
        "c2": -260.0,
        "c3": -60.0,
        "c4": -420.0,
    }
    deals = []
    for opp in [n for n in elos if n not in ("anchor", "champ")]:
        tri = {"anchor": 0.0, "champ": elos["champ"], opp: elos[opp]}
        for deal in synthetic_deals(tri, 1500, seed=1):
            if "champ" in (deal.a, deal.b):
                deals.append(deal)
    assert len(deals) == 15000
    res = fit_ratings(deals, "anchor")
    assert abs(res.elo("champ") + 510.0) <= 60, res.elo("champ")
    assert abs(res.edge_elo) <= 20, res.edge_elo
