"""Nash averaging: uniform on cycles, zero mass on dominated, clone-invariant."""

from __future__ import annotations

import numpy as np

from ginrl.eval.population import clone_drift, nash_averaging


def test_rps_mixture_is_uniform() -> None:
    names = ["rock", "paper", "scissors"]
    payoff = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    mix = nash_averaging(names, payoff)
    for name in names:
        assert abs(mix.weights[name] - 1 / 3) <= 0.02, mix.weights
    assert abs(mix.value) < 1e-6
    assert mix.max_regret < 1e-6
    assert mix.zero_sum_residual < 1e-12


def test_dominated_strategy_gets_no_mass() -> None:
    names = ["strong", "weak", "dominated"]
    payoff = np.array([[0.0, 2.0, 5.0], [-2.0, 0.0, 4.0], [-5.0, -4.0, 0.0]])
    mix = nash_averaging(names, payoff)
    assert mix.weights["dominated"] < 1e-6, mix.weights
    assert mix.weights["strong"] > mix.weights["weak"]
    assert mix.max_regret < 1e-6


def test_cloning_moves_no_other_mass() -> None:
    names = ["a", "b", "c"]
    payoff = np.array([[0.0, 1.0, -0.5], [-1.0, 0.0, 2.0], [0.5, -2.0, 0.0]])
    drift = clone_drift(names, payoff, "a")
    assert set(drift) == {"b", "c"}
    for name, delta in drift.items():
        assert delta < 1e-6, (name, delta)


def test_noisy_matrix_converges() -> None:
    """A noisy near-tied 7x7 (20-deal-style smoke) must solve, not stall.

    Regression: SLSQP with maxiter=1000/ftol=1e-12 raised 'Iteration limit
    reached' here. The solve quality bar is max_regret, not iterations."""
    rng = np.random.default_rng(11)
    base = rng.normal(0.0, 3.0, size=(7, 7))
    payoff = (base - base.T) / 2
    names = [f"s{i}" for i in range(7)]
    mix = nash_averaging(names, payoff)
    assert abs(sum(mix.weights.values()) - 1.0) < 1e-6
    assert all(w >= -1e-9 for w in mix.weights.values())
    assert mix.max_regret < 1e-4, mix.max_regret


def test_seat_mirror_residual_is_checked() -> None:
    names = ["x", "y"]
    bad = np.array([[0.0, 3.0], [1.0, 0.0]])  # not antisymmetric
    mix = nash_averaging(names, bad)
    assert mix.zero_sum_residual == 4.0
    assert mix.max_regret < 1e-6  # still solves the symmetrised game
