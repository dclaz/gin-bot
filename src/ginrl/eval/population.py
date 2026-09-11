"""Population ranking: Nash averaging (max-entropy Nash of the meta-game).

Alpha-Rank is not clone invariant and reads only matchup signs, so the
headline population ranking is Nash averaging (Balduzzi et al. 2018 §4):
the maximum-entropy Nash equilibrium of the symmetric zero-sum meta-game.
Alpha-Rank is a Tier 3 evolutionary lens, off by default and not built here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize


@dataclass(frozen=True)
class NashMixture:
    weights: dict[str, float]
    value: float
    max_regret: float  # max over rows of (P p - v); ~0 verifies the solve
    entropy: float
    zero_sum_residual: float  # max |P + P^T| before symmetrisation


def nash_averaging(names: list[str], payoff: np.ndarray) -> NashMixture:
    """Max-entropy Nash of the symmetric zero-sum game defined by `payoff`.

    `payoff[i, j]` is mean points for i against j. The harness mirrors seats,
    so the input should already be near-antisymmetric; `zero_sum_residual`
    checks that for free, then the matrix is explicitly symmetrised.
    """
    p = np.asarray(payoff, dtype=np.float64)
    if p.shape != (len(names), len(names)):
        raise ValueError(f"payoff shape {p.shape} != ({len(names)}, {len(names)})")
    residual = float(np.abs(p + p.T).max())
    a = (p - p.T) / 2  # symmetric zero-sum game; value is exactly 0
    n = len(names)

    # Stage 1: game value via LP (sanity: ~0 for a symmetrised matrix).
    # max v s.t. A^T q >= v 1, sum q = 1, q >= 0.
    c = np.zeros(n + 1)
    c[-1] = -1.0
    a_ub = np.zeros((n, n + 1))
    a_ub[:, :n] = -a.T
    a_ub[:, n] = 1.0
    b_ub = np.zeros(n)
    a_eq = np.zeros((1, n + 1))
    a_eq[0, :n] = 1.0
    lp = linprog(
        c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=[(0, None)] * n + [(None, None)]
    )
    if not lp.success:
        raise RuntimeError(f"meta-game LP failed: {lp.message}")
    value = float(lp.x[-1])

    # Stage 2: max entropy s.t. A p <= value (no profitable deviation).
    def neg_entropy(q: np.ndarray) -> float:
        q = np.clip(q, 1e-12, 1.0)
        return float(q @ np.log(q))

    cons = [
        {"type": "eq", "fun": lambda q: float(np.sum(q)) - 1.0},
        {"type": "ineq", "fun": lambda q, a=a: value + 1e-9 - (a @ q)},
    ]

    def _solve(x0: np.ndarray) -> object:
        return minimize(
            neg_entropy,
            x0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=cons,
            options={"maxiter": 10000, "ftol": 1e-9},
        )

    # ftol 1e-12 starved the solver on noisy matrices (a 20-deal smoke hit
    # "Iteration limit reached"); 1e-9 is ~1e-9 nats off the max-entropy
    # point, far below any reporting precision. Second start from the LP
    # vertex covers infeasible-start stalls; both must fail to raise.
    sol = _solve(np.full(n, 1.0 / n))
    if not sol.success:
        vertex = np.zeros(n)
        vertex[int(np.argmax(lp.x[:n]))] = 1.0
        sol = _solve(vertex)
    if not sol.success:
        raise RuntimeError(f"max-entropy Nash failed: {sol.message}")
    w = np.clip(sol.x, 0.0, 1.0)
    w /= w.sum()
    regret = float((a @ w).max() - value)
    entropy = float(-(w[w > 0] @ np.log(w[w > 0]))) if n > 1 else 0.0
    return NashMixture(
        weights={name: float(w[i]) for i, name in enumerate(names)},
        value=value,
        max_regret=regret,
        entropy=entropy,
        zero_sum_residual=residual,
    )


def clone_drift(names: list[str], payoff: np.ndarray, clone: str) -> dict[str, float]:
    """Per-strategy Nash-mass drift from duplicating `clone` (clone check).

    Returns {other_name: |mass_after - mass_before|} over strategies other
    than the clone and its copy. Near-zero is the clone-invariance claim.
    """
    base = nash_averaging(names, payoff)
    k = names.index(clone)
    ext_names = names + [f"{clone}'"]
    p = np.asarray(payoff, dtype=np.float64)
    ext = np.zeros((len(ext_names), len(ext_names)))
    ext[: len(names), : len(names)] = p
    ext[-1, : len(names)] = p[k, :]
    ext[: len(names), -1] = p[:, k]
    dup = nash_averaging(ext_names, ext)
    drift = {}
    for name in names:
        if name == clone:
            continue
        drift[name] = abs(dup.weights[name] - base.weights[name])
    return drift
