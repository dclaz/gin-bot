"""Paired statistics and sample-size discipline for evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class InsufficientDealsError(Exception):
    """Raised when a report is requested below the resolved sample count."""


@dataclass(frozen=True)
class Summary:
    mean: float
    lo: float
    hi: float
    n: int


def bootstrap_ci(
    values: list[float] | np.ndarray, *, alpha: float = 0.05, draws: int = 2000, seed: int = 0
) -> Summary:
    """Percentile bootstrap CI of the mean. Deterministic given seed."""
    arr = np.asarray(values, dtype=np.float64)
    assert len(arr) > 0
    rng = np.random.RandomState(seed)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(draws)])
    return Summary(
        mean=float(arr.mean()),
        lo=float(np.percentile(means, 100 * alpha / 2)),
        hi=float(np.percentile(means, 100 * (1 - alpha / 2))),
        n=len(arr),
    )


def required_deals(pilot: list[float], delta: float, *, alpha: float = 0.05) -> int:
    """Duplicate deals needed to resolve `delta` points/deal at 1-alpha.

    Normal approximation on paired differences: n = (z * sd / delta)^2.
    Answers 'how many deals' BEFORE the full run; the full run then refuses
    to report below that count.
    """
    from scipy.stats import norm

    arr = np.asarray(pilot, dtype=np.float64)
    assert len(arr) >= 2
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        return 2
    z = float(norm.ppf(1 - alpha / 2))
    return max(2, int(np.ceil((z * sd / delta) ** 2)))


def require(n_have: int, n_needed: int, what: str) -> None:
    """Refuse to report a result below its resolved sample count."""
    if n_have < n_needed:
        raise InsufficientDealsError(f"{what}: have {n_have} deals, need {n_needed}")
