"""Sample-size discipline: required_deals answers before the run."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ginrl.eval.stats import InsufficientDealsError, require, required_deals


def test_required_deals_matches_closed_form() -> None:
    pilot = [1.0, -1.0, 3.0, -3.0, 2.0, -2.0]
    sd = float(np.std(pilot, ddof=1))
    z = 1.9599639861201952  # norm.ppf(0.975)
    assert required_deals(pilot, 1.0) == max(2, math.ceil((z * sd / 1.0) ** 2))
    assert required_deals(pilot, 0.5) > required_deals(pilot, 1.0)


def test_required_deals_degenerate_pilot() -> None:
    assert required_deals([2.0, 2.0, 2.0], 0.5) == 2
    with pytest.raises(AssertionError):
        required_deals([1.0], 0.5)


def test_require_refuses() -> None:
    with pytest.raises(InsufficientDealsError):
        require(3, 10, "elo")
    require(10, 10, "elo")
