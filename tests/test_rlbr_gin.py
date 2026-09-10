"""Regression tests for BR team seat bookkeeping (Phase 5).

A seat flip moves the learner object inside the team tuple, so tuple
position is not the learner. An earlier revision fished the learner out
by position when reseating after each update, silently replacing it with
a second opponent (double-bot teams): the manual-phase set then covered
both seats and learner-side Layoff stopped auto-resolving (SpielError
at the table). Learners live in their own list and are fished by
identity; these tests pin that contract with sentinel objects (no engine).
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from ginrl.eval.rlbr_gin import _flip_seat, _seat_team


def test_seat_team_places_learner_at_its_seat() -> None:
    learner, opp = object(), object()
    assert _seat_team(learner, opp, 0) == (learner, opp)
    assert _seat_team(learner, opp, 1) == (opp, learner)


def test_flip_keeps_learner_identity_at_new_seat() -> None:
    learner, opp = object(), object()
    learners, seats, teams = [learner], [0], [(learner, opp)]
    _flip_seat(teams, seats, learners, 0)
    assert seats == [1]
    assert teams[0][1] is learner
    assert teams[0][0] is opp
    _flip_seat(teams, seats, learners, 0)
    assert seats == [0]
    assert teams[0][0] is learner


def test_reseat_after_flip_keeps_learner() -> None:
    """The update-boundary reseat (fresh opponent, possibly new seat) must
    take the learner from the learners list, never from tuple position."""
    learner, opp, fresh = object(), object(), object()
    learners, seats, teams = [learner], [0], [(learner, opp)]
    _flip_seat(teams, seats, learners, 0)  # learner now at position 1
    seats[0] = 0  # new batch, learner back at seat 0, fresh opponent
    teams[0] = _seat_team(learners[0], fresh, seats[0])
    assert teams[0][0] is learner
    assert teams[0][1] is fresh


def test_flip_trips_on_lost_learner() -> None:
    """A team that lost its learner object fails loudly, not as a
    double-opponent team that crashes hands later."""
    learner, opp = object(), object()
    learners, seats, teams = [learner], [0], [(opp, opp)]
    with pytest.raises(AssertionError, match="learner object lost"):
        _flip_seat(teams, seats, learners, 0)
