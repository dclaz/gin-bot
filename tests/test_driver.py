"""Small-game self-play driver: legal, seeded, bit-identical reruns."""

from __future__ import annotations

import random

import pytest

from ginrl.config import Seeds
from ginrl.train.driver import SmallGameVecEnv, net_config_for_game


def collect(env: SmallGameVecEnv, rng: random.Random, max_rounds: int = 40) -> list:
    """Random-policy rollout; re-deals finished tables with fresh seeds."""
    rows = []
    seed_counter = [10_000]
    for _ in range(max_rounds):
        actions = []
        for i in range(env.n_envs):
            if env.pending_returns(i) is not None:
                seed_counter[0] += 1
                env.reset_table(i, seed_counter[0])
                actions.append(None)  # re-dealt table sits out one round
                continue
            state = env._states[i]
            if state.is_terminal():  # pragma: no cover - act() terminals first
                actions.append(None)
                continue
            legal = state.legal_actions()
            actions.append(rng.choice(legal))
        rollout = env.act(actions)
        rows.extend(rollout.steps)
        if len(rows) >= env.n_envs * 4:
            break
    return rows


def test_driver_runs_legal_hands_and_scores_zero_sum() -> None:
    env = SmallGameVecEnv("kuhn_poker", n_envs=4, seeds=Seeds(master=1))
    env.reset_all(base_seed=0)
    rows = collect(env, random.Random(0))
    assert len(rows) > 0
    assert all(sum(m) >= 1 for m in (r.mask for r in rows))
    assert {r.seat for r in rows} == {0, 1}  # both seats act
    terminals = [r for r in rows if r.done]
    assert terminals, "a random rollout should finish hands"
    assert all(0 <= r.opp_card < 3 for r in rows)
    for i in range(env.n_envs):
        pending = env.pending_returns(i)
        if pending is not None:
            assert pending[0] + pending[1] == pytest.approx(0.0)


def test_same_seed_same_rollout() -> None:
    def run() -> list:
        env = SmallGameVecEnv("leduc_poker", n_envs=3, seeds=Seeds(master=7))
        env.reset_all(base_seed=42)
        return collect(env, random.Random(3))

    first, second = run(), run()
    assert len(first) == len(second) > 0
    for a, b in zip(first, second, strict=True):
        assert a == b


def test_net_config_matches_live_game() -> None:
    kuhn = net_config_for_game("kuhn_poker")
    assert (kuhn.obs_dim, kuhn.n_actions, kuhn.aux_dim) == (11, 2, 3)
    leduc = net_config_for_game("leduc_poker")
    assert (leduc.obs_dim, leduc.n_actions, leduc.aux_dim) == (30, 3, 6)
    with pytest.raises(ValueError):
        SmallGameVecEnv("gin_rummy", n_envs=1)
