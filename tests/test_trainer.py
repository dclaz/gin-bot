"""TrainerConfig validation and TrainerConfig <-> trajectory contract."""

from __future__ import annotations

import json

import pytest

from ginrl.config import MAGNET_EMA, MAGNET_SNAPSHOT, Seeds, TrainerConfig
from ginrl.train.trajectory import Step, Trajectory, append_trajectory, load


def test_defaults_are_self_consistent() -> None:
    cfg = TrainerConfig()
    assert cfg.batch_size == cfg.n_envs * cfg.rollout_len
    assert cfg.minibatch_size * cfg.minibatches == cfg.batch_size
    assert cfg.seeds == Seeds()


def test_rejects_bad_env_and_magnet() -> None:
    with pytest.raises(ValueError):
        TrainerConfig(env="chess")
    with pytest.raises(ValueError):
        TrainerConfig(magnet_mode="horseshoe")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TrainerConfig(magnet_mode=MAGNET_SNAPSHOT, snapshot_every=0)
    ok = TrainerConfig(magnet_mode=MAGNET_SNAPSHOT, snapshot_every=10)
    assert ok.snapshot_every == 10
    assert TrainerConfig(magnet_mode=MAGNET_EMA).magnet_ema_decay == 0.999


def test_rejects_nonpositive_sizes() -> None:
    with pytest.raises(ValueError):
        TrainerConfig(n_envs=0)
    with pytest.raises(ValueError):
        TrainerConfig(n_envs=1, rollout_len=2, minibatches=4)


def test_trajectory_roundtrip_is_bit_identical(tmp_path) -> None:
    path = tmp_path / "trajectories.jsonl"
    steps = tuple(
        Step(
            t=t,
            obs=(0.1 + t, -1.5 * t, 1.0 / 3.0),
            mask=(True, False, True),
            action=t % 2,
            reward=0.1 * t - 0.05,
            done=t == 3,
            logp=-0.6931471805599453,
            value=1e-8,
        )
        for t in range(4)
    )
    traj = Trajectory(env="kuhn", seed=7, steps=steps)
    assert append_trajectory(path, traj, config_hash="abc") == 4
    (refit,) = load(path)
    assert refit == traj
    assert refit.steps[2].obs[2] == 1.0 / 3.0  # exact float, not approximate


def test_trajectory_namespace_ignores_evaluator_rows(tmp_path) -> None:
    """traj loader skips header/leg rows; game-record loader skips traj rows."""
    from ginrl.eval.record import LEG, load_deals

    path = tmp_path / "mixed.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "header", "git_sha": "x"}) + "\n")
        f.write(
            json.dumps(
                {
                    "type": LEG,
                    "a": "a",
                    "b": "b",
                    "a_first": True,
                    "first_won": True,
                    "margin_a": 1.0,
                    "deal": 0,
                    "wall": 0,
                }
            )
            + "\n"
        )
    assert load(path) == []
    assert load_deals(path) == []  # incomplete deal skipped, no crash

    traj_path = tmp_path / "traj.jsonl"
    append_trajectory(
        traj_path,
        Trajectory(
            env="leduc",
            seed=1,
            steps=(Step(0, (0.5,), (True,), 0, 1.0, True, 0.0, 0.0),),
        ),
    )
    assert load_deals(traj_path) == []  # traj rows are not legs
    (refit,) = load(traj_path)
    assert (refit.env, refit.seed) == ("leduc", 1)
