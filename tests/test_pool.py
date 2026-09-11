"""Collection pool (design A): determinism, table rebasing, validation."""

from __future__ import annotations

import hashlib

import pytest
import torch

from ginrl.config import HandConfig, TrainerConfig
from ginrl.env.features import feature_dim
from ginrl.nets.gin import GinNet
from ginrl.train.pool import PoolCollector


def _pool(tmp_path, n_workers=2, n_envs=2):
    run = tmp_path / "run"
    (run / "snapshots").mkdir(parents=True, exist_ok=True)
    return PoolCollector(
        n_workers,
        n_envs,
        HandConfig(),
        100,
        50,
        7,
        run,
    )


def _state_dict():
    net = GinNet("mlp", feat_dim=feature_dim(52), deck=52)
    return {k: v.cpu() for k, v in net.state_dict().items()}


def _digest(steps) -> str:
    h = hashlib.sha256()
    for s in steps:
        h.update(
            repr(
                (
                    s.table,
                    s.action,
                    round(s.reward, 12),
                    s.done,
                    s.seat,
                    s.opp_hand,
                    tuple(round(x, 6) for x in s.obs),
                )
            ).encode()
        )
    return h.hexdigest()


def test_pool_repeatable_bit_exact(tmp_path) -> None:
    """Same (master, round) through two independently built pools: identical rows."""
    sd = _state_dict()
    p1 = _pool(tmp_path)
    try:
        a = p1.collect(sd, 64, 1000, 0.02, 1.0, 5)
    finally:
        p1.close()
    p2 = _pool(tmp_path)
    try:
        b = p2.collect(sd, 64, 1000, 0.02, 1.0, 5)
    finally:
        p2.close()
    assert _digest(a[0]) == _digest(b[0])
    assert a[1] == b[1] and a[2] == b[2] and a[3] == b[3] and a[4] == b[4]


def test_pool_tables_rebased_and_rows_requested(tmp_path) -> None:
    """Shard-local table ids are rebased to disjoint global env ids."""
    sd = _state_dict()
    pool = _pool(tmp_path)
    try:
        steps, logps, values, _seed, stats = pool.collect(sd, 64, 1000, 0.02, 1.0, 0)
    finally:
        pool.close()
    assert len(steps) == 64 and len(logps) == 64 and len(values) == 64
    assert sorted({s.table for s in steps}) == [0, 1]
    assert stats["hands"] > 0 and stats["matches"] >= 0


def test_pool_snapshot_cache_path(tmp_path) -> None:
    """A snapshot on disk is loadable worker-side (covers the cache branch)."""
    run = tmp_path / "run"
    (run / "snapshots").mkdir(parents=True)
    torch.save(_state_dict(), run / "snapshots" / "round_10.pt")
    pool = PoolCollector(2, 2, HandConfig(), 100, 50, 7, run)
    try:
        steps, _, _, _, _ = pool.collect(_state_dict(), 64, 1000, 0.02, 1.0, 0)
    finally:
        pool.close()
    assert len(steps) == 64


def test_collect_workers_validation() -> None:
    assert TrainerConfig().collect_workers == 1  # legacy default untouched
    with pytest.raises(Exception, match="evenly"):
        TrainerConfig(n_envs=16, collect_workers=3)
    with pytest.raises(Exception, match=">= 1"):
        TrainerConfig(collect_workers=0)
