"""Self-play loop smoke test: record, trajectories, checkpoint (fast)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ginrl.config import Seeds, TrainerConfig
from ginrl.nets.actor_critic import MaskedActorCritic
from ginrl.train.driver import net_config_for_game
from ginrl.train.loop import train_selfplay
from ginrl.train.trajectory import load as load_traj


def test_loop_writes_record_trajectories_and_checkpoint(tmp_path: Path) -> None:
    cfg = TrainerConfig(
        total_steps=256,
        n_envs=4,
        rollout_len=16,
        epochs=1,
        minibatches=2,
        eval_every=2,
        seeds=Seeds(master=21),
    )
    run_dir = tmp_path / "run"
    result = train_selfplay("kuhn_poker", cfg, 21, torch.device("cpu"), run_dir, hidden=8, layers=1)
    assert result.final_nash_conv == result.final_nash_conv  # finite
    # 256 steps need not beat uniform (a random init is worse than 0.9167);
    # convergence is the gate's job with a real budget. The loop's contract
    # is mechanical: evals run, the record fills, trajectories refit.
    assert len(result.evals) >= 1

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").open()]
    kinds = {m.get("metric", m.get("type")) for m in metrics}
    assert "loss/total" in kinds and "ratings/exploitability" in kinds
    assert "perf/rows_per_sec" in kinds

    trajs = load_traj(run_dir / "trajectories.jsonl")
    assert len(trajs) >= 1
    total_rows = sum(len(t.steps) for t in trajs)
    assert total_rows >= 256
    assert all(t.env == "kuhn_poker" for t in trajs)

    state = torch.load(run_dir / "final.pt", weights_only=True)
    fresh = MaskedActorCritic(net_config_for_game("kuhn_poker", hidden=8, layers=1))
    fresh.load_state_dict(state)  # checkpoint loads into a live net
    assert (run_dir / "config.json").exists()


def _tiny_cfg() -> TrainerConfig:
    return TrainerConfig(
        total_steps=400,
        n_envs=2,
        rollout_len=8,
        epochs=1,
        minibatches=2,
        eval_every=0,
        seeds=Seeds(master=5),
    )


def _run_chain(tmp_path: Path, tag: str, resume_at: int | None) -> list[Path]:
    """One straight run [dir], or two chained runs [part_a, part_b]."""
    cfg = _tiny_cfg()
    first = tmp_path / f"{tag}-a"
    if resume_at is None:
        train_selfplay("kuhn_poker", cfg, 5, torch.device("cpu"), first, hidden=8, layers=1)
        return [first]
    part = TrainerConfig(
        total_steps=resume_at,
        n_envs=2,
        rollout_len=8,
        epochs=1,
        minibatches=2,
        eval_every=0,
        seeds=Seeds(master=5),
    )
    train_selfplay("kuhn_poker", part, 5, torch.device("cpu"), first, hidden=8, layers=1)
    second = tmp_path / f"{tag}-b"
    train_selfplay(
        "kuhn_poker",
        cfg,
        5,
        torch.device("cpu"),
        second,
        hidden=8,
        layers=1,
        resume=first / "checkpoint.pt",
    )
    return [first, second]


def _read_final(run_dir: Path) -> dict[str, object]:
    return torch.load(run_dir / "final.pt", weights_only=True)


def _assert_identical_weights(a: Path, b: Path) -> None:
    sa, sb = _read_final(a), _read_final(b)
    assert sa.keys() == sb.keys()
    for key in sa:
        assert torch.equal(sa[key], sb[key]), f"param {key} differs"


def _traj_rows(dirs: list[Path]) -> bytes:
    """Step rows of chained trajectory files (each file's header skipped)."""
    out = b""
    for d in dirs:
        lines = (d / "trajectories.jsonl").read_bytes().splitlines(keepends=True)
        out += b"".join(lines[1:])
    return out


def test_resume_is_bit_identical_to_uninterrupted(tmp_path: Path) -> None:
    (straight,) = _run_chain(tmp_path, "straight", None)
    first, resumed = _run_chain(tmp_path, "resumed", 200)
    _assert_identical_weights(straight, resumed)
    # Chained trajectory rows (headers excluded: one per file) refit identically.
    assert _traj_rows([straight]) == _traj_rows([first, resumed])
    for name in ("metrics.jsonl", "trajectories.jsonl", "final.pt", "checkpoint.pt"):
        assert (resumed / name).exists()
    # Same seed twice: the gate's determinism check in miniature.
    (again,) = _run_chain(tmp_path, "again", None)
    _assert_identical_weights(straight, again)
