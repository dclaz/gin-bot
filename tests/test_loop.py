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
