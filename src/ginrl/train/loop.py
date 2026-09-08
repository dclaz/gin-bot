"""Self-play training loop for small games (Phase 3).

Collect -> update -> log, with periodic exact-exploitability evals. Every
scalar goes through the Recorder (JSONL is the record; Trackio the view).
Each collection round appends one Trajectory keyed by the round's base
seed; rows carry their table id, and table i's chance stream is
base_seed + i, so the round re-simulates exactly. `load` refits the rows
bit-identically.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ginrl.algos.regpg import Magnet, collect, make_batch, ppo_update
from ginrl.config import Seeds, TrainerConfig
from ginrl.eval.exploitability import NetPolicy, load_small_game, nash_conv
from ginrl.nets.actor_critic import MaskedActorCritic
from ginrl.telemetry.recorder import Recorder, RecorderConfig
from ginrl.train.driver import SmallGameVecEnv, net_config_for_game
from ginrl.train.trajectory import Step, Trajectory, append_trajectory

STAT_TO_METRIC = {
    "pg": "loss/policy",
    "vf": "loss/value",
    "aux": "loss/belief_bce",
    "loss": "loss/total",
    "kl": "reg/kl_to_magnet",
    "reg_alpha": "reg/alpha",
    "clip_frac": "reg/clip_frac",
    "lr": "opt/lr",
    "grad_norm": "opt/grad_norm",
    "entropy": "policy/entropy",
    "explained_variance": "value/explained_variance",
}


@dataclass
class TrainResult:
    """What a self-play run produced (weights live on CPU for comparison)."""

    run_dir: Path
    final_nash_conv: float
    evals: list[tuple[int, float]] = field(default_factory=list)


def train_selfplay(
    game_name: str,
    cfg: TrainerConfig,
    master_seed: int,
    device: torch.device | str,
    run_dir: Path,
    dashboard: bool = False,
    hidden: int = 128,
    layers: int = 2,
) -> TrainResult:
    """Run self-play PPO to `cfg.total_steps` decisions. Returns the record."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    seeds = Seeds(master=master_seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "game": game_name,
                "master_seed": master_seed,
                "device": str(dev),
                **{f: getattr(cfg, f) for f in cfg.__dataclass_fields__ if f != "seeds"},
                "seeds_master": cfg.seeds.master,
            },
            indent=2,
            sort_keys=True,
        )
    )
    recorder = Recorder(
        RecorderConfig(
            run_dir=run_dir,
            run_name=f"{game_name}-s{master_seed}",
            config_hash="",
            dashboard_enabled=dashboard,
        )
    )
    torch.manual_seed(master_seed)
    # Gate determinism is defined at 1 thread on CPU: intra-op reduction
    # order is then fixed, so same seed gives bit-identical checkpoints.
    torch.set_num_threads(1)
    env = SmallGameVecEnv(game_name, n_envs=cfg.n_envs, seeds=seeds)
    net = MaskedActorCritic(net_config_for_game(game_name, hidden, layers)).to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    magnet = Magnet(net, cfg)
    gen = torch.Generator().manual_seed(master_seed + 1)
    game = load_small_game(game_name)

    evals: list[tuple[int, float]] = []
    steps_done, deal_round, redeal_seed = 0, 0, master_seed * 1_000_003 + 11
    t_start = time.perf_counter()
    while steps_done < cfg.total_steps:
        round_seed = master_seed * 10_000 + deal_round
        steps, logps, values, redeal_seed = collect(
            env, net, cfg.batch_size, gen, dev, round_seed, redeal_seed
        )
        deal_round += 1
        if not steps:
            continue
        batch = make_batch(steps, logps, values, cfg, dev)
        stats = ppo_update(net, optimizer, batch, cfg, magnet, gen, steps_done)
        steps_done += len(steps)
        append_trajectory(
            run_dir / "trajectories.jsonl",
            Trajectory(
                env=game_name,
                seed=round_seed,
                steps=tuple(
                    Step(
                        t=t,
                        obs=s.obs,
                        mask=s.mask,
                        action=s.action,
                        reward=s.reward,
                        done=s.done,
                        logp=lp,
                        value=v,
                    )
                    for t, (s, lp, v) in enumerate(zip(steps, logps, values, strict=True))
                ),
            ),
        )
        for stat, metric in STAT_TO_METRIC.items():
            recorder.log_scalar(steps_done, metric, stats[stat])
        elapsed = time.perf_counter() - t_start
        recorder.log_scalar(steps_done, "perf/rows_per_sec", steps_done / elapsed)
        if cfg.eval_every > 0 and deal_round % max(1, cfg.eval_every) == 0:
            # CPU-only move for the exact eval; the magnet ref stays on dev,
            # so mid-run evals on accelerators would need a device-matched
            # copy. gate-p3 runs CPU, where this is a no-op.
            conv = nash_conv(game, NetPolicy(game, net.to("cpu"), torch.device("cpu")))
            net.to(dev)
            evals.append((steps_done, conv))
            recorder.log_scalar(steps_done, "ratings/exploitability", conv)
    final_conv = nash_conv(game, NetPolicy(game, net.to("cpu"), torch.device("cpu")))
    recorder.log_scalar(steps_done, "ratings/exploitability", final_conv)
    recorder.flush()
    recorder.close()
    torch.save(net.state_dict(), run_dir / "final.pt")
    return TrainResult(run_dir=run_dir, final_nash_conv=final_conv, evals=evals)
