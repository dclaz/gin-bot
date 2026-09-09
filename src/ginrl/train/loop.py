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
from ginrl.eval.rlbr import net_fixed, run_br_phase
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


def save_checkpoint(
    path: Path,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    magnet: Magnet,
    gen: torch.Generator,
    steps_done: int,
    deal_round: int,
    redeal_seed: int,
) -> None:
    """Persist full training state: weights, optimizer, magnet, RNG, counters."""
    torch.save(
        {
            "net": net.state_dict(),
            "opt": optimizer.state_dict(),
            "magnet": magnet.snapshot_state(),
            "gen_state": gen.get_state(),
            "torch_state": torch.get_rng_state(),
            "steps_done": steps_done,
            "deal_round": deal_round,
            "redeal_seed": redeal_seed,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    magnet: Magnet,
    gen: torch.Generator,
    device: torch.device,
) -> tuple[int, int, int]:
    """Restore training state saved by `save_checkpoint`. Returns counters."""
    # weights_only=True suffices: every entry is tensors or ints.
    state = torch.load(path, weights_only=True, map_location=device)
    net.load_state_dict(state["net"])
    optimizer.load_state_dict(state["opt"])
    magnet.restore_state(state["magnet"], net)
    gen.set_state(state["gen_state"])
    torch.set_rng_state(state["torch_state"])
    return int(state["steps_done"]), int(state["deal_round"]), int(state["redeal_seed"])


def train_selfplay(
    game_name: str,
    cfg: TrainerConfig,
    master_seed: int,
    device: torch.device | str,
    run_dir: Path,
    dashboard: bool = False,
    hidden: int = 128,
    layers: int = 2,
    resume: Path | None = None,
    save_every: int = 0,
) -> TrainResult:
    """Run self-play PPO to `cfg.total_steps` decisions. Returns the record.

    `resume` continues a checkpoint from `save_checkpoint`: weights,
    optimizer, magnet, RNG states and counters are restored, so resuming
    with the same config is bit-identical to one uninterrupted run (the
    determinism test proves it). Each call writes its own run dir, so the
    record stays append-only per phase. `save_every` writes a checkpoint
    every N updates (0 = only at the end). `total_steps` is cumulative
    across a resume chain: a settle phase with its own 2M-step budget on top
    of a 500k-step approach passes total_steps=2_500_000. Cross-config
    resume (e.g. a new learning rate for a settle phase) is mechanical, not
    identical: keep `anneal='none'` across the chain or the schedules will
    surprise you.
    """
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
    if resume is not None:
        steps_done, deal_round, redeal_seed = load_checkpoint(
            resume, net, optimizer, magnet, gen, dev
        )
    t_start = time.perf_counter()
    n_updates = 0
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
        n_updates += 1
        if save_every > 0 and n_updates % save_every == 0:
            save_checkpoint(
                run_dir / "checkpoint.pt",
                net,
                optimizer,
                magnet,
                gen,
                steps_done,
                deal_round,
                redeal_seed,
            )
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
    save_checkpoint(
        run_dir / "checkpoint.pt",
        net,
        optimizer,
        magnet,
        gen,
        steps_done,
        deal_round,
        redeal_seed,
    )
    return TrainResult(run_dir=run_dir, final_nash_conv=final_conv, evals=evals)


def train_alternating(
    game_name: str,
    cfg: TrainerConfig,
    master_seed: int,
    device: torch.device | str,
    run_dir: Path,
    phase_steps: int,
    n_outer: int,
    dashboard: bool = False,
    hidden: int = 128,
    layers: int = 2,
) -> TrainResult:
    """Self-play by alternating best-response phases (fictitious-play style).

    Each outer round trains seat 0 against a frozen snapshot, then seat 1
    against the new snapshot, sharing one net/optimizer/magnet throughout.
    Freezing the opponent for a whole phase removes the simultaneous
    co-adaptation cycle: every phase faces a stationary opponent (which the
    BR machinery is proven to exploit), while the shared magnet damps
    overfitting to each snapshot. Deterministic given `master_seed`.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    seeds = Seeds(master=master_seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "game": game_name,
                "master_seed": master_seed,
                "device": str(dev),
                "mode": "alternating",
                "phase_steps": phase_steps,
                "n_outer": n_outer,
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
            run_name=f"{game_name}-alt-s{master_seed}",
            config_hash="",
            dashboard_enabled=dashboard,
        )
    )
    torch.manual_seed(master_seed)
    torch.set_num_threads(1)
    env = SmallGameVecEnv(game_name, n_envs=cfg.n_envs, seeds=seeds)
    net = MaskedActorCritic(net_config_for_game(game_name, hidden, layers)).to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    magnet = Magnet(net, cfg)
    gen = torch.Generator().manual_seed(master_seed + 1)
    game = load_small_game(game_name)

    evals: list[tuple[int, float]] = []
    total_steps = 0
    redeal_seed = master_seed * 1_000_003 + 11
    for outer in range(n_outer):
        for seat in (0, 1):
            opponent = net_fixed(net, dev, seed=master_seed * 7919 + outer * 2 + seat)
            fixed_rng = seeds.spawn(f"alt-{outer}-{seat}")
            mean, redeal_seed, taken, stats = run_br_phase(
                env,
                net,
                optimizer,
                magnet,
                seat,
                opponent,
                fixed_rng,
                cfg,
                gen,
                dev,
                base_seed=master_seed * 1_000 + outer * 100 + seat * 37,
                redeal_seed=redeal_seed,
                budget_steps=phase_steps,
                steps_done=total_steps,
            )
            total_steps += taken
            recorder.log_scalar(total_steps, f"ratings/br_seat{seat}", mean)
            for stat, metric in STAT_TO_METRIC.items():
                if stat in stats:
                    recorder.log_scalar(total_steps, metric, stats[stat])
        conv = nash_conv(game, NetPolicy(game, net.to("cpu"), torch.device("cpu")))
        net.to(dev)
        evals.append((total_steps, conv))
        recorder.log_scalar(total_steps, "ratings/exploitability", conv)
    final_conv = nash_conv(game, NetPolicy(game, net.to("cpu"), torch.device("cpu")))
    recorder.log_scalar(total_steps, "ratings/exploitability", final_conv)
    recorder.flush()
    recorder.close()
    torch.save(net.state_dict(), run_dir / "final.pt")
    save_checkpoint(
        run_dir / "checkpoint.pt",
        net,
        optimizer,
        magnet,
        gen,
        total_steps,
        0,
        redeal_seed,
    )
    return TrainResult(run_dir=run_dir, final_nash_conv=final_conv, evals=evals)
