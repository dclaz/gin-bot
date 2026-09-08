"""Replayable training trajectories: `runs/<run>/trajectories.jsonl`.

Each rollout round appends one trajectory row per environment: the env name,
the seed that produced it, and every step (obs, mask, action, reward, done,
logp, value). Re-simulation is never needed to audit a run -- `load`
returns bit-identical floats (Python's JSON float round-trip is exact), so a
refit from disk reproduces the rollout exactly.

Label hygiene: trajectory rows use the `traj_header` / `traj_step` type tags,
never the evaluator's `header` / `leg` tags, so `ginrl.eval.record` loaders
skip them and vice versa. Provenance (git SHA, config hash, facts hash) rides
on the header, like the game record.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ginrl.telemetry.provenance import git_sha, spiel_facts_hash

TRAJ_HEADER = "traj_header"
TRAJ_STEP = "traj_step"


@dataclass(frozen=True)
class Step:
    """One environment transition as stored on disk."""

    t: int
    obs: tuple[float, ...]
    mask: tuple[bool, ...]
    action: int
    reward: float
    done: bool
    logp: float
    value: float


@dataclass(frozen=True)
class Trajectory:
    """One full rollout from one seeded environment."""

    env: str
    seed: int
    steps: tuple[Step, ...] = field(default_factory=tuple)


def ensure_traj_header(path: Path, config_hash: str = "") -> None:
    """Write the provenance header if the trajectory file is new."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "type": TRAJ_HEADER,
                    "git_sha": git_sha(),
                    "config_hash": config_hash,
                    "facts_hash": spiel_facts_hash(),
                    "t": time.time(),
                }
            )
            + "\n"
        )


def append_trajectory(path: Path, traj: Trajectory, config_hash: str = "") -> int:
    """Append one row per step. Returns rows written."""
    ensure_traj_header(path, config_hash)
    with path.open("a", encoding="utf-8") as f:
        for step in traj.steps:
            f.write(
                json.dumps(
                    {
                        "type": TRAJ_STEP,
                        "env": traj.env,
                        "seed": traj.seed,
                        "t": step.t,
                        "obs": list(step.obs),
                        "mask": list(step.mask),
                        "action": step.action,
                        "reward": step.reward,
                        "done": step.done,
                        "logp": step.logp,
                        "value": step.value,
                    }
                )
                + "\n"
            )
    return len(traj.steps)


def load(path: Path) -> list[Trajectory]:
    """Refit trajectories from disk, grouped by (env, seed) in write order."""
    groups: dict[tuple[str, int], list[Step]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("type") != TRAJ_STEP:
                continue  # evaluator rows and headers are not our namespace
            step = Step(
                t=int(row["t"]),
                obs=tuple(float(x) for x in row["obs"]),
                mask=tuple(bool(x) for x in row["mask"]),
                action=int(row["action"]),
                reward=float(row["reward"]),
                done=bool(row["done"]),
                logp=float(row["logp"]),
                value=float(row["value"]),
            )
            groups.setdefault((str(row["env"]), int(row["seed"])), []).append(step)
    return [
        Trajectory(env=env, seed=seed, steps=tuple(steps)) for (env, seed), steps in groups.items()
    ]
