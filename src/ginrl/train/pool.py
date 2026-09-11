"""Multiprocess collection pool for match self-play (Phase 5, design A).

Workers run the UNMODIFIED collect_match on fixed env shards; the parent
concatenates shard rows in shard order and runs the update. All inference
for the learner stays central only in the single-process path — here each
worker forwards its own shard (learner weights shipped per round, ~0.5MB).

Determinism contract (documented tradeoff, see commit):
- pool-vs-pool is bit-exact: every worker stream derives from
  Seeds(master).spawn(f"pool:{shard}:{round}"), so the same seed gives the
  same rows across runs. Reproducible debugging holds.
- pool-vs-single-process is DISTRIBUTIONALLY equivalent, not bit-identical:
  per-shard opponent draws, mirror-learner sampling streams, and snapshot
  reuse (workers cache snapshot nets per match instead of reloading from
  disk) all differ from the shared-stream single-process rollout by
  construction. SGD consumes a distribution, not a trajectory; the
  equivalence probe and gate-p5 carry correctness, not a rows hash.
  Do not assert pool==single rows: that test would be a lie.

Constraints honored: workers are CPU-only, spawned (never forked) before
any MPS context exists, one thread each; engine bots stay per-game via the
untouched wire()/begin_game path (landmine 3); the Recorder stays
parent-side, workers return scalars (landmine 11).
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

from ginrl.config import HandConfig

TIMEOUT_S = 300.0
SEED_STRIDE = 1 << 20  # reset-seed range per (round, shard); asserted, never wraps


@dataclass
class _RoundSpec:
    round_idx: int
    state_dict: dict
    n_rows: int
    first_seed: int
    reward_scale: float
    match_bonus: float


def _snapshot_key(path: Path) -> tuple[str, float]:
    return (str(path), path.stat().st_mtime)


def _worker_main(conn, init: dict) -> None:
    """Long-lived shard worker: own envs, snapshot cache, per-round collect."""
    import sys

    sys.path.insert(0, "src")
    import dataclasses

    import torch as _torch

    _torch.set_num_threads(1)
    from ginrl.config import MatchConfig, Seeds
    from ginrl.env.features import feature_dim
    from ginrl.env.match import MatchEnv
    from ginrl.nets.gin import GinNet
    from ginrl.train.match import collect_match

    torch = _torch
    hand_config = init["hand_config"]
    match_config = MatchConfig(
        hand=hand_config,
        target_score=init["target_score"],
        max_hands=init["max_hands"],
        auto_advance=False,
    )
    deck = hand_config.deck_size
    feat_dim = feature_dim(deck)
    dev = torch.device("cpu")
    seeds = Seeds(master=init["master_seed"])
    envs = [MatchEnv(config=match_config, seeds=seeds) for _ in range(init["shard_envs"])]
    run = Path(init["run"])
    snap_cache: dict = {}

    import ginrl.train.match as _match_mod

    _real_load = _match_mod._load_snapshot

    def _cached_load(path, feat_dim, deck, device):
        key = _snapshot_key(Path(path))
        hit = snap_cache.get(key)
        if hit is not None:
            return hit
        agent = _real_load(path, feat_dim, deck, device)
        if len(snap_cache) >= 8:
            snap_cache.pop(next(iter(snap_cache)))
        snap_cache[key] = agent
        return agent

    _match_mod._load_snapshot = _cached_load
    try:
        conn.send(("ready", init["shard"]))
        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                return
            assert msg[0] == "round", msg[0]
            spec: _RoundSpec = msg[1]
            t0 = time.perf_counter()
            net = GinNet("mlp", feat_dim=feat_dim, deck=deck).to(dev)
            net.load_state_dict(spec.state_dict)
            net.eval()
            # Every stream derives from (master, shard, round): pool-vs-pool
            # bit-exact. collect_match builds its own seed-0 learner per call
            # (mirror opponent), exactly like one single-process round.
            rng = Seeds(master=init["master_seed"]).spawn(f"pool:{init['shard']}:{spec.round_idx}")
            gen = torch.Generator().manual_seed(rng.randrange(1 << 63))
            steps, logps, values, next_seed, stats = collect_match(
                envs,
                net,
                spec.n_rows,
                gen,
                dev,
                feat_dim,
                spec.reward_scale,
                spec.match_bonus,
                run,
                rng,
                spec.first_seed,
            )
            consumed = next_seed - spec.first_seed
            assert consumed < SEED_STRIDE, f"seed stride wrapped: {consumed}"
            conn.send(
                (
                    "rows",
                    [dataclasses.asdict(s) for s in steps],
                    logps,
                    values,
                    consumed,
                    dict(stats),
                    time.perf_counter() - t0,
                )
            )
    except Exception:
        conn.send(("error", traceback.format_exc()))
        raise


class PoolCollector:
    """Parent side: ship weights, gather shard rows in shard order."""

    def __init__(
        self,
        n_workers: int,
        n_envs: int,
        hand_config: HandConfig,
        target_score: int,
        max_hands: int,
        master_seed: int,
        run: Path,
    ) -> None:
        assert n_workers > 1
        assert n_envs % n_workers == 0, "envs must split evenly across workers"
        ctx = get_context("spawn")
        self.n_workers = n_workers
        self.shard_envs = n_envs // n_workers
        self.pipes = []
        self.procs = []
        for shard in range(n_workers):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                target=_worker_main,
                args=(
                    child_conn,
                    {
                        "shard": shard,
                        "shard_envs": self.shard_envs,
                        "hand_config": hand_config,
                        "target_score": target_score,
                        "max_hands": max_hands,
                        "master_seed": master_seed,
                        "run": str(run),
                    },
                ),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            tag, got = parent_conn.recv()
            assert tag == "ready" and got == shard, (tag, got)
            self.pipes.append(parent_conn)
            self.procs.append(proc)

    def collect(
        self,
        state_dict: dict,
        n_rows: int,
        first_seed: int,
        reward_scale: float,
        match_bonus: float,
        round_idx: int,
    ) -> tuple[list, list, list, int, dict]:
        base, rem = divmod(n_rows, self.n_workers)
        for shard, pipe in enumerate(self.pipes):
            rows_s = base + (1 if shard < rem else 0)
            pipe.send(
                (
                    "round",
                    _RoundSpec(
                        round_idx=round_idx,
                        state_dict=state_dict,
                        n_rows=rows_s,
                        first_seed=first_seed + shard * SEED_STRIDE,
                        reward_scale=reward_scale,
                        match_bonus=match_bonus,
                    ),
                )
            )
        all_rows: list = []
        all_logps: list = []
        all_values: list = []
        stats = {"hands": 0, "matches": 0, "caps": 0, "learner_wins": 0}
        for shard, pipe in enumerate(self.pipes):
            if not pipe.poll(TIMEOUT_S):
                raise RuntimeError("pool worker timed out; failing loud, no partial rows")
            msg = pipe.recv()
            if msg[0] == "error":
                raise RuntimeError(f"pool worker failed:\n{msg[1]}")
            assert msg[0] == "rows", msg[0]
            _, rows, logps, values, _consumed, st, _dt = msg
            # table ids group rows into per-episode GAE streams: rebase the
            # worker-local ids into global env ids or shards would collide.
            base_table = shard * self.shard_envs
            for r in rows:
                r["table"] = r["table"] + base_table
            all_rows.extend(rows)
            all_logps.extend(logps)
            all_values.extend(values)
            for k in stats:
                stats[k] += st.get(k, 0)
        # Deterministic shard-order truncation, mirroring rows[:n_rows].
        from ginrl.train.driver import RolloutStep

        steps = [
            RolloutStep(
                table=int(r["table"]),
                obs=tuple(r["obs"]),
                mask=tuple(r["mask"]),
                action=int(r["action"]),
                reward=float(r["reward"]),
                done=bool(r["done"]),
                seat=int(r["seat"]),
                opp_card=int(r["opp_card"]),
                opp_hand=tuple(r["opp_hand"]) if r["opp_hand"] is not None else None,
            )
            for r in all_rows[:n_rows]
        ]
        return (
            steps,
            all_logps[:n_rows],
            all_values[:n_rows],
            first_seed + self.n_workers * SEED_STRIDE,
            stats,
        )

    def close(self) -> None:
        for pipe in self.pipes:
            try:
                pipe.send(("stop",))
            except Exception:
                pass
        for proc in self.procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
