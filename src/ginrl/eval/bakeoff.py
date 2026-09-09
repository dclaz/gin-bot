"""Five-torso architecture bake-off on reduced gin (Phase 4).

Every cell trains the same gradient-step budget (same env steps, batch,
epochs, minibatches — hence identical optimizer-update counts) from its
own seed, then reports: paired score vs the heuristic with CI, held-out
belief AUC, parameter count, training rows/s, inference latency, and final
policy entropy. The table decides nothing by itself (the gate asserts the
checks); the winner is recorded, not prescribed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ginrl.agents.baselines import HeuristicAgent
from ginrl.agents.learned import BeliefAblatedAgent, GinNetAgent
from ginrl.config import REDUCED_HAND_CONFIG, Seeds, TrainerConfig
from ginrl.env.features import feature_dim
from ginrl.eval.arena import Arena
from ginrl.eval.belief import BeliefState, belief_auc
from ginrl.nets.gin import GIN_TORSOS, GinNet
from ginrl.train.gin import train_gin_selfplay


@dataclass
class BakeoffCell:
    torso: str
    seed: int
    run_dir: str
    updates: int
    final_score: float
    score_lo: float
    score_hi: float
    belief_auc: float
    base_auc: float
    params: int
    rows_per_sec: float
    infer_ms: float
    final_entropy: float


def bakeoff_cfg(total_steps: int, seed: int) -> TrainerConfig:
    return TrainerConfig(
        total_steps=total_steps,
        n_envs=16,
        rollout_len=128,
        epochs=2,
        minibatches=4,
        seeds=Seeds(master=seed),
        magnet_mode="uniform",
        reg_coef=0.1,
        reward_scale=0.1,
    )


def updates_per_run(cfg: TrainerConfig) -> int:
    rounds = (cfg.total_steps + cfg.batch_size - 1) // cfg.batch_size
    return rounds * cfg.epochs * cfg.minibatches


def mean_rows_per_sec(run_dir: Path) -> float:
    vals = []
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar" and row.get("metric") == "perf/rows_per_sec":
            vals.append(float(row["value"]))
    return sum(vals) / len(vals) if vals else 0.0


def inference_ms(net: GinNet, device: torch.device, reps: int = 200) -> float:
    net.eval()
    obs = torch.zeros(1, net.obs_dim)
    mask = torch.ones(1, 56, dtype=torch.bool)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            net(obs, mask)
    return (time.perf_counter() - t0) / reps * 1000.0


def run_cell(
    torso: str,
    seed: int,
    *,
    total_steps: int,
    device: torch.device,
    parent: Path,
    eval_deals_final: int,
    belief_states: list[BeliefState],
    belief_seed: int,
    mask_beliefs: bool = False,
) -> BakeoffCell:
    """Train one (torso, seed) cell and evaluate it fully."""
    if torso not in GIN_TORSOS:
        raise ValueError(f"unknown torso {torso!r}")
    run_dir = parent / f"{torso}-s{seed}"
    cfg = bakeoff_cfg(total_steps, seed)
    result = train_gin_selfplay(
        REDUCED_HAND_CONFIG,
        torso,
        cfg,
        seed,
        device,
        run_dir,
        eval_every=20,
        eval_deals=100,
        mask_beliefs=mask_beliefs,
    )
    deck = REDUCED_HAND_CONFIG.deck_size
    feat_dim = feature_dim(deck)
    net = GinNet(torso, feat_dim=feat_dim, deck=deck, hidden=128)
    net.load_state_dict(torch.load(run_dir / "final.pt", weights_only=True))
    net.eval()
    arena = Arena()
    arena.config = REDUCED_HAND_CONFIG
    agent_cls = BeliefAblatedAgent if mask_beliefs else GinNetAgent
    summary = arena.duplicate_summary(
        agent_cls(net, feat_dim, device, seed=belief_seed),
        HeuristicAgent(),
        seed=belief_seed,
        n_deals=eval_deals_final,
    )
    ppb = summary.points_per_hand()
    aux_auc, base_auc = belief_auc(net, belief_states, device)
    return BakeoffCell(
        torso=torso,
        seed=seed,
        run_dir=str(run_dir),
        updates=updates_per_run(cfg),
        final_score=ppb.mean,
        score_lo=ppb.lo,
        score_hi=ppb.hi,
        belief_auc=aux_auc,
        base_auc=base_auc,
        params=result.params,
        rows_per_sec=mean_rows_per_sec(run_dir),
        infer_ms=inference_ms(net, device),
        final_entropy=result.final_entropy,
    )


def cell_dict(cell: BakeoffCell) -> dict[str, object]:
    return asdict(cell)
