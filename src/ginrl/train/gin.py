"""Reduced-gin self-play with the Phase 3 trainer (Phase 4).

The rollout rows carry the concatenated gin observation (belief features,
public-event window, raw tensor) plus a joint opponent-hand label; the PPO
update, GAE, magnet, and Recorder machinery is shared verbatim with the
small-game path. Both seats share one GinNet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ginrl.agents.learned import BeliefAblatedAgent, GinNetAgent
from ginrl.algos.regpg import Magnet, make_batch, ppo_update, sample_actions
from ginrl.config import REDUCED_HAND_CONFIG, HandConfig, Seeds, TrainerConfig
from ginrl.env.features import EVENT_WINDOW, feature_dim
from ginrl.env.game import HandEnv
from ginrl.eval.arena import Arena
from ginrl.nets.gin import N_CARD_BLOCKS, RAW_DIM, GinNet, count_params
from ginrl.telemetry.recorder import Recorder, RecorderConfig
from ginrl.train.driver import RolloutStep
from ginrl.train.loop import STAT_TO_METRIC, load_checkpoint, save_checkpoint

GIN_STAT_TO_METRIC = dict(
    STAT_TO_METRIC,
    **{
        "paired_score": "ratings/paired_score",
        "belief_auc": "belief/auc",
    },
)


def multi_hot(hand: tuple[int, ...], deck: int) -> tuple[float, ...]:
    vec = [0.0] * deck
    for card in hand:
        vec[card] = 1.0
    return tuple(vec)


def gin_observation_row(
    env: HandEnv, seat: int, feat_dim: int, mask_beliefs: bool = False
) -> tuple[tuple[float, ...], tuple[bool, ...], tuple[int, ...]]:
    """(belief+events+raw flat, 56-mask, event tokens) for one decision.

    mask_beliefs zeroes the known_opp/declined_opp blocks (9, 10): the
    no-belief-input control. Same masking as BeliefAblatedAgent at eval.
    """
    deck = env.config.deck_size
    obs = [0.0] * (feat_dim + EVENT_WINDOW + RAW_DIM)
    obs[:feat_dim] = env.features_for(seat)
    if mask_beliefs:
        obs[9 * deck : N_CARD_BLOCKS * deck] = [0.0] * (N_CARD_BLOCKS * deck - 9 * deck)
    events = env.trackers[seat].event_window()
    obs[feat_dim : feat_dim + EVENT_WINDOW] = [float(t) for t in events]
    obs[feat_dim + EVENT_WINDOW :] = list(env.tensor_for(seat))
    return tuple(obs), tuple(env.legal_mask()), events


def collect_gin(
    envs: list[HandEnv],
    net: GinNet,
    n_rows: int,
    gen: torch.Generator,
    device: torch.device,
    base_seed: int,
    redeal_seed: int,
    feat_dim: int,
    reward_scale: float = 1.0,
    mask_beliefs: bool = False,
) -> tuple[list[RolloutStep], list[float], list[float], int]:
    """Self-play rollout: one shared net plays both seats on every table."""
    deck = envs[0].config.deck_size
    for i, env in enumerate(envs):
        env.reset(seed=base_seed + i)
    steps: list[RolloutStep] = []
    logps: list[float] = []
    values: list[float] = []
    net.eval()
    while len(steps) < n_rows:
        live = list(range(len(envs)))
        feats, masks, seats, events, opps = [], [], [], [], []
        for i in live:
            env = envs[i]
            seat = env.state().current_player()
            obs, mask, tokens = gin_observation_row(env, seat, feat_dim, mask_beliefs)
            feats.append(obs)
            masks.append(mask)
            seats.append(seat)
            events.append(tokens)
            opps.append(multi_hot(env.hidden_opp_hand(seat), deck))
        with torch.no_grad():
            logits, value, _ = net(
                torch.tensor(feats, dtype=torch.float32, device=device),
                torch.tensor(masks, dtype=torch.bool, device=device),
            )
        actions, blogp = sample_actions(logits.cpu(), gen)
        for k, i in enumerate(live):
            env = envs[i]
            seat = seats[k]
            result = env.step(actions[k])
            steps.append(
                RolloutStep(
                    table=i,
                    obs=feats[k],
                    mask=masks[k],
                    action=actions[k],
                    reward=(result.returns[seat] if result.done else 0.0) * reward_scale,
                    done=result.done,
                    seat=seat,
                    opp_card=-1,  # unused: opp_hand selects the joint BCE path
                    opp_hand=opps[k],
                )
            )
            logps.append(float(blogp[k]))
            values.append(float(value[k].cpu()))
            if result.done:
                redeal_seed += 1
                env.reset(seed=redeal_seed)
    net.train()
    return steps, logps, values, redeal_seed


@dataclass
class GinTrainResult:
    run_dir: Path
    evals: list[tuple[int, float]] = field(default_factory=list)
    final_score: float = 0.0
    final_entropy: float = 0.0
    params: int = 0


def eval_vs_heuristic(
    net: GinNet,
    config: HandConfig,
    feat_dim: int,
    device: torch.device,
    seed: int,
    n_deals: int,
    mask_beliefs: bool = False,
) -> float:
    """Mean paired points/hand of the current net vs the heuristic baseline."""
    from ginrl.agents.baselines import HeuristicAgent

    arena = Arena()
    arena.config = config
    cls = BeliefAblatedAgent if mask_beliefs else GinNetAgent
    agent = cls(net, feat_dim, device, seed=seed)
    summary = arena.duplicate_summary(agent, HeuristicAgent(), seed=seed, n_deals=n_deals)
    return summary.points_per_hand().mean


def train_gin_selfplay(
    config: HandConfig = REDUCED_HAND_CONFIG,
    torso: str = "mlp",
    cfg: TrainerConfig | None = None,
    master_seed: int = 0,
    device: torch.device | None = None,
    run_dir: Path | None = None,
    hidden: int = 128,
    layers: int = 2,
    eval_every: int = 20,
    eval_deals: int = 100,
    resume: Path | None = None,
    mask_beliefs: bool = False,
) -> GinTrainResult:
    """Self-play a GinNet on reduced gin; periodic heuristic evals."""
    cfg = cfg or TrainerConfig()
    dev = device or torch.device("cpu")
    run = run_dir or Path(f"runs/gin-{torso}-{master_seed}")
    run.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(RecorderConfig(run_name=run.name, run_dir=run))
    torch.manual_seed(master_seed)
    torch.set_num_threads(1)
    seeds = Seeds(master=master_seed)
    deck = config.deck_size
    feat_dim = feature_dim(deck)
    envs = [HandEnv(config, seeds) for _ in range(cfg.n_envs)]
    net = GinNet(torso, feat_dim=feat_dim, deck=deck, hidden=hidden, layers=layers).to(dev)
    if cfg.orthogonal_init:
        from ginrl.nets.actor_critic import apply_orthogonal

        apply_orthogonal(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    magnet = Magnet(net, cfg)
    gen = torch.Generator().manual_seed(master_seed + 1)
    steps_done, deal_round, redeal_seed = 0, 0, master_seed * 1_000_000 + 7
    if resume is not None:
        steps_done, deal_round, redeal_seed = load_checkpoint(
            resume, net, optimizer, magnet, gen, dev
        )
    result = GinTrainResult(run_dir=run, params=count_params(net))
    last_stats: dict[str, float] = {}
    while steps_done < cfg.total_steps:
        t0 = time.perf_counter()
        steps, logps, values, redeal_seed = collect_gin(
            envs,
            net,
            cfg.batch_size,
            gen,
            dev,
            master_seed * 1_000 + deal_round,
            redeal_seed,
            feat_dim,
            cfg.reward_scale,
            mask_beliefs,
        )
        rows_per_sec = len(steps) / max(time.perf_counter() - t0, 1e-9)
        deal_round += 1
        batch = make_batch(steps, logps, values, cfg, dev)
        last_stats = ppo_update(net, optimizer, batch, cfg, magnet, gen, steps_done)
        steps_done += len(steps)
        for name, metric in GIN_STAT_TO_METRIC.items():
            if name in last_stats:
                recorder.log_scalar(steps_done, metric, last_stats[name])
        recorder.log_scalar(steps_done, "perf/rows_per_sec", rows_per_sec)
        if deal_round % eval_every == 0 or steps_done >= cfg.total_steps:
            score = eval_vs_heuristic(
                net, config, feat_dim, dev, master_seed, eval_deals, mask_beliefs
            )
            result.evals.append((steps_done, score))
            recorder.log_scalar(steps_done, "ratings/paired_score", score)
        save_checkpoint(
            run / "checkpoint.pt",
            net,
            optimizer,
            magnet,
            gen,
            steps_done,
            deal_round,
            redeal_seed,
        )
    torch.save(net.state_dict(), run / "final.pt")
    result.final_score = result.evals[-1][1] if result.evals else 0.0
    result.final_entropy = last_stats.get("entropy", 0.0)
    recorder.close()
    return result
