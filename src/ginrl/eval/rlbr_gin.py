"""RL best-response on gin (hand episodes).

The small-game rlbr.py cannot apply here (it needs information_state_tensor,
which gin does not have, and MaskedActorCritic). This is the gin path: a
fresh GinNet trains against one frozen opponent with the shared PPO update,
and the bound is the final-iterate duplicate-deal return. The opponent may
be a manual-phase engine bot: Knock/Layoff route through raw_step with
per-hand begins and inform-all listeners, mirroring the arena.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from ginrl.algos.regpg import Magnet, make_batch, ppo_update, sample_actions
from ginrl.config import HandConfig, Seeds, TrainerConfig
from ginrl.env.features import feature_dim
from ginrl.env.game import HandEnv
from ginrl.eval.arena import Arena
from ginrl.nets.gin import GinNet
from ginrl.telemetry.recorder import Recorder, RecorderConfig
from ginrl.train.driver import RolloutStep
from ginrl.train.gin import gin_observation_row, multi_hot
from ginrl.train.loop import save_checkpoint

_MANUAL_PHASES = frozenset({"Knock", "Layoff"})


def _seat_team(learner, opp, learner_seat: int) -> tuple:
    """Team tuple indexed by seat: teams[learner_seat] is always the learner."""
    return (learner, opp) if learner_seat == 0 else (opp, learner)


def _flip_seat(teams: list, learner_seats: list[int], learners: list, i: int) -> None:
    """Alternate the learner seat per hand so a perpetual second-seat can
    never starve collection entirely. (The learner-side agent object is never
    stepped — phase 1 uses batched actions — so only the opp side matters.)

    The learner object is fished by identity from the learners list, never by
    tuple position: a flip moves it, so position 0 is not the learner."""
    old_seat = learner_seats[i]
    assert teams[i][old_seat] is learners[i], "learner object lost from its seat"
    learner_seats[i] ^= 1
    teams[i] = _seat_team(learners[i], teams[i][1 - old_seat], learner_seats[i])


def _wire_hand(env: HandEnv, agents: tuple) -> None:
    def listener(acting_seat: int, state, player: int, action: int) -> None:
        for agent in agents:
            agent.inform(state, player, action)

    env.listener = listener
    env.manual_phases = {s for s, a in enumerate(agents) if getattr(a, "manual_phases", False)}
    agents[0].begin_game(0)
    agents[1].begin_game(1)


def _drive_hand(env: HandEnv, agents: tuple):
    seat = env.state().current_player()
    agent = agents[seat]
    if str(env.state().to_dict()["phase"]) in _MANUAL_PHASES and getattr(
        agent, "manual_phases", False
    ):
        return agent.choose_raw(env.state()), True
    return agent.choose(env, seat), False


def _apply_learner_actions(envs: list[HandEnv], order: list[int], actions) -> list:
    """Step each learner decision; returns [(env_idx, StepResult)] in order."""
    return [(i, envs[i].step(actions[k])) for k, i in enumerate(order)]


def _drain_to_learner(envs, learner_seats, teams, learners, reseed) -> None:
    """Step opponent turns until every env is waiting on the learner.

    reseed(i) supplies the reset seed for env i. The guard resets on hand
    boundaries: a turn-1 knock can pass a whole hand with no learner turn,
    which is legitimate — only a single endless hand trips.
    """
    for i, env in enumerate(envs):
        guard = 0
        while env.state().current_player() != learner_seats[i]:
            action, raw = _drive_hand(env, teams[i])
            result = env.raw_step(action) if raw else env.step(action)
            if result.done:
                # Flip first: _wire_hand binds each stateful bot to its new
                # seat via begin_game, so wiring must follow the flip.
                _flip_seat(teams, learner_seats, learners, i)
                env.reset(seed=reseed(i))
                _wire_hand(env, teams[i])
                guard = 0
            guard += 1
            assert guard < 10000, "BR drain failed to reach a learner turn"


def collect_br_hand(
    envs: list[HandEnv],
    learner_seats: list[int],
    teams: list[tuple],
    learners: list,
    net: GinNet,
    n_rows: int,
    gen: torch.Generator,
    device: torch.device,
    base_seed: int,
    feat_dim: int,
    deck: int,
    reward_scale: float,
) -> tuple[list[RolloutStep], list[float], list[float], int]:
    """One-sided rollout: rows for the learner seat only, hands are episodes."""
    for i, env in enumerate(envs):
        env.reset(seed=base_seed + i)
        _wire_hand(env, teams[i])
    steps: list[RolloutStep] = []
    logps: list[float] = []
    values: list[float] = []
    net.eval()
    while len(steps) < n_rows:
        feats, masks, seats, opps, order = [], [], [], [], []
        for i, env in enumerate(envs):
            seat = env.state().current_player()
            if seat == learner_seats[i]:
                obs, mask, _ = gin_observation_row(env, seat, feat_dim)
                feats.append(obs)
                masks.append(mask)
                seats.append(seat)
                opps.append(multi_hot(env.hidden_opp_hand(seat), deck))
                order.append(i)
        if order:
            with torch.no_grad():
                logits, value, _ = net(
                    torch.tensor(feats, dtype=torch.float32, device=device),
                    torch.tensor(masks, dtype=torch.bool, device=device),
                )
            actions, blogp = sample_actions(logits.cpu(), gen)
            outcomes = _apply_learner_actions(envs, order, actions)
            for k, (i, result) in enumerate(outcomes):
                steps.append(
                    RolloutStep(
                        table=i,
                        obs=feats[k],
                        mask=masks[k],
                        action=actions[k],
                        reward=(result.returns[seats[k]] if result.done else 0.0) * reward_scale,
                        done=result.done,
                        seat=seats[k],
                        opp_card=-1,
                        opp_hand=opps[k],
                    )
                )
                logps.append(float(blogp[k]))
                values.append(float(value[k].cpu()))
                if result.done:
                    envs[i].reset(seed=base_seed + len(steps) + i)
                    _wire_hand(envs[i], teams[i])
        _drain_to_learner(
            envs,
            learner_seats,
            teams,
            learners,
            lambda i: base_seed + len(steps) + i + 999983,
        )
    net.train()
    return steps, logps, values, base_seed + len(steps)


def train_br_gin(
    fixed_factory,
    fixed_name: str,
    hand_config: HandConfig | None = None,
    torso: str = "mlp",
    cfg: TrainerConfig | None = None,
    master_seed: int = 0,
    device: torch.device | None = None,
    run_dir: Path | None = None,
    reward_scale: float = 0.02,
    eval_deals: int = 500,
    agent_name: str = "br",
) -> float:
    """Train a best response vs one frozen opponent; return final pph."""
    from ginrl.agents.learned import GinNetAgent

    cfg = cfg or TrainerConfig()
    dev = device or torch.device("cpu")
    run = run_dir or Path(f"runs/br-{fixed_name}-{master_seed}")
    run.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(RecorderConfig(run_name=run.name, run_dir=run))
    torch.manual_seed(master_seed)
    torch.set_num_threads(1)
    config = hand_config or HandConfig()
    deck = config.deck_size
    feat_dim = feature_dim(deck)
    seeds = Seeds(master=master_seed)
    envs = [HandEnv(config, seeds) for _ in range(cfg.n_envs)]
    rng = random.Random(master_seed + 5)
    learner_seats = [rng.randrange(2) for _ in envs]
    net = GinNet(torso, feat_dim=feat_dim, deck=deck, hidden=128).to(dev)
    learners = [GinNetAgent(net.eval(), feat_dim, dev, seed=0) for _ in envs]
    teams = [_seat_team(learners[i], fixed_factory(), learner_seats[i]) for i in range(len(envs))]
    # NOTE: the learner-side GinNetAgents above share weights with `net` by
    # construction (same module); they only route acts, training reads `net`.
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    magnet = Magnet(net, cfg)
    gen = torch.Generator().manual_seed(master_seed + 1)
    steps_done, deal_round, base_seed = 0, 0, master_seed * 1_000_000 + 7
    last_stats: dict[str, float] = {}
    while steps_done < cfg.total_steps:
        steps, logps, values, base_seed = collect_br_hand(
            envs,
            learner_seats,
            teams,
            learners,
            net,
            cfg.batch_size,
            gen,
            dev,
            base_seed,
            feat_dim,
            deck,
            reward_scale,
        )
        # New hands, new learner seats (keeps both seats' play represented).
        # The learner object comes from the learners list by identity — never
        # from the team tuple by position, which flips move.
        for i in range(len(envs)):
            learner_seats[i] = rng.randrange(2)
            teams[i] = _seat_team(learners[i], fixed_factory(), learner_seats[i])
        batch = make_batch(steps, logps, values, cfg, dev)
        last_stats = ppo_update(net, optimizer, batch, cfg, magnet, gen, steps_done)
        steps_done += len(steps)
        deal_round += 1
        save_checkpoint(
            run / "checkpoint.pt",
            net,
            optimizer,
            magnet,
            gen,
            steps_done,
            deal_round,
            base_seed,
        )
    torch.save(net.state_dict(), run / "final.pt")
    for name, metric in {
        "loss/total": "loss/total",
        "entropy": "policy/entropy",
    }.items():
        if name in last_stats:
            recorder.log_scalar(steps_done, metric, last_stats[name])
    br_agent = GinNetAgent(net.to(dev).eval(), feat_dim, dev, seed=master_seed)
    br_agent.name = agent_name
    arena = Arena(config=config, seeds=Seeds(master_seed))
    summary = arena.duplicate_summary(br_agent, fixed_factory(), master_seed, eval_deals).checked(0)
    pph = summary.points_per_hand()
    recorder.log_scalar(steps_done, "ratings/paired_score", pph.mean)
    recorder.close()
    return pph.mean
