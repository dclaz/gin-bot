"""RL best response: the only exploitability bound gin rummy gets (landmine 5).

A best response is itself a training run — undertrained, it reads as "the
target is hard to exploit". So RL-BR is built and calibrated here, on Kuhn
and Leduc, where its answer is checkable against exact `nash_conv`, before
it is ever pointed at gin. Calibration lives in `gate-p3`, not in the unit
tests (training takes minutes; tests stay under 60s).

Protocol: freeze the target policy, train one PPO seat against it with the
same machinery as self-play (`collect_br` records only the BR seat's rows),
then report the BR's mean return per seat. Estimated NashConv is the sum of
both seats' BR gains over the target's own seat values.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import torch

from ginrl.algos.regpg import Magnet, make_batch, ppo_update, sample_actions
from ginrl.config import Seeds, TrainerConfig
from ginrl.nets.actor_critic import MaskedActorCritic
from ginrl.train.driver import RolloutStep, SmallGameVecEnv

# Fixed opponent: maps (legal mask) to an action. Closes over its own RNG.
FixedPolicy = Callable[[tuple[bool, ...], random.Random], int]


def uniform_fixed(mask: tuple[bool, ...], rng: random.Random) -> int:
    """Uniform random over legal actions (the calibration target)."""
    legal = [a for a, ok in enumerate(mask) if ok]
    return rng.choice(legal)


@torch.no_grad()
def _choose_actions(
    net: MaskedActorCritic,
    live: list[int],
    seats: list[int | None],
    seen: list[tuple[tuple[float, ...], tuple[bool, ...]]],
    br_seat: int,
    fixed: FixedPolicy,
    fixed_rng: random.Random,
    gen: torch.Generator,
    device: torch.device,
) -> tuple[dict[int, int], dict[int, int], torch.Tensor, torch.Tensor]:
    """Map tables to actions; BR-seat slots also return (logp, value) slots.

    Returns (chosen actions, forward-slot per BR table, br logps, br values).
    """
    learn_slot = {k: j for j, k in enumerate(q for q, i in enumerate(live) if seats[q] == br_seat)}
    chosen: dict[int, int] = {}
    if not learn_slot:
        for k, i in enumerate(live):
            chosen[i] = fixed(seen[k][1], fixed_rng)
        return chosen, {}, torch.zeros(0), torch.zeros(0)
    obs = torch.tensor([o for o, _ in seen], dtype=torch.float32, device=device)
    masks = torch.tensor([m for _, m in seen], dtype=torch.bool, device=device)
    logits, value, _ = net(obs, masks)
    br_rows = [k for k in range(len(live)) if k in learn_slot]
    br_actions, br_logp = sample_actions(logits[br_rows].cpu(), gen)
    for k, i in enumerate(live):
        if k in learn_slot:
            chosen[i] = int(br_actions[learn_slot[k]])
        else:
            chosen[i] = fixed(seen[k][1], fixed_rng)
    return chosen, learn_slot, br_logp, value


@torch.no_grad()
def collect_br(
    env: SmallGameVecEnv,
    net: MaskedActorCritic,
    br_seat: int,
    fixed: FixedPolicy,
    fixed_rng: random.Random,
    n_rows: int,
    gen: torch.Generator,
    device: torch.device,
    base_seed: int,
    redeal_seed: int,
) -> tuple[list[RolloutStep], list[float], list[float], int]:
    """Rollout where only `br_seat` learns; the other seat plays `fixed`.

    Returns (BR-seat steps, behaviour logps, behaviour values, next seed).
    The fixed seat's rows are discarded — they carry no gradient for the BR.
    """
    env.reset_all(base_seed=base_seed)
    steps: list[RolloutStep] = []
    logps: list[float] = []
    values: list[float] = []
    net.eval()
    while len(steps) < n_rows:
        live = [i for i in range(env.n_envs) if env.pending_returns(i) is None]
        if not live:
            break
        seats = [env.acting_seat(i) for i in live]
        seen = [env.observe(i) for i in live]
        chosen, learn_slot, br_logp, value = _choose_actions(
            net, live, seats, seen, br_seat, fixed, fixed_rng, gen, device
        )
        at = {table: k for k, table in enumerate(live)}
        rollout = env.act([chosen[i] if i in at else None for i in range(env.n_envs)])
        for step in rollout.steps:
            if step.seat != br_seat:
                continue
            k = at[step.table]
            steps.append(step)
            logps.append(float(br_logp[learn_slot[k]]))
            values.append(float(value[k].cpu()))
        for i in range(env.n_envs):
            if env.pending_returns(i) is not None:
                redeal_seed += 1
                env.reset_table(i, redeal_seed)
    net.train()
    return steps, logps, values, redeal_seed


def train_br(
    game_name: str,
    fixed: FixedPolicy,
    br_seat: int,
    cfg: TrainerConfig,
    master_seed: int,
    device: torch.device,
    make_net: Callable[[], MaskedActorCritic],
) -> tuple[MaskedActorCritic, float]:
    """Train a best response for one seat; return (net, mean BR-seat return)."""
    seeds = Seeds(master=master_seed)
    env = SmallGameVecEnv(game_name, n_envs=cfg.n_envs, seeds=seeds)
    torch.manual_seed(master_seed)
    net = make_net().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    magnet = Magnet(net, cfg)
    gen = torch.Generator().manual_seed(master_seed + 1)
    fixed_rng = seeds.spawn(f"rlbr-fixed-{br_seat}")
    base_seed = master_seed * 1_000
    redeal_seed = master_seed * 1_000_000 + 7
    steps_done, deal_round, returns = 0, 0, []
    while steps_done < cfg.total_steps:
        steps, logps, values, redeal_seed = collect_br(
            env,
            net,
            br_seat,
            fixed,
            fixed_rng,
            cfg.batch_size,
            gen,
            device,
            base_seed + deal_round,
            redeal_seed,
        )
        deal_round += 1
        if not steps:
            continue
        batch = make_batch(steps, logps, values, cfg, device)
        ppo_update(net, optimizer, batch, cfg, magnet, gen, steps_done)
        steps_done += len(steps)
        returns.extend(s.reward for s in steps if s.done)
    mean_return = sum(returns) / len(returns) if returns else 0.0
    return net, mean_return
