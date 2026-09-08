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

import copy
import random
from collections.abc import Callable
from dataclasses import replace

import pyspiel
import torch

from ginrl.algos.regpg import Magnet, make_batch, ppo_update, sample_actions
from ginrl.config import Seeds, TrainerConfig
from ginrl.nets.actor_critic import MaskedActorCritic
from ginrl.train.driver import RolloutStep, SmallGameVecEnv

# Fixed opponent: maps (obs, legal mask) to an action. Closes over its own
# RNG (and, for a frozen net, its own weights and sampling generator).
FixedPolicy = Callable[[tuple[float, ...], tuple[bool, ...], random.Random], int]


def uniform_fixed(obs: tuple[float, ...], mask: tuple[bool, ...], rng: random.Random) -> int:
    """Uniform random over legal actions (the calibration target)."""
    legal = [a for a, ok in enumerate(mask) if ok]
    return rng.choice(legal)


def net_fixed(net: MaskedActorCritic, device: torch.device, seed: int) -> FixedPolicy:
    """Freeze `net` as a sampling opponent (alternating-BR phases)."""
    frozen = copy.deepcopy(net).requires_grad_(False).to(device).eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def frozen_policy(obs: tuple[float, ...], mask: tuple[bool, ...], rng: random.Random) -> int:
        obs_t = torch.tensor([obs], dtype=torch.float32, device=device)
        mask_t = torch.tensor([mask], dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, _, _ = frozen(obs_t, mask_t)
        probs = torch.softmax(logits.cpu(), dim=-1)
        return int(torch.multinomial(probs, 1, generator=gen).squeeze())

    return frozen_policy


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
            chosen[i] = fixed(seen[k][0], seen[k][1], fixed_rng)
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
            chosen[i] = fixed(seen[k][0], seen[k][1], fixed_rng)
    return chosen, learn_slot, br_logp, value


def attach_terminal(
    steps: list[RolloutStep],
    last_br: dict[int, int],
    table: int,
    fixed_reward: float,
    gamma: float,
) -> bool:
    """Credit a fixed-ended hand to the BR's last row of that hand.

    The BR's gradient rows would otherwise end mid-hand with reward 0 and
    `done=False`: the hand's terminal reward would never enter GAE, and the
    table stream would bootstrap across the hand boundary into the next
    hand. Returns True when a row was credited (False: the BR never acted
    in this hand, so there is no row to learn from).
    """
    if table not in last_br:
        return False
    j = last_br.pop(table)
    steps[j] = replace(
        steps[j],
        reward=steps[j].reward + gamma * -fixed_reward,
        done=True,
    )
    return True


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
    gamma: float,
) -> tuple[list[RolloutStep], list[float], list[float], int, list[float]]:
    """Rollout where only `br_seat` learns; the other seat plays `fixed`.

    Returns (BR-seat steps, behaviour logps, behaviour values, next seed,
    BR returns from hands the BR never acted in). Every hand the BR acted
    in ends in a BR row with `done=True` carrying the full terminal return
    (hands the fixed seat closes are credited via `attach_terminal`, one
    ply discounted); `fixed_terminals` covers only hands with no BR row,
    so the two tilings together score every hand exactly once. Without the
    back-fill, the BR mean — and worse, its GAE targets — would condition
    on "BR moves last", a lying lens (a slowplaying BR moves last exactly
    when it wins).
    """
    env.reset_all(base_seed=base_seed)
    steps: list[RolloutStep] = []
    logps: list[float] = []
    values: list[float] = []
    fixed_terminals: list[float] = []
    last_br: dict[int, int] = {}
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
                if step.done:
                    # By zero-sum the BR's return is the negation: credit
                    # the BR's last row so GAE sees the terminal. Only
                    # hands with no BR row at all land in fixed_terminals,
                    # so the two tilings score every hand exactly once.
                    if not attach_terminal(steps, last_br, step.table, step.reward, gamma):
                        fixed_terminals.append(-step.reward)
                continue
            k = at[step.table]
            steps.append(step)
            logps.append(float(br_logp[learn_slot[k]]))
            values.append(float(value[k].cpu()))
            if step.done:
                last_br.pop(step.table, None)
            else:
                last_br[step.table] = len(steps) - 1
        for i in range(env.n_envs):
            if env.pending_returns(i) is not None:
                redeal_seed += 1
                env.reset_table(i, redeal_seed)
                last_br.pop(i, None)
    net.train()
    return steps, logps, values, redeal_seed, fixed_terminals


def run_br_phase(
    env: SmallGameVecEnv,
    net: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    magnet: Magnet,
    br_seat: int,
    fixed: FixedPolicy,
    fixed_rng: random.Random,
    cfg: TrainerConfig,
    gen: torch.Generator,
    device: torch.device,
    base_seed: int,
    redeal_seed: int,
    budget_steps: int,
    steps_done: int,
) -> tuple[float, int, int, dict[str, float]]:
    """Train `br_seat` against `fixed` for `budget_steps` decisions.

    Carries the caller's net/optimizer/magnet (shared across alternating
    phases). Returns (true per-hand BR mean, next redeal seed, phase steps,
    last update stats).
    """
    phase_steps, deal_round, returns = 0, 0, []
    last_stats: dict[str, float] = {}
    while phase_steps < budget_steps:
        steps, logps, values, redeal_seed, fixed_terminals = collect_br(
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
            cfg.gamma,
        )
        deal_round += 1
        if not steps:
            continue
        batch = make_batch(steps, logps, values, cfg, device)
        last_stats = ppo_update(net, optimizer, batch, cfg, magnet, gen, steps_done + phase_steps)
        phase_steps += len(steps)
        # Every hand the BR acted in ends in a BR done-row (fixed-closed
        # hands are back-filled); fixed_terminals holds only hands with no
        # BR row. Together they tile all hands, so this mean is the BR's
        # true per-hand return, not the BR-moves-last conditional.
        returns.extend(s.reward for s in steps if s.done)
        returns.extend(fixed_terminals)
    mean_return = sum(returns) / len(returns) if returns else 0.0
    return mean_return, redeal_seed, phase_steps, last_stats


def simulate_return(
    game: pyspiel.Game,
    net: MaskedActorCritic,
    seat: int,
    fixed: FixedPolicy,
    n_deals: int,
    seed: int,
) -> float:
    """Final-iterate mean return of `net` (sampled) vs `fixed`, `n_deals` hands.

    Training-time means average the exploring behaviour policy; calibration
    compares final weights, so the gate measures here, not there. Both seats
    observe via `information_state_tensor`, matching training.
    """
    net.eval()
    total = 0.0
    rng = random.Random(seed)
    for _ in range(n_deals):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                pairs = state.chance_outcomes()
                total_p = sum(p for _, p in pairs)
                roll, acc = rng.random() * total_p, 0.0
                for action, prob in pairs:
                    acc += prob
                    if acc >= roll:
                        state.apply_action(action)
                        break
                continue
            player = state.current_player()
            if player == seat:
                obs = torch.tensor([state.information_state_tensor(seat)], dtype=torch.float32)
                mask = torch.tensor([state.legal_actions_mask(seat)], dtype=torch.bool)
                with torch.no_grad():
                    logits, _, _ = net(obs, mask)
                probs = torch.softmax(logits.squeeze(0), dim=-1).tolist()
                legal = state.legal_actions()
                total_p = sum(probs[a] for a in legal)
                roll, acc = rng.random() * total_p, 0.0
                for action in legal:
                    acc += probs[action]
                    if acc >= roll:
                        state.apply_action(action)
                        break
            else:
                state.apply_action(
                    fixed(
                        tuple(state.information_state_tensor(player)),
                        tuple(bool(x) for x in state.legal_actions_mask(player)),
                        rng,
                    )
                )
        total += state.player_return(seat)
    return total / n_deals if n_deals > 0 else 0.0


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
    mean_return, _, _, _ = run_br_phase(
        env,
        net,
        optimizer,
        magnet,
        br_seat,
        fixed,
        fixed_rng,
        cfg,
        gen,
        device,
        base_seed=master_seed * 1_000,
        redeal_seed=master_seed * 1_000_000 + 7,
        budget_steps=cfg.total_steps,
        steps_done=0,
    )
    return net, mean_return
