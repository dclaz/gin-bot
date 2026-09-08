"""RL-BR mechanics (fast): BR-only rows, fixed-policy determinism, smoke run."""

from __future__ import annotations

import random

import torch

from ginrl.config import Seeds, TrainerConfig
from ginrl.eval.exploitability import load_small_game
from ginrl.eval.rlbr import (
    attach_terminal,
    collect_br,
    simulate_return,
    train_br,
    uniform_fixed,
)
from ginrl.nets.actor_critic import MaskedActorCritic
from ginrl.train.driver import RolloutStep, SmallGameVecEnv, net_config_for_game

DEVICE = torch.device("cpu")


def test_uniform_fixed_is_legal_and_seeded() -> None:
    mask = (True, False, True)
    assert uniform_fixed((0.5,), mask, random.Random(0)) in (0, 2)
    assert uniform_fixed((0.5,), mask, random.Random(0)) == uniform_fixed(
        (0.5,), mask, random.Random(0)
    )


def test_collect_br_records_only_br_seat() -> None:
    torch.manual_seed(0)
    env = SmallGameVecEnv("kuhn_poker", n_envs=4, seeds=Seeds(master=5))
    net = MaskedActorCritic(net_config_for_game("kuhn_poker", hidden=8, layers=1))
    gen = torch.Generator().manual_seed(1)
    for br_seat in (0, 1):
        steps, logps, values, nxt, fixed_terminals = collect_br(
            env,
            net,
            br_seat,
            uniform_fixed,
            random.Random(2),
            24,
            gen,
            DEVICE,
            100 + br_seat,
            5_000,
            0.99,
        )
        assert len(steps) >= 24 and len(logps) == len(values) == len(steps)
        assert {s.seat for s in steps} == {br_seat}
        assert all(v == v for v in logps + values)
        assert nxt > 5_000
        # BR-done rewards plus negated fixed-done rewards tile every hand.
        n_hands = sum(1 for s in steps if s.done) + len(fixed_terminals)
        assert n_hands > 0
        hand_mean = (sum(s.reward for s in steps if s.done) + sum(fixed_terminals)) / n_hands
        assert -2.0 <= hand_mean <= 2.0  # Kuhn payoff range, no conditioning blow-up


def _row(table: int, reward: float = 0.0, done: bool = False) -> RolloutStep:
    return RolloutStep(table, (0.0,), (True,), 0, reward, done, 0, 0)


def test_attach_terminal_credits_last_br_row() -> None:
    steps = [_row(0), _row(1), _row(0)]
    last_br = {0: 2, 1: 1}
    assert attach_terminal(steps, last_br, 0, fixed_reward=2.0, gamma=0.99) is True
    # Zero-sum negation, one ply discounted; the row now ends its hand.
    assert steps[2].reward == 0.99 * -2.0
    assert steps[2].done is True
    assert last_br == {1: 1}
    # A table the BR never acted on has no row to credit.
    assert attach_terminal(steps, last_br, 7, fixed_reward=1.0, gamma=0.99) is False
    assert steps[0].reward == 0.0 and steps[0].done is False


def test_simulate_return_is_bounded_and_seeded() -> None:
    torch.manual_seed(0)
    game = load_small_game("kuhn_poker")
    net = MaskedActorCritic(net_config_for_game("kuhn_poker", hidden=8, layers=1))
    for seat in (0, 1):
        mean = simulate_return(game, net, seat, uniform_fixed, 50, seed=0)
        assert -2.0 <= mean <= 2.0  # Kuhn payoff range
        assert mean == simulate_return(game, net, seat, uniform_fixed, 50, seed=0)


def test_train_br_smoke_runs_and_scores_finite() -> None:
    cfg = TrainerConfig(
        total_steps=300,
        n_envs=2,
        rollout_len=16,
        epochs=1,
        minibatches=2,
        reg_coef=0.01,
        seeds=Seeds(master=9),
    )
    net, mean_return = train_br(
        "kuhn_poker",
        uniform_fixed,
        0,
        cfg,
        master_seed=9,
        device=DEVICE,
        make_net=lambda: MaskedActorCritic(net_config_for_game("kuhn_poker", hidden=8, layers=1)),
    )
    assert mean_return == mean_return  # finite, no crash
    assert all(p.isfinite().all() for p in net.parameters())


def test_acting_seat_none_after_terminal() -> None:
    env = SmallGameVecEnv("kuhn_poker", n_envs=1, seeds=Seeds(master=0))
    env.reset_all(base_seed=0)
    assert env.acting_seat(0) in (0, 1)
    rng = random.Random(0)
    for _ in range(10):
        if env.pending_returns(0) is not None:
            break
        env.act([rng.choice(env._states[0].legal_actions())])
    assert env.pending_returns(0) is not None
    assert env.acting_seat(0) is None
