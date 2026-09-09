"""Trainer mechanics: advantages, magnet modes, PPO update, measurement chain."""

from __future__ import annotations

import pytest
import torch

from ginrl.algos.regpg import (
    Magnet,
    advantages,
    collect,
    linear_factor,
    make_batch,
    ppo_update,
    sample_actions,
)
from ginrl.config import MAGNET_EMA, MAGNET_SNAPSHOT, MAGNET_UNIFORM, Seeds, TrainerConfig
from ginrl.eval.exploitability import NetPolicy, load_small_game, nash_conv
from ginrl.nets.actor_critic import MaskedActorCritic, NetConfig
from ginrl.train.driver import RolloutStep, SmallGameVecEnv, net_config_for_game

DEVICE = torch.device("cpu")


def toy_steps() -> tuple[list[RolloutStep], list[float]]:
    """One table, one 2-decision episode, reward 1 at the end."""
    steps = [
        RolloutStep(0, (0.0, 0.0, 0.0, 0.0), (True, True), 0, 0.0, False, 0, 0),
        RolloutStep(0, (1.0, 0.0, 0.0, 0.0), (True, True), 1, 1.0, True, 1, 1),
    ]
    return steps, [0.0, 0.0]


def test_gae_and_mc_match_hand_computation() -> None:
    # Zero-sum stream: seat 0 moves (r=0), then seat 1 moves and wins (+1),
    # so seat 0's return is -1 and seat 1's is +1. Cross-seat rows must be
    # negated, not summed raw — that sign flip is the regression under test.
    steps, values = toy_steps()
    adv, ret = advantages(steps, values, gamma=1.0, lam=1.0, mode="mc")
    assert adv == [-1.0, 1.0] and ret == [-1.0, 1.0]
    adv, ret = advantages(steps, values, gamma=1.0, lam=1.0, mode="gae")
    assert adv == [-1.0, 1.0] and ret == [-1.0, 1.0]
    adv, _ = advantages(steps, values, gamma=1.0, lam=0.0, mode="gae")
    assert adv == pytest.approx([0.0, 1.0])  # TD(0): no bootstrapped credit yet
    with pytest.raises(ValueError):
        advantages(steps, values, 0.99, 0.95, "td-lambda")


def test_advantages_split_episodes_per_table() -> None:
    steps, values = toy_steps()
    other = RolloutStep(1, (0.5, 0.0, 0.0, 0.0), (True, True), 0, 2.0, True, 0, 1)
    adv, ret = advantages([*steps, other], [*values, 0.0], 1.0, 1.0, "mc")
    assert (adv[2], ret[2]) == (2.0, 2.0)  # table 1 is its own episode
    assert adv[0] == -1.0  # seat 0 lost the hand seat 1 won


def test_linear_factor_boundaries() -> None:
    assert linear_factor(0, 100) == 1.0
    assert linear_factor(50, 100) == 0.5
    assert linear_factor(100, 100) == 0.0
    assert linear_factor(200, 100) == 0.0


def tiny_net() -> MaskedActorCritic:
    torch.manual_seed(0)
    return MaskedActorCritic(NetConfig(obs_dim=4, n_actions=2, aux_dim=2, hidden=8, layers=1))


def tiny_cfg(**over: object) -> TrainerConfig:
    return TrainerConfig(
        total_steps=100,
        n_envs=2,
        rollout_len=8,
        epochs=1,
        minibatches=2,
        **over,  # type: ignore[arg-type]
    )


def test_uniform_magnet_kl_is_zero_at_uniform() -> None:
    net = tiny_net()
    with torch.no_grad():
        net.policy_head.weight.zero_()
        net.policy_head.bias.zero_()
    magnet = Magnet(net, tiny_cfg(magnet_mode=MAGNET_UNIFORM))
    obs = torch.zeros(3, 4)
    mask = torch.ones(3, 2, dtype=torch.bool)
    logits, _, _ = net(obs, mask)
    assert torch.allclose(magnet.kl(logits, mask, net, obs), torch.zeros(3), atol=1e-6)


def test_kl_term_differentiates_policy_not_magnet() -> None:
    """The KL gradient is the regulariser: it must reach the online net.

    Regression test: kl() used to run under no_grad with detached logits,
    so every magnet mode trained exactly like the reg_coef=0 control and
    the ablation guard could not have caught anything.
    """
    for mode, kwargs in (
        (MAGNET_UNIFORM, {}),
        (MAGNET_SNAPSHOT, {"snapshot_every": 1}),
        (MAGNET_EMA, {}),
    ):
        net = tiny_net()
        magnet = Magnet(net, tiny_cfg(magnet_mode=mode, **kwargs))  # type: ignore[arg-type]
        # Move the learner away from a freshly copied reference. KL is
        # stationary at an exact match, so without this divergence the test
        # can depend on platform floating-point dust rather than the gradient.
        with torch.no_grad():
            net.policy_head.bias[0] += 0.25
        obs = torch.randn(4, 4)
        mask = torch.ones(4, 2, dtype=torch.bool)
        logits, _, _ = net(obs, mask)
        magnet.kl(logits, mask, net, obs).mean().backward()
        assert net.policy_head.weight.grad is not None
        assert bool((net.policy_head.weight.grad != 0).any())
        if magnet.ref is not None:
            assert all(p.grad is None for p in magnet.ref.parameters())


def test_snapshot_refreshes_and_ema_drifts() -> None:
    net = tiny_net()
    snap = Magnet(net, tiny_cfg(magnet_mode=MAGNET_SNAPSHOT, snapshot_every=2))
    ref_before = [p.clone() for p in snap.ref.parameters()]  # type: ignore[union-attr]
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    snap.post_update(net)  # update 1: no refresh yet
    for a, b in zip(ref_before, snap.ref.parameters(), strict=True):  # type: ignore[union-attr]
        assert torch.equal(a, b)
    snap.post_update(net)  # update 2: snapshot copies the learner
    for a, b in zip(net.parameters(), snap.ref.parameters(), strict=True):  # type: ignore[union-attr]
        assert torch.equal(a, b)

    ema = Magnet(net, tiny_cfg(magnet_mode=MAGNET_EMA, magnet_ema_decay=0.5))
    with torch.no_grad():
        for p in net.parameters():
            p.add_(2.0)
    ema.post_update(net)
    for ref_p, net_p in zip(ema.ref.parameters(), net.parameters(), strict=True):  # type: ignore[union-attr]
        assert torch.all(ref_p < net_p) and torch.all(ref_p > net_p - 2.0)


def test_ppo_update_is_finite_and_deterministic() -> None:
    torch.set_num_threads(1)
    cfg = tiny_cfg()
    runs = []
    for _ in range(2):
        torch.manual_seed(11)
        net = tiny_net()
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        gen = torch.Generator().manual_seed(5)
        batch_steps, vals = toy_steps()
        steps = batch_steps * 4
        batch = make_batch(steps, [0.0] * len(steps), [0.0] * len(steps), cfg, DEVICE)
        assert batch.adv.shape == (len(steps),)
        stats = ppo_update(net, opt, batch, cfg, Magnet(net, cfg), gen, steps_done=0)
        assert all(v == v and abs(v) < 1e6 for v in stats.values())  # finite
        runs.append([p.clone() for p in net.parameters()])
    for a, b in zip(runs[0], runs[1], strict=True):
        assert torch.equal(a, b)  # bit-identical rerun on CPU


def test_collect_returns_behaviour_triples_and_chains_seeds() -> None:
    torch.manual_seed(2)
    env = SmallGameVecEnv("kuhn_poker", n_envs=4, seeds=Seeds(master=3))
    net = MaskedActorCritic(net_config_for_game("kuhn_poker", hidden=8, layers=1))
    gen = torch.Generator().manual_seed(9)
    steps, logps, values, nxt = collect(env, net, 32, gen, DEVICE, 0, 1_000)
    assert len(steps) >= 32 and len(logps) == len(values) == len(steps)
    assert all(v == v for v in logps + values)
    assert nxt > 1_000
    assert {s.table for s in steps} == {0, 1, 2, 3}


def test_sample_actions_respects_mask_and_seed() -> None:
    logits = torch.tensor([[0.0, 0.0, -1e9]])
    gen = torch.Generator().manual_seed(0)
    actions, logps = sample_actions(logits, gen)
    assert int(actions[0]) in (0, 1)
    gen2 = torch.Generator().manual_seed(0)
    assert int(sample_actions(logits, gen2)[0][0]) == int(actions[0])


def test_uniform_net_recovers_uniform_nash_conv() -> None:
    """The measurement chain is exact: a uniform net reads 0.9167 on Kuhn."""
    game = load_small_game("kuhn_poker")
    torch.manual_seed(4)
    net = MaskedActorCritic(net_config_for_game("kuhn_poker", hidden=8, layers=1))
    with torch.no_grad():
        net.policy_head.weight.zero_()
        net.policy_head.bias.zero_()
    assert nash_conv(game, NetPolicy(game, net, DEVICE)) == pytest.approx(0.9167, abs=1e-4)
