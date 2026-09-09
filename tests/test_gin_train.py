"""Phase 4 training-path unit tests (fast, no learning)."""

import torch

from ginrl.agents.learned import BeliefAblatedAgent, GinNetAgent, gin_observation
from ginrl.config import REDUCED_HAND_CONFIG, Seeds
from ginrl.env.features import EVENT_PAD, EVENT_WINDOW, feature_dim
from ginrl.env.game import HandEnv
from ginrl.eval.belief import (
    auc_score,
    belief_auc,
    collect_belief_states,
    undetermined_mask,
)
from ginrl.nets.gin import GIN_TORSOS, GinNet
from ginrl.train.driver import RolloutStep
from ginrl.train.gin import collect_gin


def _net(torso: str = "mlp") -> GinNet:
    torch.manual_seed(0)
    return GinNet(torso, feat_dim=feature_dim(10), deck=10, hidden=16)


def test_event_log_records_public_moves_only() -> None:
    env = HandEnv(REDUCED_HAND_CONFIG, Seeds(5))
    result = env.reset(seed=5)
    assert env.trackers[0].event_window() == (EVENT_PAD,) * EVENT_WINDOW
    seat = result.seat
    env.step(54 if result.mask[54] else 52)
    log = env.trackers[seat].event_window()
    assert log[-1] != EVENT_PAD
    assert len(log) == EVENT_WINDOW


def test_gin_torsos_forward_shapes() -> None:
    for torso in GIN_TORSOS:
        net = _net(torso)
        obs = torch.zeros(3, net.obs_dim)
        obs[:, feature_dim(10) : feature_dim(10) + EVENT_WINDOW] = float(EVENT_PAD)
        mask = torch.ones(3, 56, dtype=torch.bool)
        logits, value, aux = net(obs, mask)
        assert logits.shape == (3, 56)
        assert value.shape == (3,)
        assert aux.shape == (3, 10)


def test_collect_gin_rows_carry_joint_labels() -> None:
    torch.manual_seed(0)
    net = _net()
    gen = torch.Generator().manual_seed(0)
    envs = [HandEnv(REDUCED_HAND_CONFIG, Seeds(0)) for _ in range(2)]
    steps, logps, values, _ = collect_gin(
        envs, net, 32, gen, torch.device("cpu"), 0, 7, feature_dim(10)
    )
    assert len(steps) == len(logps) == len(values) == 32
    for step in steps:
        assert isinstance(step, RolloutStep)
        assert step.opp_hand is not None and len(step.opp_hand) == 10
        assert sum(step.opp_hand) <= 4.0  # reduced hand size 3 + drawn card
    hands = sum(1 for s in steps if s.done)
    assert hands >= 1
    for step in steps:
        if step.done:
            assert isinstance(step.reward, float)


def test_belief_states_and_random_net_auc() -> None:
    states = collect_belief_states(REDUCED_HAND_CONFIG, 40, seed=1000)
    assert len(states) == 40
    assert all(len(s.undetermined) == 10 and any(s.undetermined) for s in states)
    net = _net()
    aux_auc, base_auc = belief_auc(net, states, torch.device("cpu"))
    assert base_auc == 0.5
    assert 0.0 <= aux_auc <= 1.0


def test_undetermined_mask_excludes_seen_cards() -> None:
    env = HandEnv(REDUCED_HAND_CONFIG, Seeds(3))
    result = env.reset(seed=3)
    mask = undetermined_mask(env, result.seat)
    assert len(mask) == 10
    # Own hand is always determined-absent.
    own = env.state().to_dict()["hands"][result.seat]
    from ginrl.env.melds import CardLayout

    layout = CardLayout(5, 2)
    for card in own:
        assert mask[layout.card_to_index(card)] is False


def test_learned_agents_choose_legal() -> None:
    net = _net()
    device = torch.device("cpu")
    env = HandEnv(REDUCED_HAND_CONFIG, Seeds(9))
    result = env.reset(seed=9)
    for agent in (
        GinNetAgent(net, feature_dim(10), device, seed=1),
        BeliefAblatedAgent(net, feature_dim(10), device, seed=1),
    ):
        agent.begin_game(result.seat)
        action = agent.choose(env, result.seat)
        assert result.mask[action]
    assert len(gin_observation(env, result.seat, feature_dim(10))) == net.obs_dim


def test_auc_score_separates_and_ties() -> None:
    assert auc_score([0.9, 0.1, 0.8, 0.2], [1.0, 0.0, 1.0, 0.0]) == 1.0
    assert auc_score([0.5, 0.5], [1.0, 0.0]) == 0.5
    assert auc_score([0.1, 0.9], [1.0, 0.0]) == 0.0
