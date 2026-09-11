"""Agents driven by trained gin networks (Phase 4).

GinNetAgent plays a HandEnv through the Agent protocol: per decision it
builds the same concatenated observation the trainer uses (belief
features, public-event window, raw tensor), forwards the net, and samples.
BeliefAblatedAgent wraps the same weights but zeroes the belief blocks,
for the decision-value ablation (uniform-beliefs control).
"""

from __future__ import annotations

import random

import numpy as np
import torch

from ginrl.env.features import EVENT_WINDOW
from ginrl.env.game import N_LEARNED_ACTIONS, HandEnv
from ginrl.nets.gin import RAW_DIM, GinNet, gin_obs_slices


def gin_observation(env: HandEnv, seat: int, feat_dim: int) -> list[float]:
    """Concatenated actor observation: belief, events, raw tensor."""
    f_slice, _, _ = gin_obs_slices(feat_dim, EVENT_WINDOW, RAW_DIM)
    assert f_slice.stop == feat_dim
    obs = [0.0] * (feat_dim + EVENT_WINDOW + RAW_DIM)
    obs[:feat_dim] = env.features_for(seat)
    obs[feat_dim : feat_dim + EVENT_WINDOW] = [float(t) for t in env.trackers[seat].event_window()]
    obs[feat_dim + EVENT_WINDOW :] = list(env.tensor_for(seat))
    return obs


class GinNetAgent:
    """Sampled policy from a trained GinNet. Owns its RNG (seeded)."""

    name = "gin-net"
    manual_phases = False

    def __init__(self, net: GinNet, feat_dim: int, device: torch.device, seed: int = 0) -> None:
        self.net = net.to(device).eval()
        self.feat_dim = feat_dim
        self.device = device
        self._rng = random.Random(seed)

    def begin_game(self, seat: int) -> None:
        pass

    def inform(self, state, player: int, action: int) -> None:
        pass

    def choose(self, env: HandEnv, seat: int) -> int:
        mask = env.legal_mask()
        # Zero-copy views: identical float32 values to torch.tensor([...]).
        obs = torch.from_numpy(
            np.asarray([gin_observation(env, seat, self.feat_dim)], dtype=np.float32)
        ).to(self.device)
        legal_mask = torch.as_tensor([mask[:N_LEARNED_ACTIONS]], dtype=torch.bool).to(self.device)
        with torch.no_grad():
            logits, _, _ = self.net(obs, legal_mask)
        probs = torch.softmax(logits.squeeze(0), dim=-1).tolist()
        legal = [a for a, allowed in enumerate(mask) if allowed]
        total = sum(probs[a] for a in legal)
        roll, acc = self._rng.random() * total, 0.0
        for action in legal:
            acc += probs[action]
            if acc >= roll:
                return action
        return legal[-1]


class BeliefAblatedAgent(GinNetAgent):
    """Same weights, learned-belief blocks zeroed (uniform-beliefs control).

    Only the known_opp/declined_opp blocks are zeroed; own hand, pile, and
    scalars stay (public observation, not belief). The paired score of this
    agent versus GinNetAgent is the decision-value ablation for beliefs.
    """

    name = "gin-net-no-belief"

    def choose(self, env: HandEnv, seat: int) -> int:
        from ginrl.nets.gin import N_CARD_BLOCKS

        deck = self.net.deck
        mask = env.legal_mask()
        obs = gin_observation(env, seat, self.feat_dim)
        # Blocks 9,10 are known_opp/declined_opp: the learned-belief state.
        # Own hand, pile, and scalars stay (public observation, not belief).
        start = 9 * deck
        obs[start : N_CARD_BLOCKS * deck] = [0.0] * (N_CARD_BLOCKS * deck - start)
        obs_t = torch.from_numpy(np.asarray([obs], dtype=np.float32)).to(self.device)
        legal_mask = torch.as_tensor([mask[:N_LEARNED_ACTIONS]], dtype=torch.bool).to(self.device)
        with torch.no_grad():
            logits, _, _ = self.net(obs_t, legal_mask)
        probs = torch.softmax(logits.squeeze(0), dim=-1).tolist()
        legal = [a for a, allowed in enumerate(mask) if allowed]
        total = sum(probs[a] for a in legal)
        roll, acc = self._rng.random() * total, 0.0
        for action in legal:
            acc += probs[action]
            if acc >= roll:
                return action
        return legal[-1]
