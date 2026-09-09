"""Policy-level information hygiene (Phase 4).

The feature-level test guards the tracker; this guards the whole inference
graph including recurrent state. For decision states from real play, resample
the hidden information with resample_from_infostate (which provably varies
the opponent's hand — asserted so the test cannot pass vacuously) and require
the policy logits to be bit-identical across resamples. Any variation is a
leak of hidden state into the policy.
"""

from __future__ import annotations

import random
import sys

import torch

sys.path.insert(0, "src")

from ginrl.agents.learned import gin_observation
from ginrl.config import REDUCED_HAND_CONFIG, Seeds
from ginrl.env.features import EVENT_WINDOW, feature_dim
from ginrl.env.game import LEARNED_PHASES, HandEnv
from ginrl.env.melds import parse_hand
from ginrl.nets.gin import RAW_DIM, GinNet

FIXTURES = 24
RESAMPLES = 8


def _logits(net: GinNet, obs: list[float], mask: list[bool]) -> list[float]:
    with torch.no_grad():
        out, _, _ = net(
            torch.tensor([obs], dtype=torch.float32),
            torch.tensor([mask], dtype=torch.bool),
        )
    return out.squeeze(0).tolist()


def test_policy_logits_bit_identical_across_resamples() -> None:
    torch.manual_seed(0)
    deck = REDUCED_HAND_CONFIG.deck_size
    feat_dim = feature_dim(deck)
    net = GinNet("mlp", feat_dim=feat_dim, deck=deck).eval()
    env = HandEnv(config=REDUCED_HAND_CONFIG, seeds=Seeds(41))
    rng = random.Random(41)
    checked = 0
    for seed in range(64):
        if checked >= FIXTURES:
            break
        r = env.reset(seed=seed)
        while not r.done and checked < FIXTURES:
            state = env.state().clone()
            seat = state.current_player()
            mask = [bool(m) for m in env.legal_mask()]
            base = _logits(net, gin_observation(env, seat, feat_dim), mask)
            opp_hands = set()
            for _ in range(RESAMPLES):
                rs = state.resample_from_infostate(seat, lambda: rng.random())
                d = rs.to_dict()
                opp_hands.add(tuple(sorted(parse_hand(d["hands"][1 - seat]))))
                obs_d = rs.to_observation_struct(seat).to_dict()
                learned = str(d["phase"]) in LEARNED_PHASES
                fmask = mask if (learned and seat == rs.current_player()) else None
                feat = env.trackers[seat].features(obs_d, fmask).tolist()
                events = [float(t) for t in env.trackers[seat].event_window()]
                raw = [float(x) for x in rs.observation_tensor(seat)]
                assert len(raw) == RAW_DIM
                assert _logits(net, feat + events + raw, mask) == base, (
                    "policy leak: logits vary across resamples of one infostate"
                )
            assert len(opp_hands) > 1, "resample never varied the hidden hand"
            checked += 1
            legal = [a for a, m in enumerate(r.mask) if m]
            r = env.step(rng.choice(legal))
    assert checked == FIXTURES
    assert EVENT_WINDOW == 32
