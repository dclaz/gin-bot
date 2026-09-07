"""History necessity: BeliefTracker features must beat raw observations.

A held-out multilabel probe predicts "is card c in the opponent's hand"
for all 52 cards. If the raw-observation probe does as well as the
tracker probe, the tracker is buggy or redundant. Split is by game.
"""

from __future__ import annotations

import random

import numpy as np

from ginrl.config import Seeds
from ginrl.env import probe
from ginrl.env.game import HandEnv
from ginrl.env.melds import parse_hand

MARGIN = 0.015


def collect(n_games: int, base_seed: int):
    env = HandEnv(seeds=Seeds(41))
    rng = random.Random(41)
    raw_x, trk_x, y, games = [], [], [], []
    for game in range(n_games):
        r = env.reset(seed=base_seed + game)
        while not r.done:
            st = env.state()
            seat = st.current_player()
            opp_hand = st.to_dict()["hands"][1 - seat]
            if opp_hand and len(opp_hand) == 10 and "XX" not in opp_hand:
                label = np.zeros(52, dtype=np.float32)
                for c in parse_hand(opp_hand):
                    label[c] = 1.0
                raw_x.append(np.array(st.observation_tensor(seat), dtype=np.float32))
                trk_x.append(np.array(env.features_for(seat), dtype=np.float32))
                y.append(label)
                games.append(game)
            legal = [a for a, m in enumerate(r.mask) if m]
            r = env.step(rng.choice(legal))
    split = np.random.RandomState(41).permutation(n_games)
    test_games = set(split[: n_games // 4].tolist())
    games = np.array(games)
    te = np.array([i for i, g in enumerate(games) if g in test_games])
    tr = np.array([i for i, g in enumerate(games) if g not in test_games])
    return (
        np.stack(raw_x).astype(np.float32)[tr],
        np.stack(trk_x).astype(np.float32)[tr],
        np.stack(y).astype(np.float32)[tr],
        np.stack(raw_x).astype(np.float32)[te],
        np.stack(trk_x).astype(np.float32)[te],
        np.stack(y).astype(np.float32)[te],
    )


def test_tracker_probe_beats_raw_observation_probe() -> None:
    raw_tr, trk_tr, y_tr, raw_te, trk_te, y_te = collect(60, 4100)
    raw_auc, _ = probe.mean_auc(probe.train_multilabel_probe(raw_tr, y_tr, raw_te), y_te)
    trk_auc, _ = probe.mean_auc(probe.train_multilabel_probe(trk_tr, y_tr, trk_te), y_te)
    assert trk_auc > raw_auc + MARGIN, (
        f"tracker probe must beat raw-observation probe: {trk_auc:.4f} vs {raw_auc:.4f}"
    )
