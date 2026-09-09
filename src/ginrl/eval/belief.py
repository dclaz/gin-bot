"""Belief metrics on held-out states (Phase 4).

The aux head predicts the joint opponent hand (multilabel over the deck).
Two rules keep the metric honest:

1. Undetermined-only scoring: cards the tracker has already determined
   (taken set, own hand, visible pile) are excluded. Echoing tracker
   inputs scores nothing — BeliefTracker's own knowledge is a constant
   predictor (AUC 0.5) on this set by construction.
2. Held-out states: collected from fixed-policy games on seeds the
   learner never trained on (self-play never sees these deals).

Decision value pairs a learned-belief user against a uniform-belief
control over duplicate deals (see agents.learned).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from ginrl.agents.baselines import HeuristicAgent
from ginrl.config import HandConfig, Seeds
from ginrl.env.features import EVENT_WINDOW, feature_dim
from ginrl.env.game import HandEnv
from ginrl.env.melds import CardLayout
from ginrl.nets.gin import RAW_DIM, GinNet


@dataclass
class BeliefState:
    obs: tuple[float, ...]
    mask: tuple[bool, ...]
    opp: tuple[float, ...]  # joint true hand, multi-hot (label namespace)
    undetermined: tuple[bool, ...]  # eval set: tracker-undetermined cards


def undetermined_mask(env: HandEnv, seat: int) -> list[bool]:
    """Cards the tracker has not determined, in layout indices.

    Excluded: own hand, visible pile, the upcard (cannot be opp-held), and
    taken cards (determined present). Declined cards stay: declining then
    does not imply not-holding now.
    """
    layout = CardLayout(env.config.num_ranks, env.config.num_suits)
    full = env.state().to_dict()
    excluded = set(layout.parse_hand(full["hands"][seat]))
    pile = full["discard_pile"] if isinstance(full["discard_pile"], list) else []
    excluded.update(layout.parse_hand(pile))
    if isinstance(full["upcard"], str):
        excluded.add(layout.card_to_index(full["upcard"]))
    excluded.update(env.trackers[seat].known_cards())
    return [c not in excluded for c in range(layout.deck_size)]


def collect_belief_states(
    config: HandConfig,
    n_states: int,
    seed: int,
    max_games: int = 400,
) -> list[BeliefState]:
    """Held-out decision states from heuristic-vs-heuristic games."""
    deck = config.deck_size
    feat_dim = feature_dim(deck)
    states: list[BeliefState] = []
    game_seed = seed
    agent_a, agent_b = HeuristicAgent(), HeuristicAgent()
    agents = (agent_a, agent_b)
    while len(states) < n_states and game_seed < seed + max_games:
        env = HandEnv(config, Seeds(game_seed))

        def listener(acting_seat: int, state, player: int, action: int, _agents=agents) -> None:
            for agent, side in zip(_agents, (0, 1), strict=True):
                if side != acting_seat:
                    agent.inform(state, player, action)

        env.listener = listener
        result = env.reset(seed=game_seed)
        agent_a.begin_game(0)
        agent_b.begin_game(1)
        while not result.done and len(states) < n_states:
            seat = env.state().current_player()
            obs = [0.0] * (feat_dim + EVENT_WINDOW + RAW_DIM)
            obs[:feat_dim] = env.features_for(seat)
            obs[feat_dim : feat_dim + EVENT_WINDOW] = [
                float(t) for t in env.trackers[seat].event_window()
            ]
            obs[feat_dim + EVENT_WINDOW :] = list(env.tensor_for(seat))
            opp = [0.0] * deck
            for card in env.hidden_opp_hand(seat):
                opp[card] = 1.0
            states.append(
                BeliefState(
                    obs=tuple(obs),
                    mask=tuple(env.legal_mask()),
                    opp=tuple(opp),
                    undetermined=tuple(undetermined_mask(env, seat)),
                )
            )
            result = env.step(agents[seat].choose(env, seat))
        game_seed += 1
    return states


def auc_score(scores: list[float], labels: list[float]) -> float:
    """Pooled Mann-Whitney AUC of scores against binary labels."""
    pos = sorted((s for s, y in zip(scores, labels, strict=True) if y > 0.5), reverse=True)
    neg = sorted((s for s, y in zip(scores, labels, strict=True) if y <= 0.5), reverse=True)
    if not pos or not neg:
        return 0.5
    # Rank all scores ascending (rank 1 = lowest); ties get the average rank.
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(ranks[i] for i, y in enumerate(labels) if y > 0.5)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def belief_auc(net: GinNet, states: list[BeliefState], device: torch.device) -> tuple[float, float]:
    """(aux AUC, tracker-baseline AUC) over undetermined cards, pooled."""
    net.eval()
    scores: list[float] = []
    taken_scores: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for start in range(0, len(states), 256):
            chunk = states[start : start + 256]
            obs = torch.tensor([s.obs for s in chunk], dtype=torch.float32).to(device)
            mask = torch.tensor([s.mask for s in chunk], dtype=torch.bool).to(device)
            _, _, aux = net(obs, mask)
            post = torch.sigmoid(aux).cpu().tolist()
            for s, probs in zip(chunk, post, strict=True):
                for c, keep in enumerate(s.undetermined):
                    if keep:
                        scores.append(probs[c])
                        labels.append(s.opp[c])
                        taken_scores.append(0.0)  # taken cards are excluded by design
    return auc_score(scores, labels), auc_score(taken_scores, labels)


def held_out_returns(config: HandConfig, seed: int, n_games: int) -> list[tuple[float, float]]:
    """Heuristic-vs-heuristic game returns on fresh seeds (sanity pool)."""
    rng = random.Random(seed)
    out = []
    for game in range(n_games):
        env = HandEnv(config, Seeds(seed + game))
        result = env.reset(seed=seed + game)
        while not result.done:
            legal = [a for a, m in enumerate(result.mask) if m]
            result = env.step(rng.choice(legal))
        assert result.returns is not None
        out.append(result.returns)
    return out
