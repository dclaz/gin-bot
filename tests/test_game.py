"""Hand/match wrapper: termination, zero-sum, masks, duplicates, matches."""

from __future__ import annotations

import random

from ginrl.config import HandConfig, MatchConfig, Seeds
from ginrl.env.game import HandEnv
from ginrl.env.match import MatchEnv


def _play_hand(env: HandEnv, seed: int, rng: random.Random, knock: bool = False):
    r = env.reset(seed=seed)
    while not r.done:
        legal = [a for a, m in enumerate(r.mask) if m]
        assert legal, "mask all-zero at a decision node"
        action = 55 if knock and r.mask[55] else rng.choice(legal)
        r = env.step(action)
    return r


def test_random_hands_finish_zero_sum() -> None:
    env = HandEnv(HandConfig(), Seeds(21))
    rng = random.Random(21)
    for seed in range(30):
        r = _play_hand(env, seed, rng)
        assert r.returns is not None
        assert r.returns[0] + r.returns[1] == 0.0


def test_mask_never_offers_engine_only_actions() -> None:
    env = HandEnv(HandConfig(), Seeds(22))
    rng = random.Random(22)
    for seed in range(30, 45):
        r = env.reset(seed=seed)
        while not r.done:
            state = env.state()
            for action in state.legal_actions():
                assert action < 56, f"engine-only action {action} at learned phase"
            legal = [a for a, m in enumerate(r.mask) if m]
            r = env.step(rng.choice(legal))


def _play_to_end(r, rng: random.Random, env: HandEnv):
    while not r.done:
        legal = [a for a, m in enumerate(r.mask) if m]
        r = env.step(rng.choice(legal))
    assert r.returns is not None
    return r.returns


def test_duplicate_replay_is_identical() -> None:
    # Same seed, same policy, two env instances -> identical chance stream
    # (deal AND stock draws) and identical outcome.
    env_a = HandEnv(HandConfig(), Seeds(23))
    env_b = HandEnv(HandConfig(), Seeds(23))
    returns_a = _play_to_end(env_a.reset(seed=5), random.Random(23), env_a)
    returns_b = _play_to_end(env_b.reset(seed=5), random.Random(23), env_b)
    assert env_a.chance_record, "expected a non-empty chance record"
    assert env_a.chance_record == env_b.chance_record
    assert returns_a == returns_b


def test_stock_draws_follow_the_deal_prefix() -> None:
    # Same seed, divergent play: the deal is identical and the stock draws
    # form one shared order (the shorter draw sequence prefixes the longer),
    # so duplicates stay duplicate past the first divergent decision.
    records = []
    for policy_seed in (101, 102):
        env = HandEnv(HandConfig(), Seeds(23))
        r = env.reset(seed=9)
        _play_to_end(r, random.Random(policy_seed), env)
        records.append(env.chance_record)
    deal_a, stock_a = records[0][:21], records[0][21:]
    deal_b, stock_b = records[1][:21], records[1][21:]
    assert deal_a == deal_b
    short, long = sorted((stock_a, stock_b), key=len)
    assert short and long[: len(short)] == short


def test_knock_policy_scores_some_hands() -> None:
    env = HandEnv(HandConfig(), Seeds(24))
    rng = random.Random(24)
    scored = sum(
        1
        for seed in range(60, 90)
        if (_play_hand(env, seed, rng, knock=True).returns or (0.0, 0.0)) != (0.0, 0.0)
    )
    assert scored > 0, "knock-when-offered never scored in 30 hands"


def test_match_completes_with_sane_scores() -> None:
    env = MatchEnv(MatchConfig(), Seeds(25))
    rng = random.Random(25)
    r = env.reset(seed=1)
    hands = 0
    while not r.done:
        legal = [a for a, m in enumerate(r.hand.mask) if m]
        r = env.step(rng.choice(legal))
        if r.hand_returns is not None:
            hands += 1
            assert r.hand_returns[0] + r.hand_returns[1] == 0.0
    assert hands > 0
    assert r.winner in (0, 1)
    assert min(r.scores) >= 0.0, "match scores must never go negative"
    assert max(r.scores) >= env.config.target_score


def test_duplicate_matches_replay_identical_hands() -> None:
    """Same seed sequence both legs: identical per-hand returns with same policies."""
    first, second = [], []
    for out, base in ((first, 7), (second, 7)):
        env = MatchEnv(MatchConfig(), Seeds(26))
        rng = random.Random(26)
        r = env.reset(seed=base)
        while not r.done:
            legal = [a for a, m in enumerate(r.hand.mask) if m]
            r = env.step(rng.choice(legal))
            if r.hand_returns is not None:
                out.append(r.hand_returns)
    assert first and first == second
