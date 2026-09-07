"""Acceptance gate for Phase 1 — game layer (IMPLEMENTATION_PLAN.md gate-p1).

Thresholds from configs/gates.yaml. Fixed seeds are printed first (the
correction loop reproduces with them). Never edit this script or a threshold
to make it pass. `--quick` runs every check at 1-5% scale for iteration.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pyspiel  # noqa: E402
import yaml  # noqa: E402

from ginrl.config import HandConfig, Seeds  # noqa: E402
from ginrl.env import melds, probe  # noqa: E402
from ginrl.env.game import N_LEARNED_ACTIONS, HandEnv  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def random_returns(rng: random.Random) -> tuple[float, float]:
    """One raw-engine random playout (fast path: no wrapper overhead)."""
    game = pyspiel.load_game("gin_rummy")
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outs, probs = zip(*state.chance_outcomes(), strict=True)
            state.apply_action(rng.choices(outs, list(probs))[0])
        else:
            state.apply_action(rng.choice(state.legal_actions()))
    return (float(state.returns()[0]), float(state.returns()[1]))


def gate_round_trip() -> None:
    game = pyspiel.load_game("gin_rummy")
    rng = random.Random(1)
    state = game.new_initial_state()
    while state.is_chance_node():
        outs, probs = zip(*state.chance_outcomes(), strict=True)
        state.apply_action(rng.choices(outs, list(probs))[0])
    named = sum(1 for a in range(game.num_distinct_actions()) if state.action_to_string(a))
    check("action round-trip (241 stringify)", named == 241, f"{named}/241")
    check("reduced action set is 0..55", N_LEARNED_ACTIONS == 56)


def gate_zero_sum(n_playouts: int) -> None:
    total = 0.0
    for i in range(n_playouts):
        r0, r1 = random_returns(random.Random(1000 + i))
        total += r0 + r1
    check(f"zero-sum over {n_playouts} playouts", total == 0.0, f"sum={total}")


def _check_terminal(env: HandEnv, returns: tuple[float, float], tag: str) -> tuple[bool, str]:
    """Agreement assertions for one finished hand. Returns (ok, branch-or-detail)."""
    terminal = env.state().to_dict()
    ours = melds.reproduce_returns(terminal)
    if abs(ours[0] - returns[0]) + abs(ours[1] - returns[1]) > 0.0:
        return (False, f"{tag}: returns {ours} != engine {returns}")
    if returns == (0.0, 0.0):
        return (True, "wall")
    knocker = terminal["knocked"].index(True)
    # Engine knocker deadwood must equal our optimal meld group deadwood:
    # the auto-policy always lays the best group.
    eng_dw = terminal["deadwood"][knocker]
    remaining = melds.parse_hand(terminal["hands"][knocker])
    laid = [melds.parse_hand(m) for m in terminal["layed_melds"][knocker]]
    full = remaining + [c for m in laid for c in m]
    ours_dw = melds.deadwood_of_melded(full, laid)
    optimal = melds.min_deadwood(full)
    if not (ours_dw == eng_dw == optimal):
        return (False, f"{tag}: ours={ours_dw} eng={eng_dw} opt={optimal}")
    if ours_dw == 0:
        return (True, "gin")
    defender = 1 - knocker
    def_dw = sum(melds.card_value(c) for c in melds.parse_hand(terminal["hands"][defender]))
    if def_dw < eng_dw:
        return (True, "undercut")
    return (True, "knock" if def_dw > eng_dw else "tie")


def _greedy_action(env: HandEnv, r) -> int:
    """Gate-only scaffolding: minimise own deadwood every decision.

    Draws the upcard iff it lowers post-discard deadwood, discards the
    deadwood-minimising card, knocks whenever offered. A precursor to the
    Phase 2 heuristic family, kept here so the oracle sees low-deadwood
    terminals (undercuts) that random play never reaches.
    """
    from ginrl.env.meld_policy import choose_knock_discard

    state = env.state()
    seat = state.current_player()
    info = state.to_dict()
    hand = melds.parse_hand(info["hands"][seat])
    up = info["upcard"]
    phase = info["phase"]
    mask = r.mask
    if up and phase in ("FirstUpcard", "Draw") and mask[52]:
        up_idx = melds.card_to_index(up)
        take = melds.min_deadwood(
            [h for h in hand + [up_idx] if h != choose_knock_discard(hand + [up_idx])]
        )
        if take < melds.min_deadwood(hand):
            return 52
    if phase == "Discard":
        if mask[55]:
            return 55
        options = [c for c in hand if mask[c]]
        assert options, "no legal discard at Discard phase"
        return min(options, key=lambda c: (melds.min_deadwood([h for h in hand if h != c]), c))
    for action in (54, 53):
        if action < len(mask) and mask[action]:
            return action
    raise AssertionError(f"greedy policy found no legal action at {phase}")


def gate_oracle(env: HandEnv, prng: random.Random, n_terminals: int, quick: bool) -> None:
    from collections import Counter

    branches: Counter[str] = Counter()
    for i in range(n_terminals):
        r = env.reset(seed=5000 + i)
        while not r.done:
            legal = [a for a, m in enumerate(r.mask) if m]
            r = env.step(prng.choice(legal))
        assert r.returns is not None
        ok, branch = _check_terminal(env, r.returns, f"hand {i}")
        if not ok:
            check("meld oracle agreement", False, branch)
            return
        branches[branch] += 1
    # Rare branches get a supplementary phase with identical assertions
    # rather than weakening the coverage demand. Random defenders carry ~40
    # deadwood, so P(undercut|knock) is ~0 even knock-heavy (0 in ~1000
    # knocks); greedy-meld play on both seats produces all branches
    # (measured: 1923 knock / 29 gin / 33 undercut / 15 wall in 2000 hands).
    if not quick and (branches["undercut"] == 0 or branches["gin"] == 0):
        for i in range(2000):
            r = env.reset(seed=60000 + i)
            while not r.done:
                r = env.step(_greedy_action(env, r))
            assert r.returns is not None
            ok, branch = _check_terminal(env, r.returns, f"greedy {i}")
            if not ok:
                check("meld oracle agreement", False, branch)
                return
            branches[branch] += 1
            if branches["undercut"] >= 5 and branches["gin"] >= 1:
                break
    check("meld oracle agreement", True, f"{n_terminals} terminals + knock-heavy top-up")
    print(f"  branches: {dict(branches)}")
    if not quick:
        check(
            "oracle saw all branches",
            all(branches[b] > 0 for b in ("wall", "knock", "gin", "undercut")),
        )
        print(f"  (ties seen as knock-0: {branches['tie']})")


def gate_mask(env: HandEnv, prng: random.Random, n_states: int) -> None:
    fuzzed = 0
    seed = 9000
    ok = True
    while fuzzed < n_states:
        r = env.reset(seed=seed)
        seed += 1
        while not r.done and fuzzed < n_states:
            mask_set = {a for a, m in enumerate(r.mask) if m}
            if set(env.state().legal_actions()) != mask_set or not mask_set:
                ok = False
                break
            fuzzed += 1
            r = env.step(prng.choice(sorted(mask_set)))
        if not ok:
            break
    check(f"mask soundness ({fuzzed} states)", ok)


def gate_hygiene(n_states: int, n_resamples: int) -> None:
    hrng = random.Random(2)
    henv = HandEnv(HandConfig(), Seeds(2))
    tested = 0
    seed = 7000
    leak = ""
    varying = 0
    while tested < n_states and not leak:
        r = henv.reset(seed=seed)
        seed += 1
        while not r.done and tested < n_states and not leak:
            st = henv.state().clone()
            seat = st.current_player()
            tracker = henv.trackers[seat]
            base = tracker.features(st.to_observation_struct(seat).to_dict(), None)
            varied = False
            first_opp = None
            for _ in range(n_resamples):
                rs = st.resample_from_infostate(seat, lambda: hrng.random())
                opp = tuple(sorted(melds.parse_hand(rs.to_dict()["hands"][1 - seat])))
                if first_opp is None:
                    first_opp = opp
                varied |= opp != first_opp
                obs = rs.to_observation_struct(seat).to_dict()
                if not np.array_equal(tracker.features(obs, None), base):
                    leak = f"state {tested}, seat {seat}"
                    break
            varying += varied
            tested += 1
            legal = [a for a, m in enumerate(r.mask) if m]
            r = henv.step(hrng.choice(legal))
    check(f"information hygiene ({tested}x{n_resamples})", not leak, leak)
    print(f"  resamples varied the hidden hand in {varying}/{tested} states")


def gate_necessity(n_games: int, margin: float, quick: bool) -> None:
    env = HandEnv(HandConfig(), Seeds(3))
    nrng = random.Random(3)
    raw_x, trk_x, y, gg = [], [], [], []
    for gi in range(n_games):
        r = env.reset(seed=3000 + gi)
        while not r.done:
            st = env.state()
            seat = st.current_player()
            opp_hand = st.to_dict()["hands"][1 - seat]
            if opp_hand and len(opp_hand) == 10 and "XX" not in opp_hand:
                label = np.zeros(52, dtype=np.float32)
                for c in melds.parse_hand(opp_hand):
                    label[c] = 1.0
                raw_x.append(np.array(st.observation_tensor(seat), dtype=np.float32))
                trk_x.append(np.array(env.features_for(seat), dtype=np.float32))
                y.append(label)
                gg.append(gi)
            legal = [a for a, m in enumerate(r.mask) if m]
            r = env.step(nrng.choice(legal))
    raw_x = np.stack(raw_x).astype(np.float32)
    trk_x = np.stack(trk_x).astype(np.float32)
    y = np.stack(y).astype(np.float32)
    gg = np.array(gg)
    order = np.random.RandomState(3).permutation(n_games)
    test_games = set(order[: max(1, n_games // 4)].tolist())
    te = np.array([i for i, g in enumerate(gg) if g in test_games])
    tr = np.array([i for i, g in enumerate(gg) if g not in test_games])
    raw_auc, _ = probe.mean_auc(probe.train_multilabel_probe(raw_x[tr], y[tr], raw_x[te]), y[te])
    trk_auc, _ = probe.mean_auc(probe.train_multilabel_probe(trk_x[tr], y[tr], trk_x[te]), y[te])
    detail = f"trk={trk_auc:.4f} raw={raw_auc:.4f} ({n_games} games)"
    if quick:
        # Informational only: 8 games cannot train either probe; the full
        # scale run is the verdict.
        print(f"[SKIP] history necessity — {detail}")
        return
    check("history necessity", trk_auc > raw_auc + margin, detail)


def gate_throughput(env: HandEnv, prng: random.Random, n_games: int, max_drop: float) -> None:
    t0 = time.perf_counter()
    decisions = 0
    for gi in range(n_games):
        r = env.reset(seed=8000 + gi)
        while not r.done:
            env.features_for(r.seat)
            decisions += 1
            legal = [a for a, m in enumerate(r.mask) if m]
            r = env.step(prng.choice(legal))
    elapsed = time.perf_counter() - t0
    rate = decisions / elapsed
    line = f"- feature_steps_per_sec: {rate:.0f} ({decisions} decisions, {elapsed:.1f}s)"
    md_path = ROOT / "docs" / "ENV_FACTS.md"
    marker = "<!-- phase-results-append-below -->"
    text = md_path.read_text()
    baseline = None
    section = f"{marker}\n## Phase 1 throughput\n{line}\n"
    if marker in text:
        found = re.search(r"feature_steps_per_sec: (\d+)", text.split(marker, 1)[1])
        baseline = float(found.group(1)) if found else None
        head, _ = text.split(marker, 1)
        md_path.write_text(head + section)
    else:
        md_path.write_text(text.rstrip("\n") + "\n\n" + section)
    print(f"  throughput: {line} (baseline: {baseline})")
    regress = baseline is not None and rate < baseline / max_drop
    check("throughput (no >2x regression)", not regress, f"{rate:.0f} steps/s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    gates = yaml.safe_load((ROOT / "configs" / "gates.yaml").read_text())
    expected = gates["gate-p1"]
    scale = 0.02 if args.quick else 1.0
    print(f"gate-p1 seed=1 quick={args.quick}")

    env = HandEnv(HandConfig(), Seeds(1))
    prng = random.Random(1)
    gate_round_trip()
    gate_zero_sum(max(100, int(expected["zero_sum_playouts"] * scale)))
    gate_oracle(env, prng, max(50, int(expected["meld_oracle_states"] * scale)), args.quick)
    gate_mask(env, prng, max(200, int(expected["mask_fuzz_states"] * scale)))
    gate_hygiene(max(10, int(expected["hygiene_states"] * scale)), 32 if not args.quick else 4)
    gate_necessity(max(8, int(400 * scale)), 0.02, args.quick)
    gate_throughput(env, prng, max(5, int(200 * scale)), expected["throughput_regression_max"])

    if FAILURES:
        print(f"gate-p1 FAIL ({len(FAILURES)} checks): {FAILURES}")
        return 1
    print("gate-p1 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
