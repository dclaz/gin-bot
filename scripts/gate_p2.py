"""Phase 2 gate: opponents and evaluation harness.

Reads thresholds from configs/gates.yaml (gate-p2). Live checks only; the
synthetic ratings checks (recovery, coverage, cycles, undefeated agent) run
under `make test`, which this gate depends on via `gate-p2: lint test`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from ginrl.agents.baselines import (  # noqa: E402
    HeuristicAgent,
    HeuristicParams,
    RandomAgent,
)
from ginrl.agents.spiel_bots import ISMCTSAgent, SimpleGinRummyAgent  # noqa: E402
from ginrl.config import HandConfig, Seeds  # noqa: E402
from ginrl.env.game import HandEnv  # noqa: E402
from ginrl.eval.arena import Arena, replay  # noqa: E402
from ginrl.eval.native import play_native  # noqa: E402
from ginrl.eval.ratings import fit_ratings  # noqa: E402
from ginrl.eval.record import deals_from_summary  # noqa: E402
from ginrl.eval.styles import TABLE_COLUMNS, profile, table_row  # noqa: E402
from ginrl.telemetry.recorder import Recorder, RecorderConfig  # noqa: E402

GATES = yaml.safe_load((ROOT / "configs" / "gates.yaml").read_text())["gate-p2"]
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def make_env() -> HandEnv:
    return HandEnv(HandConfig(), Seeds())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    div = 20 if args.quick else 1

    def count(key: str, floor: int = 2) -> int:
        return max(floor, int(GATES[key] / div))

    t0 = time.time()
    arena = Arena()
    if args.quick:
        print("gate-p2 --quick: sample counts divided by 20", flush=True)
    simple = SimpleGinRummyAgent().name
    random = RandomAgent().name

    # 1. Random vs SimpleBot: CI excludes zero; magnitude is the baseline.
    s = arena.duplicate_summary(
        SimpleGinRummyAgent(), RandomAgent(), seed=1, n_deals=count("random_vs_simple_deals", 20)
    )
    ppb = s.points_per_hand()
    check(
        "random_vs_simple",
        ppb.lo > 0,
        f"simple {ppb.mean:+.1f} [{ppb.lo:+.1f}, {ppb.hi:+.1f}] pts/hand over {s.n_deals} deals",
    )

    # 2. Default heuristic beats random decisively.
    h = arena.duplicate_summary(
        HeuristicAgent(),
        RandomAgent(),
        seed=2,
        n_deals=count("heuristic_vs_random_deals", 20),
    )
    hppb = h.points_per_hand()
    check(
        "heuristic_vs_random",
        hppb.lo > 0,
        f"heuristic {hppb.mean:+.1f} [{hppb.lo:+.1f}, {hppb.hi:+.1f}] pts/hand",
    )

    # 3. Duplicate variance check on a close matchup: pairing must cut the
    # score-estimator variance materially (seat luck cancels; on blowouts the
    # legs correlate positively through the deal size and nothing cancels).
    # The leg-variance floor guards a vacuous pass: two broken mirrors that
    # both wall every game have ratio 0.00 with nothing behind it.
    v = arena.duplicate_summary(
        HeuristicAgent(HeuristicParams(knock_threshold=7)),
        HeuristicAgent(HeuristicParams(knock_threshold=9)),
        seed=3,
        n_deals=count("duplicate_variance_deals", 40),
    )
    legs = np.array(v.legs_for_a())
    per_deal = np.array([(p.leg1.returns[0] + p.leg2.returns[1]) / 2 for p in v.pairs])
    ratio = 2 * float(per_deal.var()) / float(legs.var())
    check(
        "duplicate_variance",
        ratio < GATES["duplicate_variance_ratio_max"] and float(legs.var()) > 5,
        f"paired/unpaired estimator variance ratio {ratio:.2f} "
        f"(leg var {float(legs.var()):.1f}; guards a degenerate all-tie pass)",
    )

    # 4. Null calibration: deterministic mirror gives exact zeros; the same
    # run yields the self-play reference numbers.
    n0 = arena.duplicate_summary(
        SimpleGinRummyAgent(),
        SimpleGinRummyAgent(),
        seed=4,
        n_deals=count("null_calibration_duplicate_deals", 200),
    )
    scores = np.array(n0.paired_scores())
    null_ppb = n0.points_per_hand()
    check(
        "null_calibration",
        float(np.abs(scores).max()) == 0.0 and null_ppb.lo <= 0 <= null_ppb.hi,
        f"max|paired|={float(np.abs(scores).max())} CI=[{null_ppb.lo}, {null_ppb.hi}] over {n0.n_deals} deals",
    )
    all_legs = [leg for p in n0.pairs for leg in (p.leg1, p.leg2)]
    totals = np.array([leg.total_player_actions for leg in all_legs])
    mags = np.array([abs(leg.returns[0]) for leg in all_legs])
    walls = sum(1 for leg in all_legs if leg.wall)
    ties = sum(1 for leg in all_legs if leg.returns == (0.0, 0.0) and not leg.wall)
    lo_a, hi_a = GATES["total_actions_band"]
    lo_m, hi_m = GATES["points_per_hand_magnitude_band"]
    check(
        "reference_actions",
        lo_a <= float(totals.mean()) <= hi_a,
        f"{float(totals.mean()):.1f} total actions/game (band [{lo_a}, {hi_a}])",
    )
    check(
        "reference_magnitude",
        lo_m <= float(mags.mean()) <= hi_m,
        f"{float(mags.mean()):.1f} pts/hand magnitude "
        f"(probe-era ref {GATES['ref_points_per_hand_magnitude']}, band [{lo_m}, {hi_m}])",
    )
    check(
        "reference_walls",
        walls / len(all_legs) < GATES["wall_fraction_max"],
        f"wall {walls}/{len(all_legs)}, scoreless tied knocks {ties}/{len(all_legs)}",
    )

    # 5. Stochastic null: random self-play has no seat advantage to find, and
    # the first-player edge sits within its CI of zero (the mirror check).
    # Random play almost never knocks, so ~99% of these legs wall 0-0: the
    # paired mean (tight, symmetric zeros) carries this check, while the edge
    # rests on the few dozen decisive legs and its CI is honestly wide.
    r = arena.duplicate_summary(
        RandomAgent(), RandomAgent(), seed=5, n_deals=count("random_null_deals", 60)
    )
    rppb = r.points_per_hand()
    res = fit_ratings(deals_from_summary(r), random)
    elo, ehi = res.edge_elo_ci()
    check(
        "random_null",
        rppb.lo <= 0 <= rppb.hi and elo <= 0 <= ehi,
        f"paired mean {rppb.mean:+.2f} [{rppb.lo:+.2f}, {rppb.hi:+.2f}], "
        f"edge {res.edge_elo:+.1f} [{elo:+.1f}, {ehi:+.1f}] Elo",
    )

    # 6. Parity: the harness drives engine bots exactly like the native loop.
    params = HandConfig().game_params()
    n_parity = count("parity_seeds", 10)
    bad = 0
    for seed in range(1000, 1000 + n_parity):
        g = arena.play_game(SimpleGinRummyAgent(), SimpleGinRummyAgent(), seed)
        env = make_env()
        replay(g, env)
        ret_n, act_n = play_native(params, list(env.chance_record))
        mine = [(x.seat, x.action) for x in g.actions]
        if g.returns != ret_n or mine != act_n:
            bad += 1
    check(
        "parity",
        bad == 0,
        f"{n_parity - bad}/{n_parity} identical seeds",
    )

    # 7. Behavioural profiler table over every baseline.
    members = [
        (RandomAgent(), count("profiler_hands_fast", 8)),
        (SimpleGinRummyAgent(), count("profiler_hands_fast", 8)),
        (HeuristicAgent(), count("profiler_hands_fast", 8)),
        (
            ISMCTSAgent(simulations=GATES["ismcts_smoke_simulations"], seed=0),
            count("profiler_hands_ismcts", 2),
        ),
    ]
    print("profile: " + " | ".join(TABLE_COLUMNS), flush=True)
    for agent, hands in members:
        games = []
        for i in range(hands):
            games.append(arena.play_game(SimpleGinRummyAgent(), agent, 7000 + i))
            games.append(arena.play_game(agent, SimpleGinRummyAgent(), 7000 + i))
        p = profile(agent.name, games, make_env)
        print("profile: " + " | ".join(table_row(p)), flush=True)
    check("profiler", True, f"table printed for {len(members)} baselines")

    # 8. Telemetry smoke: header, scalars, histogram, table through the fan-out.
    with tempfile.TemporaryDirectory() as tmp:
        with Recorder(
            RecorderConfig(
                run_dir=Path(tmp) / "run",
                run_name="gate-p2-smoke",
                dashboard_enabled=False,
            )
        ) as rec:
            rec.log_scalar(0, "loss/total", 1.0)
            rec.log_histogram(0, "style/deadwood_at_knock", [1.0, 7.0, 7.0])
            rec.log_table(0, "ratings/ladder", ["agent"], [[simple]])
        kinds = [
            json.loads(line)["type"]
            for line in (Path(tmp) / "run" / "metrics.jsonl").read_text().splitlines()
        ]
    check(
        "telemetry",
        kinds == ["header", "scalar", "histogram", "table"],
        f"JSONL record kinds {kinds} (sink-kill isolation is unit-tested)",
    )

    # 8b. Arena wiring: every later phase logs through the Recorder because
    # the arena carries one, not because callers remember to.
    with tempfile.TemporaryDirectory() as tmp:
        rec = Recorder(
            RecorderConfig(
                run_dir=Path(tmp) / "wired",
                run_name="gate-p2-wired",
                dashboard_enabled=False,
            )
        )
        wired = Arena(recorder=rec)
        wired.duplicate_summary(HeuristicAgent(), RandomAgent(), seed=8, n_deals=10)
        rec.close()
        wired_rows = [
            json.loads(line)
            for line in (Path(tmp) / "wired" / "metrics.jsonl").read_text().splitlines()
        ]
    tables = [r for r in wired_rows if r.get("metric") == "ratings/head_to_head"]
    check(
        "arena_wiring",
        len(tables) == 1 and wired.legs_played == 20,
        f"head_to_head rows {[t['rows'] for t in tables]}",
    )

    print(f"gate-p2 finished in {time.time() - t0:.0f}s", flush=True)
    if FAILURES:
        print(f"FAILURES: {FAILURES}", flush=True)
        return 1
    print("gate-p2: all checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
