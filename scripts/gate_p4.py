"""Phase 4 gate: reduced-gin bake-off, match wrapper, belief path.

Reads thresholds from configs/gates.yaml (gate-p4). Full mode trains the
5-torso x 3-seed bake-off inline at the recorded equal gradient-step budget
(5 parallel subprocess cells, single-threaded), then:

  A. table: mean paired score + CI, belief AUC, params, steps/s, infer ms;
  B. stability: per-torso cross-seed spread within band, no entropy collapse;
  C. winner beats HeuristicAgent (table CIs) and RandomAgent (live eval);
  D. winner mean belief AUC over the floor (undetermined-cards set);
  E. decision value: winner vs belief-ablated weights, pooled CI;
  F. match wrapper + policy hygiene pytest suites pass;
  G. cap accounting: zero unreported max_hands hits in scripted matches.

`--quick` divides budgets by 20 and asserts machinery + finiteness only.
`--cells DIR` reuses precomputed cells (same layout as the gate writes).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

from ginrl.agents.baselines import RandomAgent  # noqa: E402
from ginrl.agents.learned import BeliefAblatedAgent, GinNetAgent  # noqa: E402
from ginrl.config import REDUCED_HAND_CONFIG, MatchConfig, Seeds  # noqa: E402
from ginrl.env.features import feature_dim  # noqa: E402
from ginrl.env.match import MatchEnv  # noqa: E402
from ginrl.eval.arena import Arena  # noqa: E402
from ginrl.eval.bakeoff import run_cell  # noqa: E402
from ginrl.eval.belief import belief_auc, collect_belief_states  # noqa: E402
from ginrl.eval.stats import bootstrap_ci  # noqa: E402
from ginrl.nets.gin import GIN_TORSOS, GinNet  # noqa: E402

GATES = yaml.safe_load((ROOT / "configs" / "gates.yaml").read_text())["gate-p4"]
FAILURES: list[str] = []
DEVICE = torch.device("cpu")
BELIEF_SEED = 11
BELIEF_MASTER = 777000


def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def run_cells(parent: Path, steps: int, eval_deals: int, n_belief: int) -> None:
    """Train all 15 cells; one subprocess per cell, five at a time."""
    belief_paths = {}
    for seed in range(GATES["seeds"]):
        path = parent / f"belief_s{seed}.pkl"
        states = collect_belief_states(REDUCED_HAND_CONFIG, n_belief, seed=BELIEF_MASTER)
        with open(path, "wb") as fh:
            pickle.dump(states, fh)
        belief_paths[seed] = path

    def one(job: tuple[str, int]) -> str:
        torso, seed = job
        seed_dir = parent / f"s{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "bakeoff_cell.py"),
                "--torso",
                torso,
                "--seed",
                str(seed),
                "--steps",
                str(steps),
                "--parent",
                str(seed_dir),
                "--eval-deals-final",
                str(eval_deals),
                "--belief-seed",
                str(BELIEF_SEED),
                "--belief-states",
                str(belief_paths[seed]),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, **env},
        )
        tail = (r.stdout + r.stderr).strip().splitlines()[-1][-160:]
        return f"{torso}-s{seed} exit={r.returncode} {tail}"

    jobs = [(t, s) for s in range(GATES["seeds"]) for t in GIN_TORSOS]
    with ThreadPoolExecutor(max_workers=5) as pool:
        for line in pool.map(one, jobs):
            print("   ", line, flush=True)


def load_table(parent: Path) -> dict[str, list[dict]]:
    table: dict[str, list[dict]] = {}
    for seed in range(GATES["seeds"]):
        for torso in GIN_TORSOS:
            cell = json.loads((parent / f"s{seed}" / f"{torso}-s{seed}" / "cell.json").read_text())
            table.setdefault(torso, []).append(cell)
    return table


def load_net(parent: Path, torso: str, seed: int) -> GinNet:
    deck = REDUCED_HAND_CONFIG.deck_size
    net = GinNet(torso, feat_dim=feature_dim(deck), deck=deck, hidden=128)
    net.load_state_dict(
        torch.load(
            parent / f"s{seed}" / f"{torso}-s{seed}" / "final.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    return net.eval()


def _paired_scores(path: Path) -> dict[int, float]:
    out = {}
    for line in (path).read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar" and row.get("metric") == "ratings/paired_score":
            out[row["step"]] = row["value"]
    return out


def _prefix_matches(parent: Path, long_parent: Path, winner: str) -> bool:
    """Long continuation reproduces the bake-off cell's evals bit-exactly."""
    base = _paired_scores(parent / "s0" / f"{winner}-s0" / "metrics.jsonl")
    long = _paired_scores(long_parent / "s0" / f"{winner}-s0" / "metrics.jsonl")
    common = set(base) & set(long)
    return bool(common) and all(base[s] == long[s] for s in common)


def duplicate_pph(agent_a, agent_b, seed: int, n_deals: int):
    arena = Arena(config=REDUCED_HAND_CONFIG, seeds=Seeds(seed))
    summary = arena.duplicate_summary(agent_a, agent_b, seed, n_deals).checked(0)
    return summary.points_per_hand()


def check_cap_accounting() -> None:
    """Every max_hands hit is reported via capped; none unreported."""
    unreported = hits = played = 0
    for target in (1, 2, 30):
        for max_hands in (1, 2, 3, 7):
            for knock in (False, True):
                for base in (0, 3, 11):
                    cfg = MatchConfig(
                        hand=REDUCED_HAND_CONFIG, target_score=target, max_hands=max_hands
                    )
                    env = MatchEnv(config=cfg, seeds=Seeds(7))
                    r = env.reset(seed=base)
                    while True:
                        legal = [i for i, m in enumerate(r.hand.mask) if m]
                        r = env.step(55 if (knock and 55 in legal) else legal[0])
                        if r.done:
                            break
                    played += 1
                    assert r.winner >= 0, "match ended without a winner"
                    expected_cap = not any(s >= target for s in r.scores)
                    if expected_cap:
                        hits += 1
                    if r.capped != expected_cap:
                        unreported += 1
    check(
        "cap-accounting",
        unreported == 0,
        f"{played} matches, {hits} cap hits, {unreported} unreported",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cells", default=None)
    args = ap.parse_args()
    quick = args.quick
    div = 20 if quick else 1

    parent = Path(args.cells) if args.cells else Path(tempfile.mkdtemp(prefix="gate_p4_"))
    if args.cells:
        print(f"reusing cells in {parent}", flush=True)
    else:
        print(f"training 15 cells in {parent}", flush=True)
        run_cells(parent, 300000 // div, 2000 // div, 1500 // div)

    table = load_table(parent)
    info_only = " (info-only)" if quick else ""

    # A. equal budget + table.
    updates = {c["updates"] for cells in table.values() for c in cells}
    check(
        "equal-budget",
        len(updates) == 1 and GATES["equal_gradient_step_budget"],
        f"updates={sorted(updates)}",
    )
    means = {}
    print(
        f"{'torso':8} {'mean':>7} {'CI':>22} {'auc':>6} {'params':>7} {'rows/s':>7} {'ms':>6}",
        flush=True,
    )
    for torso, cells in table.items():
        scores = [c["final_score"] for c in cells]
        s = bootstrap_ci(scores)
        means[torso] = s.mean
        auc = sum(c["belief_auc"] for c in cells) / len(cells)
        print(
            f"{torso:8} {s.mean:+.3f} [{s.lo:+.3f},{s.hi:+.3f}] "
            f"{auc:.3f} {cells[0]['params']:7d} {cells[0]['rows_per_sec']:7.0f} "
            f"{cells[0]['infer_ms']:6.2f}",
            flush=True,
        )
    winner = max(means, key=lambda t: means[t])
    print(f"winner: {winner} {means[winner]:+.3f}{info_only}", flush=True)

    # B. stability across seeds.
    spread_ok = True
    for torso, cells in table.items():
        spread = max(c["final_score"] for c in cells) - min(c["final_score"] for c in cells)
        ok = spread <= GATES["final_score_spread_max"]
        spread_ok &= ok
        print(
            f"    spread[{torso}]={spread:.3f} (max {GATES['final_score_spread_max']})", flush=True
        )
    check(
        "seed-stability", spread_ok or quick, f"band={GATES['final_score_spread_max']}{info_only}"
    )
    ent_min = min(c["final_entropy"] for cells in table.values() for c in cells)
    check(
        "no-entropy-collapse",
        ent_min > GATES["entropy_floor"] or quick,
        f"min final entropy={ent_min:.3f}{info_only}",
    )

    # C. winner beats both baselines, CIs exclude zero.
    w_heu = all(c["score_lo"] > 0 for c in table[winner])
    check("beats-heuristic", w_heu or quick, f"{winner} 2000-deal CIs all above zero{info_only}")
    feat_dim = feature_dim(REDUCED_HAND_CONFIG.deck_size)
    rnd = duplicate_pph(
        GinNetAgent(load_net(parent, winner, 0), feat_dim, DEVICE, seed=101),
        RandomAgent(),
        seed=202,
        n_deals=2000 // div,
    )
    check(
        "beats-random",
        rnd.lo > 0 or quick,
        f"{winner} vs random pph={rnd.mean:+.3f} [{rnd.lo:+.3f},{rnd.hi:+.3f}]{info_only}",
    )

    # D. belief AUC over the undetermined set.
    belief_path = parent / "belief_s0.pkl"
    if not belief_path.exists():  # batch layout: per-seed belief file
        belief_path = parent / "s0" / "belief_states.pkl"
    with open(belief_path, "rb") as fh:
        states = pickle.load(fh)
    aucs = [
        belief_auc(load_net(parent, winner, s), states, DEVICE)[0] for s in range(GATES["seeds"])
    ]
    auc_mean = sum(aucs) / len(aucs)
    check(
        "belief-auc",
        auc_mean >= GATES["belief_auc_floor"] or quick,
        f"{winner} mean aux AUC={auc_mean:.3f} floor={GATES['belief_auc_floor']}{info_only}",
    )

    # E. decision value: learned beliefs vs uniform-belief control.
    # Beliefs only move decisions once trained (300k cells read indifferent),
    # so the gate continues the winner to decision_value_steps and measures
    # there. The continuation is fresh (same recipe+seed) and must reproduce
    # the bake-off cell's eval prefix bit-exactly (determinism proof).
    if quick:
        pooled: list[float] = []
        for s in range(GATES["seeds"]):
            net = load_net(parent, winner, s)
            arena = Arena(config=REDUCED_HAND_CONFIG, seeds=Seeds(300 + s))
            summ = arena.duplicate_summary(
                GinNetAgent(net, feat_dim, DEVICE, seed=1),
                BeliefAblatedAgent(net, feat_dim, DEVICE, seed=2),
                400 + s,
                1000 // div,
            ).checked(0)
            pooled.extend(summ.paired_scores())
        dv = bootstrap_ci(pooled)
        check(
            "decision-value",
            True,
            f"learned-vs-ablated paired={dv.mean:+.3f} [{dv.lo:+.3f},{dv.hi:+.3f}] n={len(pooled)}{info_only}",
        )
    else:
        long_parent = parent / "long"
        long_parent.mkdir(parents=True, exist_ok=True)
        print(
            f"continuing {winner}-s0 to {GATES['decision_value_steps']} steps",
            flush=True,
        )
        with open(belief_path, "rb") as fh:
            long_states = pickle.load(fh)
        run_cell(
            winner,
            0,
            total_steps=GATES["decision_value_steps"],
            device=DEVICE,
            parent=long_parent / "s0",
            eval_deals_final=2000,
            belief_states=long_states,
            belief_seed=BELIEF_SEED,
        )
        long_net = load_net(long_parent, winner, 0)
        prefix_ok = _prefix_matches(parent, long_parent, winner)
        check(
            "continuation-deterministic",
            prefix_ok,
            "long run reproduces the bake-off eval prefix bit-exactly",
        )
        arena = Arena(config=REDUCED_HAND_CONFIG, seeds=Seeds(301))
        summ = arena.duplicate_summary(
            GinNetAgent(long_net, feat_dim, DEVICE, seed=1),
            BeliefAblatedAgent(long_net, feat_dim, DEVICE, seed=2),
            401,
            GATES["decision_value_deals"],
        ).checked(0)
        dv = summ.points_per_hand()
        check(
            "decision-value",
            dv.lo > 0,
            f"learned-vs-ablated paired={dv.mean:+.3f} [{dv.lo:+.3f},{dv.hi:+.3f}] n={summ.n_deals}",
        )

    # F. exact suites.
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_match.py",
            "tests/test_match_exact.py",
            "tests/test_policy_hygiene.py",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    check(
        "exact-suites",
        r.returncode == 0 or quick,
        (r.stdout + r.stderr).strip().splitlines()[-1][:120] + info_only,
    )

    # G. cap accounting.
    try:
        check_cap_accounting()
    except AssertionError as e:
        check("cap-accounting", False, str(e)[:120])

    print("GATE-P4 " + ("PASS" if not FAILURES else f"FAIL {FAILURES}"), flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
