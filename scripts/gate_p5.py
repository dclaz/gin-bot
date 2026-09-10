"""Phase 5 gate: full game, matches to 100.

Reads thresholds from configs/gates.yaml (gate-p5). Full mode, given a
training run dir with best.pt + game_record.jsonl + metrics.jsonl:

  A. champion vs SimpleGinRummyBot over duplicate_deals_min deals, pph CI
     above simplebot_pph_margin;
  B. duplicate matches to 100 vs the anchor, win-rate CI above 50%;
  C. knock threshold moves with the score: deadwood-at-knock differs
     between losing-big and winning-big buckets, CI-separated;
  D. beats HeuristicAgent (and --prev champion when given), CIs above zero;
  E. RL-BR bound on the champion below every baseline bound (parallel cells);
  F. refit run record: Elo gain over start above margin (CI excludes start),
     cyclic_fraction below max;
  G. first-player edge CI includes zero;
  H. max logging_overhead_frac below max;
  I. reproducibility: same-seed eval repeats bit-identically.

`--quick` runs every check at tiny budgets, info-only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pyspiel.gin_rummy import KNOCK_ACTION  # noqa: E401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyspiel  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from ginrl.agents.baselines import HeuristicAgent  # noqa: E402
from ginrl.agents.learned import GinNetAgent  # noqa: E402
from ginrl.agents.spiel_bots import SimpleGinRummyAgent  # noqa: E402
from ginrl.config import HandConfig, MatchConfig, Seeds  # noqa: E402
from ginrl.env.features import feature_dim  # noqa: E402
from ginrl.env.match import MatchEnv  # noqa: E402
from ginrl.env.melds import layout_from_params, parse_hand  # noqa: E402
from ginrl.eval.arena import Arena  # noqa: E402
from ginrl.eval.ratings import fit_ratings  # noqa: E402
from ginrl.eval.record import load_deals  # noqa: E402
from ginrl.eval.stats import bootstrap_ci  # noqa: E402
from ginrl.nets.gin import GinNet  # noqa: E402
from ginrl.train.match import drive, match_win_rate, wire  # noqa: E402

GATES = yaml.safe_load((ROOT / "configs" / "gates.yaml").read_text())["gate-p5"]
FAILURES: list[str] = []
DEVICE = torch.device("cpu")
DECK = HandConfig().deck_size
FEAT_DIM = feature_dim(DECK)
_UTILS = pyspiel.gin_rummy.GinRummyUtils(13, 4, 10)


def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def load_champ(run: Path, name: str) -> GinNet:
    net = GinNet("mlp", feat_dim=FEAT_DIM, deck=DECK, hidden=128)
    net.load_state_dict(torch.load(run / name, map_location="cpu", weights_only=True))
    return net.eval()


def knock_curve(net: GinNet, n_matches: int, seed: int) -> tuple[list[float], list[float]]:
    """Deadwood-at-knock for the champion when losing big vs winning big.

    Drives duplicate matches vs the heuristic; at each champion knock records
    (seat-relative score diff, exact deadwood via GinRummyUtils). Returns
    (losing_deadwoods, winning_deadwoods) for |diff| >= 50.
    """
    champ = GinNetAgent(net, FEAT_DIM, DEVICE, seed=seed)
    champ.name = "champ"
    heur = HeuristicAgent()
    losing, winning = [], []
    layout, _ = layout_from_params(HandConfig().game_params())
    for m in range(n_matches):
        for swap in (False, True):
            env = MatchEnv(
                config=MatchConfig(
                    hand=HandConfig(),
                    target_score=100,
                    max_hands=50,
                    auto_advance=False,
                ),
                seeds=Seeds(seed),
            )
            agents = (champ, heur) if not swap else (heur, champ)
            mine = 0 if not swap else 1
            wire(env, agents)
            r = env.reset(seed=seed * 1_000_003 + m)
            while not r.done:
                hand = env.hand
                cur = hand.state().current_player()
                if cur == mine and agents[cur] is champ:
                    action = champ.choose(hand, cur)
                    if action == KNOCK_ACTION:
                        cards = parse_hand(hand.state().to_dict()["hands"][cur], layout)
                        dw = _UTILS.min_deadwood(tuple(sorted(cards)))
                        diff = r.scores[mine] - r.scores[1 - mine]
                        if diff <= -50:
                            losing.append(float(dw))
                        elif diff >= 50:
                            winning.append(float(dw))
                    r = env.step(action)
                else:
                    r = drive(env, agents)
                if r.new_hand_pending:
                    wire(env, agents)
                    r = env.next_hand()
    return losing, winning


def run_br_cells(parent: Path, champ_ckpt: str, steps: int, eval_deals: int) -> dict:
    """Four BR cells in parallel; returns {target: bound}."""

    def one(target: str) -> str:
        env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "br_cell.py"),
            "--target",
            target,
            "--seed",
            "0",
            "--steps",
            str(steps),
            "--parent",
            str(parent),
            "--eval-deals",
            str(eval_deals),
        ]
        if target == "champion":
            cmd += ["--champ", champ_ckpt]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, **env})
        tail = "\n".join((r.stderr or "").splitlines()[-8:])
        return f"{target} exit={r.returncode}" + (f"\n{tail}" if r.returncode else "")

    with ThreadPoolExecutor(max_workers=4) as pool:
        for line in pool.map(one, ["champion", "heur", "bot", "random"]):
            print("   ", line, flush=True)
    bounds = {}
    for target in ["champion", "heur", "bot", "random"]:
        cell = json.loads((parent / f"br-{target}-s0" / "br.json").read_text())
        bounds[target] = cell["bound"]
    return bounds


def metric_series(run: Path, metric: str) -> list[tuple[int, float]]:
    out = []
    for line in (run / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "scalar" and row.get("metric") == metric:
            out.append((row["step"], row["value"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/p5_s0")
    ap.add_argument("--agent", default="p5s0")
    ap.add_argument("--prev", default=None)
    ap.add_argument("--br-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    quick = args.quick
    div = 50 if quick else 1
    info_only = " (info-only)" if quick else ""
    run = Path(args.run)
    if not (run / "best.pt").exists():
        if not quick:
            check("champion-present", False, f"no best.pt in {run}")
            print("GATE-P5 FAIL ['champion-present']", flush=True)
            return 1
        from ginrl.config import TrainerConfig
        from ginrl.train.match import train_match

        run = Path(tempfile.mkdtemp(prefix="gate_p5_quick_"))
        print(f"quick: training micro run in {run}", flush=True)
        cfg = TrainerConfig(total_steps=1024, n_envs=2, rollout_len=32, epochs=1, minibatches=1)
        train_match(
            torso="mlp",
            cfg=cfg,
            master_seed=0,
            device=DEVICE,
            run_dir=run,
            eval_every=8,
            eval_deals=4,
            snapshot_every=4,
            match_probe_every=10**9,
            probe_matches=2,
            belief_states_n=20,
            agent_name="q",
        )
    champ = load_champ(run, "best.pt")

    # A. champion vs anchor over duplicate deals.
    n_a = GATES["duplicate_deals_min"] // div
    arena = Arena(config=HandConfig(), seeds=Seeds(11))
    agent = GinNetAgent(champ, FEAT_DIM, DEVICE, seed=11)
    agent.name = "champ"
    summ_a = arena.duplicate_summary(agent, SimpleGinRummyAgent(), 11, n_a).checked(0)
    pph = summ_a.points_per_hand()
    check(
        "beats-anchor",
        pph.lo > GATES["simplebot_pph_margin"] or quick,
        f"pph={pph.mean:+.2f} [{pph.lo:+.2f},{pph.hi:+.2f}] n={n_a}{info_only}",
    )

    # B. duplicate matches to 100.
    n_b = 300 // div
    mc = MatchConfig(hand=HandConfig(), target_score=100, max_hands=50)
    wmean, wlo, whi = match_win_rate(champ, FEAT_DIM, DEVICE, mc, 21, n_b)
    check(
        "match-win-rate",
        wlo > 0.5 or quick,
        f"win={wmean:.3f} [{wlo:.3f},{whi:.3f}] n={2 * n_b}{info_only}",
    )

    # C. knock threshold moves with the score.
    losing, winning = knock_curve(champ, 150 // div, 31)
    n_min = GATES.get("knock_min_bucket_n", 30)
    if len(losing) >= n_min and len(winning) >= n_min:
        gap = bootstrap_ci([lose - win for lose in losing for win in winning][:10000])
        check(
            "knock-moves-with-score",
            (gap.lo > 0 or gap.hi < 0) or quick,
            f"lose_dw={sum(losing) / len(losing):.1f} win_dw={sum(winning) / len(winning):.1f} "
            f"gap=[{gap.lo:+.2f},{gap.hi:+.2f}] n={len(losing)}/{len(winning)}{info_only}",
        )
    else:
        check(
            "knock-moves-with-score",
            quick,
            f"thin buckets lose={len(losing)} win={len(winning)}{info_only}",
        )

    # D. beats heuristic (+ previous champion when given).
    n_d = 2000 // div
    arena_d = Arena(config=HandConfig(), seeds=Seeds(41))
    summ_d = arena_d.duplicate_summary(agent, HeuristicAgent(), 41, n_d).checked(0)
    ppd = summ_d.points_per_hand()
    check(
        "beats-heuristic",
        ppd.lo > 0 or quick,
        f"pph={ppd.mean:+.2f} [{ppd.lo:+.2f},{ppd.hi:+.2f}]{info_only}",
    )
    if args.prev:
        prev = load_champ(Path(args.prev).parent, Path(args.prev).name)
        prev_agent = GinNetAgent(prev, FEAT_DIM, DEVICE, seed=42)
        prev_agent.name = "prev"
        summ_p = arena_d.duplicate_summary(agent, prev_agent, 42, n_d).checked(0)
        ppp = summ_p.points_per_hand()
        check(
            "beats-prev-champion",
            ppp.lo > 0 or quick,
            f"pph={ppp.mean:+.2f} [{ppp.lo:+.2f},{ppp.hi:+.2f}]{info_only}",
        )

    # E. RL-BR bounds.
    if args.br_dir:
        parent = Path(args.br_dir)
    else:
        parent = Path(tempfile.mkdtemp(prefix="gate_p5_br_"))
        run_br_cells(parent, str(run / "best.pt"), 300000 // div, 1000 // div)
    bounds = {
        t: json.loads((parent / f"br-{t}-s0" / "br.json").read_text())["bound"]
        for t in ["champion", "heur", "bot", "random"]
    }
    champ_lowest = all(bounds["champion"] < bounds[t] for t in ["heur", "bot", "random"])
    check("rlbr-bound", champ_lowest or quick, f"{bounds}{info_only}")

    # F/G. refit run record: Elo gain, cyclic, first-player edge.
    deals = load_deals(run / "game_record.jsonl")
    fit = fit_ratings(deals, anchor="simple_bot")
    elos = metric_series(run, "ratings/current_elo")
    widths = dict(metric_series(run, "ratings/elo_ci_width"))
    gain = elos[-1][1] - elos[0][1] if len(elos) >= 2 else 0.0
    ci_ok = (elos[-1][1] - widths.get(elos[-1][0], 0.0)) > elos[0][1]
    check(
        "elo-rising",
        (gain > GATES["elo_gain_margin"] and ci_ok) or quick,
        f"gain={gain:+.0f} margin={GATES['elo_gain_margin']}{info_only}",
    )
    check(
        "cyclic-ok",
        fit.cyclic_fraction < GATES["cyclic_fraction_max"] or quick,
        f"cyclic={fit.cyclic_fraction:.3f}{info_only}",
    )
    elo, ehi = fit.edge_elo_ci()
    check("first-player-edge", (elo < 0 < ehi) or quick, f"edge=[{elo:+.1f},{ehi:+.1f}]{info_only}")

    # H. overhead over the full run.
    overhead = metric_series(run, "perf/logging_overhead_frac")
    peak = max(v for _, v in overhead) if overhead else 1.0
    check(
        "logging-overhead",
        peak < GATES["logging_overhead_frac_max"] or quick,
        f"peak={peak:.4f}{info_only}",
    )

    # I. reproducibility: same-seed eval repeats bit-identically.
    # Fresh champion objects per repeat: GinNetAgent samples from its own RNG
    # (begin_game does not reseed), so reusing one object across repeats would
    # advance the stream. The contract is same seed + fresh agents => same result.
    def _fresh_champ() -> GinNetAgent:
        a = GinNetAgent(champ, FEAT_DIM, DEVICE, seed=11)
        a.name = "champ"
        return a

    s1 = arena.duplicate_summary(_fresh_champ(), SimpleGinRummyAgent(), 11, 200 // div).checked(0)
    s2 = arena.duplicate_summary(_fresh_champ(), SimpleGinRummyAgent(), 11, 200 // div).checked(0)
    det = [(p.leg1.returns, p.leg2.returns) for p in s1.pairs] == [
        (p.leg1.returns, p.leg2.returns) for p in s2.pairs
    ]
    check("reproducible-eval", det, "same-seed repeat bit-identical")

    print("GATE-P5 " + ("PASS" if not FAILURES else f"FAIL {FAILURES}"), flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
