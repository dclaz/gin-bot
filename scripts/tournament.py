"""Phase 6 tournament: duplicate-deal round-robin over a member list.

Writes a payoff matrix (mean paired margin + CI per cell, A's perspective)
plus the Nash-averaging headline. Members are built fresh per duel from
specs, so stateful engine bots stay correct. Eval is always the TRUE game
payoff, regardless of what any member was trained on (METHODOLOGY §6).

Usage:
    uv run python scripts/tournament.py --members free --deals 400 --out runs/p6_free
    uv run python scripts/tournament.py --members free --deals 40 --out /tmp/tourney_smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from ginrl.agents.baselines import HeuristicAgent, HeuristicParams  # noqa: E402
from ginrl.agents.learned import GinNetAgent  # noqa: E402
from ginrl.config import HandConfig, Seeds  # noqa: E402
from ginrl.env.features import feature_dim  # noqa: E402
from ginrl.eval.arena import Arena  # noqa: E402
from ginrl.eval.population import nash_averaging  # noqa: E402
from ginrl.nets.gin import GinNet  # noqa: E402

DEVICE = torch.device("cpu")
DECK = HandConfig().deck_size
FEAT_DIM = feature_dim(DECK)
DUEL_SEED = 11


def load_champ(path: Path) -> GinNet:
    net = GinNet("mlp", feat_dim=FEAT_DIM, deck=DECK, hidden=128)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


def free_members(champ_path: Path) -> list[tuple[str, object]]:
    """No-training-cost population: champion, heuristic sweep, temperatures."""
    net = load_champ(champ_path)
    members: list[tuple[str, object]] = [("nash", GinNetAgent(net, FEAT_DIM, DEVICE, seed=7))]
    members[0][1].name = "nash"
    for t in (0, 5, 10):
        members.append(
            (f"heur-T{t}", HeuristicAgent(HeuristicParams(knock_threshold=t), name=f"heur-T{t}"))
        )
    for tau in (0.5, 1.0, 2.0):
        agent = GinNetAgent(net, FEAT_DIM, DEVICE, seed=7, temperature=tau)
        agent.name = f"nash-t{tau}"
        members.append((f"nash-t{tau}", agent))
    return members


def run_matrix(members: list[tuple[str, object]], deals: int, out: Path) -> dict:
    names = [name for name, _ in members]
    arena = Arena(config=HandConfig(), seeds=Seeds(DUEL_SEED))
    mean = [[0.0] * len(names) for _ in names]
    lo = [[0.0] * len(names) for _ in names]
    hi = [[0.0] * len(names) for _ in names]
    for i, (na, a) in enumerate(members):
        for j, (nb, b) in enumerate(members):
            if i == j:
                continue
            summ = arena.duplicate_summary(a, b, DUEL_SEED, deals).checked(0)
            pph = summ.points_per_hand()
            mean[i][j], lo[i][j], hi[i][j] = pph.mean, pph.lo, pph.hi
            print(
                f"   {na:10} vs {nb:10} {pph.mean:+.2f} [{pph.lo:+.2f},{pph.hi:+.2f}]", flush=True
            )
    import numpy as np

    mix = nash_averaging(names, np.array(mean))
    result = {
        "names": names,
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "n_deals": deals,
        "duel_seed": DUEL_SEED,
        "nash_mass": {n: float(mix.weights[n]) for n in names},
        "provenance": {
            "git_sha": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
            ).stdout.strip(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "payoff.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", choices=["free"], default="free")
    parser.add_argument("--deals", type=int, default=400)
    parser.add_argument("--out", required=True)
    parser.add_argument("--champ", default="runs/p5_s0/best.pt")
    args = parser.parse_args()
    members = free_members(Path(args.champ))
    print(f"tournament: {[n for n, _ in members]} x {args.deals} deals", flush=True)
    result = run_matrix(members, args.deals, Path(args.out))
    print("nash mass:", {k: round(v, 3) for k, v in result["nash_mass"].items()}, flush=True)


if __name__ == "__main__":
    main()
