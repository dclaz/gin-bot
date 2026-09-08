"""Refit anchored Bradley-Terry ratings from the game record; print the ladder."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ginrl.eval.ratings import fit_ratings  # noqa: E402
from ginrl.eval.record import load_deals  # noqa: E402

RECORD = Path("runs/game_record.jsonl")
ANCHOR = "simple_bot"


def main() -> int:
    if not RECORD.exists():
        print(f"no game record yet: {RECORD} does not exist")
        return 1
    deals = load_deals(RECORD)
    if not deals:
        print(f"no complete deals in {RECORD}")
        return 1
    names = sorted({n for d in deals for n in (d.a, d.b)})
    if ANCHOR not in names:
        print(f"anchor {ANCHOR!r} has no games; agents seen: {names}")
        return 1
    res = fit_ratings(deals, ANCHOR)
    print(f"deals={res.n_deals} legs={res.n_legs} tied_deals={res.n_tied_deals}")
    print(f"cyclic_fraction={res.cyclic_fraction:.4f}")
    lo, hi = res.edge_elo_ci()
    print(f"first_player_edge_elo={res.edge_elo:+.1f} [{lo:+.1f}, {hi:+.1f}]")
    print(f"{'agent':<28} {'elo':>7} {'ci95':>15}")
    for name in sorted(res.names, key=res.elo, reverse=True):
        elo_lo, elo_hi = res.elo_ci(name)
        print(f"{name:<28} {res.elo(name):+7.1f} [{elo_lo:+.1f}, {elo_hi:+.1f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
