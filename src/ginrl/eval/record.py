"""Append-only game record: `runs/game_record.jsonl` is the source of truth.

Every evaluation leg appends one row. Ratings are always refit from this
file (`load_deals`), never accumulated in memory, so any rating is
reproducible from disk. The first line is a provenance header.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ginrl.eval.arena import PairSummary
from ginrl.eval.ratings import DealPair, LegRecord
from ginrl.telemetry.provenance import git_sha, spiel_facts_hash

LEG = "leg"
HEADER = "header"


def ensure_header(path: Path, config_hash: str = "") -> None:
    """Write the provenance header if the record does not exist yet."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "type": HEADER,
                    "git_sha": git_sha(),
                    "config_hash": config_hash,
                    "facts_hash": spiel_facts_hash(),
                    "t": time.time(),
                }
            )
            + "\n"
        )


def _first_won(returns: tuple[float, float]) -> bool | None:
    if returns[0] == returns[1]:
        return None
    return returns[0] > returns[1]


def append_summary(path: Path, summary: PairSummary, config_hash: str = "") -> int:
    """Append every leg of a duplicate summary. Returns rows written."""
    ensure_header(path, config_hash)
    rows = 0
    with path.open("a", encoding="utf-8") as f:
        for pair, deal in zip(summary.pairs, summary.deal_seeds, strict=True):
            for leg, a_seat in ((pair.leg1, 0), (pair.leg2, 1)):
                f.write(
                    json.dumps(
                        {
                            "type": LEG,
                            "a": summary.agent_a,
                            "b": summary.agent_b,
                            "a_first": a_seat == 0,
                            "first_won": _first_won(leg.returns),
                            "margin_a": leg.returns[a_seat],
                            "deal": deal,
                            "wall": leg.wall,
                        }
                    )
                    + "\n"
                )
                rows += 1
    return rows


def deals_from_summary(summary: PairSummary) -> list[DealPair]:
    """Refit input straight from a duplicate summary (no disk round-trip)."""
    deals = []
    for pair in summary.pairs:
        legs = []
        for leg, a_first in ((pair.leg1, True), (pair.leg2, False)):
            legs.append(
                LegRecord(
                    a=summary.agent_a,
                    b=summary.agent_b,
                    a_first=a_first,
                    first_won=_first_won(leg.returns),
                )
            )
        deals.append(
            DealPair(
                a=summary.agent_a,
                b=summary.agent_b,
                score_a=pair.score_a,
                legs=(legs[0], legs[1]),
            )
        )
    return deals


def load_deals(path: Path) -> list[DealPair]:
    """Refit input: paired deals rebuilt from leg rows, grouped by deal."""
    legs: dict[tuple[str, str, int], list[dict]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("type") != LEG:
                continue
            key = (row["a"], row["b"], int(row["deal"]))
            legs.setdefault(key, []).append(row)
    deals = []
    for (a, b, _), rows in sorted(legs.items(), key=lambda kv: kv[0][2]):
        by_seat = sorted(rows, key=lambda r: not r["a_first"])
        if len(by_seat) != 2:
            continue  # incomplete deal (crashed run): skip, do not guess
        leg_recs = []
        score_a = 0.0
        for row in by_seat:
            leg_recs.append(LegRecord(a=a, b=b, a_first=row["a_first"], first_won=row["first_won"]))
            margin = float(row["margin_a"])
            score_a += margin
        deals.append(DealPair(a=a, b=b, score_a=score_a, legs=(leg_recs[0], leg_recs[1])))
    return deals
