"""Behavioural profiler: what kind of player is an agent, in game units.

Computed for any agent from recorded games (no model access needed):
gin/knock/undercut/wall rates, turns to knock, deadwood at knock, the
deadwood-by-turn curve, pile-draw rate, and the danger-discard rate.

A discard is *dangerous* when its rank matches a rank the opponent picked up
from the discard pile earlier in the hand — the cheapest observable proxy
for "a card the opponent is known to want".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pyspiel import gin_rummy as gr

from ginrl.env import melds
from ginrl.env.game import HandEnv
from ginrl.eval.arena import ActionRecord, GameResult

DRAW_ACTIONS = (gr.DRAW_UPCARD_ACTION, gr.DRAW_STOCK_ACTION)


@dataclass
class StyleProfile:
    agent: str
    n_hands: int
    gin_rate: float
    knock_rate: float  # non-gin knocks by this agent
    undercut_rate: float  # this agent knocked and lost
    wall_rate: float
    mean_turns_to_knock: float
    n_knocks: int
    mean_deadwood_at_knock: float
    pile_draw_rate: float
    n_draws: int
    danger_discard_rate: float
    n_discards: int
    deadwood_by_turn: list[float] = field(default_factory=list)
    deadwood_by_turn_n: list[int] = field(default_factory=list)


def profile(agent: str, results: list[GameResult], make_env: Callable[[], HandEnv]) -> StyleProfile:
    """Profile `agent` over games it played (either seat). Replays for style."""
    mine = [g for g in results if agent in g.seats]
    n = len(mine)
    if n == 0:
        raise ValueError(f"no games for {agent!r}")
    gins, my_knocks, undercuts, walls, turns, deadwoods = _tally_endings(mine, agent)
    draws, pile, discards, danger, by_turn = _tally_replay(mine, agent, make_env)
    curve = []
    counts = []
    for t in sorted(by_turn):
        vals = by_turn[t]
        curve.append(sum(vals) / len(vals))
        counts.append(len(vals))
    return StyleProfile(
        agent=agent,
        n_hands=n,
        gin_rate=gins / n,
        knock_rate=my_knocks / n,
        undercut_rate=undercuts / n,
        wall_rate=walls / n,
        mean_turns_to_knock=sum(turns) / len(turns) if turns else 0.0,
        n_knocks=gins + my_knocks,
        mean_deadwood_at_knock=sum(deadwoods) / len(deadwoods) if deadwoods else 0.0,
        pile_draw_rate=pile / draws if draws else 0.0,
        n_draws=draws,
        danger_discard_rate=danger / discards if discards else 0.0,
        n_discards=discards,
        deadwood_by_turn=curve,
        deadwood_by_turn_n=counts,
    )


def _tally_endings(
    mine: list[GameResult], agent: str
) -> tuple[int, int, int, int, list[int], list[int]]:
    """Knock/gin/undercut/wall counts plus turns and deadwood at knock."""
    turns: list[int] = []
    deadwoods: list[int] = []
    gins = my_knocks = undercuts = walls = 0
    for g in mine:
        seat = g.seats.index(agent)
        if g.wall:
            walls += 1
            continue
        if g.knock_seat == seat:
            if g.gin_seat == seat:
                gins += 1
            else:
                my_knocks += 1
            if g.undercut:
                undercuts += 1
            if g.knock_turns is not None:
                turns.append(g.knock_turns)
            if g.deadwood is not None:
                deadwoods.append(g.deadwood[seat])
    return gins, my_knocks, undercuts, walls, turns, deadwoods


def _tally_replay(
    mine: list[GameResult], agent: str, make_env: Callable[[], HandEnv]
) -> tuple[int, int, int, int, dict[int, list[int]]]:
    """Replay games: pile draws, danger discards, deadwood at each own turn."""
    draws = pile = discards = danger = 0
    by_turn: dict[int, list[int]] = {}
    for g in mine:
        seat = g.seats.index(agent)
        env = make_env()
        env.manual_phases = set(g.manual_seats)
        r = env.reset(seed=g.seed)
        pickups: list[set[int]] = [set(), set()]  # ranks taken from the pile
        for rec in g.actions:
            assert not r.done
            info = env.state().to_dict() if _needs_dict(rec) else None
            if info is not None and not rec.manual and rec.action == gr.DRAW_UPCARD_ACTION:
                pickups[rec.seat].add(_pile_top_rank(info))
            if rec.seat == seat and not rec.manual:
                if rec.action in DRAW_ACTIONS:
                    draws += 1
                    if rec.action == gr.DRAW_UPCARD_ACTION:
                        pile += 1
                    if info is not None:
                        turn = draws  # my t-th turn opens with my t-th draw
                        by_turn.setdefault(turn, []).append(int(info["deadwood"][seat]))
                elif 0 <= rec.action < gr.DRAW_UPCARD_ACTION:
                    discards += 1
                    if (rec.action % melds.NUM_RANKS) in pickups[1 - seat]:
                        danger += 1
            r = env.raw_step(rec.action) if rec.manual else env.step(rec.action)
        assert r.done
        env.manual_phases = set()
    return draws, pile, discards, danger, by_turn


def _needs_dict(rec: ActionRecord) -> bool:
    """Only pile takes, own draws, and own discards need the state dict."""
    return not rec.manual and (
        rec.action == gr.DRAW_UPCARD_ACTION
        or rec.action in DRAW_ACTIONS
        or 0 <= rec.action < gr.DRAW_UPCARD_ACTION
    )


def _pile_top_rank(info: dict) -> int:
    """Rank of the takeable top: the upcard when present, else the pile top."""
    upcard = info.get("upcard")
    if isinstance(upcard, str):
        return melds.card_to_index(upcard) % melds.NUM_RANKS
    pile = info.get("discard_pile")
    if isinstance(pile, list) and pile:
        return melds.card_to_index(pile[-1]) % melds.NUM_RANKS
    return -1


def table_row(p: StyleProfile) -> list[str]:
    """One profiler-table row (gate-p2 prints one per baseline)."""
    return [
        p.agent,
        str(p.n_hands),
        f"{p.gin_rate:.3f}",
        f"{p.knock_rate:.3f}",
        f"{p.undercut_rate:.3f}",
        f"{p.wall_rate:.4f}",
        f"{p.mean_turns_to_knock:.2f}",
        f"{p.mean_deadwood_at_knock:.2f}",
        f"{p.pile_draw_rate:.3f}",
        f"{p.danger_discard_rate:.3f}",
    ]


TABLE_COLUMNS = [
    "agent",
    "hands",
    "gin",
    "knock",
    "undercut",
    "wall",
    "turns_to_knock",
    "deadwood_at_knock",
    "pile_draw",
    "danger_discard",
]
