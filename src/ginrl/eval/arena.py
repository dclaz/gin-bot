"""Evaluation harness: duplicate-deal head-to-head with paired statistics."""

from __future__ import annotations

from dataclasses import dataclass, field

import pyspiel
from pyspiel import gin_rummy as gr

from ginrl.agents.protocol import Agent
from ginrl.config import HandConfig, Seeds
from ginrl.env.game import HandEnv
from ginrl.eval.stats import Summary, bootstrap_ci, require


@dataclass(frozen=True)
class ActionRecord:
    seat: int
    action: int  # engine action id (0..55 learned, anything at manual phases)
    manual: bool


@dataclass
class GameResult:
    returns: tuple[float, float]
    seed: int
    seats: tuple[str, str]  # agent names at seat 0, seat 1
    manual_seats: frozenset[int]
    actions: list[ActionRecord]  # every driver-applied action; replays bit-exactly
    # Learned-phase decisions only (~26/game). The ~34 reference counts
    # total_player_actions: learned plus manual knock/layoff actions.
    decisions: int
    knock_seat: int | None = None  # seat that knocked; None on a wall
    knock_turns: int | None = None  # turns taken by knock_seat (draw decisions)
    deadwood: tuple[int, int] | None = None  # per-seat deadwood at hand end

    @property
    def total_player_actions(self) -> int:
        """Every driver-applied player action, learned and manual phases."""
        return len(self.actions)

    @property
    def wall(self) -> bool:
        """No knock: stock exhaustion or an engine no-progress draw.

        Distinct from a scoreless tied knock (knock with level deadwood).
        """
        return self.knock_seat is None

    @property
    def gin_seat(self) -> int | None:
        """Knock with zero deadwood, else None."""
        if self.knock_seat is not None and self.deadwood is not None:
            if self.deadwood[self.knock_seat] == 0:
                return self.knock_seat
        return None

    @property
    def undercut(self) -> bool:
        """Someone knocked and the knocker lost the hand."""
        return self.knock_seat is not None and self.returns[self.knock_seat] < 0


@dataclass
class PairResult:
    score_a: float  # leg1 returns[seat of A] + leg2 returns[seat of A]
    leg1: GameResult
    leg2: GameResult


def play_game(agent_a: Agent, agent_b: Agent, seed: int, env: HandEnv) -> GameResult:
    """One hand, agent_a at seat 0. Fresh bot state; full inform stream."""
    agents = (agent_a, agent_b)
    names = (agent_a.name, agent_b.name)
    agent_a.begin_game(0)
    agent_b.begin_game(1)
    env.manual_phases = {s for s, a in enumerate(agents) if a.manual_phases}

    def listener(acting_seat: int, state: pyspiel.State, player: int, action: int) -> None:
        for agent, seat in zip(agents, (0, 1), strict=True):
            if seat != acting_seat:
                agent.inform(state, player, action)

    env.listener = listener
    try:
        actions: list[ActionRecord] = []
        decisions = 0
        knock_index: int | None = None
        r = env.reset(seed=seed)
        while not r.done:
            seat = env.state().current_player()
            assert seat in (0, 1)
            agent = agents[seat]
            phase = str(env.state().to_dict()["phase"])
            if phase in ("Knock", "Layoff") and agent.manual_phases:
                if phase == "Knock" and knock_index is None:
                    knock_index = len(actions)
                action = agent.choose_raw(env.state())
                actions.append(ActionRecord(seat, action, True))
                r = env.raw_step(action)
            else:
                action = agent.choose(env, seat)
                if action == gr.KNOCK_ACTION and knock_index is None:
                    knock_index = len(actions)
                actions.append(ActionRecord(seat, action, False))
                decisions += 1
                r = env.step(action)
        assert r.returns is not None
        end = env.state().to_dict()
        knocked = [bool(k) for k in end["knocked"]]
        knock_seat: int | None = next((s for s, k in enumerate(knocked) if k), None)
        knock_turns: int | None = None
        if knock_seat is not None and knock_index is not None:
            # One turn opens with exactly one draw/pass decision, so counting
            # the knocker's draw and first-upcard pass actions counts turns.
            knock_turns = sum(
                1
                for rec in actions[:knock_index]
                if rec.seat == knock_seat
                and not rec.manual
                and rec.action in (gr.DRAW_UPCARD_ACTION, gr.DRAW_STOCK_ACTION, gr.PASS_ACTION)
            )
        return GameResult(
            returns=r.returns,
            seed=seed,
            seats=names,
            manual_seats=frozenset(env.manual_phases),
            actions=actions,
            decisions=decisions,
            knock_seat=knock_seat,
            knock_turns=knock_turns,
            deadwood=(int(end["deadwood"][0]), int(end["deadwood"][1])),
        )
    finally:
        env.listener = None
        env.manual_phases = set()


def replay(result: GameResult, env: HandEnv) -> tuple[float, float]:
    """Re-apply a recorded action list. Returns must match bit-exactly."""
    env.manual_phases = set(result.manual_seats)
    r = env.reset(seed=result.seed)
    for record in result.actions:
        assert not r.done
        r = env.raw_step(record.action) if record.manual else env.step(record.action)
    assert r.done and r.returns is not None
    env.manual_phases = set()
    return r.returns


@dataclass
class PairSummary:
    """Paired duplicate-deal summary from A's perspective.

    The unit is the deal: each deal contributes one paired score (leg1 +
    leg2 returns for A), which cancels the deal's luck. Bootstrap resamples
    deals, never legs.
    """

    agent_a: str
    agent_b: str
    pairs: list[PairResult]
    deal_seeds: list[int]

    @property
    def n_deals(self) -> int:
        return len(self.pairs)

    @property
    def n_legs(self) -> int:
        return 2 * len(self.pairs)

    def paired_scores(self) -> list[float]:
        return [p.score_a for p in self.pairs]

    def legs_for_a(self) -> list[float]:
        """Per-leg returns from A's perspective (seat-corrected)."""
        out = []
        for p in self.pairs:
            out.append(p.leg1.returns[0])  # A is seat 0 in leg 1
            out.append(p.leg2.returns[1])  # A is seat 1 in leg 2
        return out

    def wins_losses_ties(self) -> tuple[int, int, int]:
        wins = sum(1 for s in self.paired_scores() if s > 0)
        losses = sum(1 for s in self.paired_scores() if s < 0)
        return wins, losses, len(self.pairs) - wins - losses

    def points_per_hand(self) -> Summary:
        """Mean leg score for A. Resamples deals (each deal's two-leg mean),
        so the paired structure survives the bootstrap."""
        per_deal = [(p.leg1.returns[0] + p.leg2.returns[1]) / 2 for p in self.pairs]
        return bootstrap_ci(per_deal)

    def checked(self, min_deals: int) -> PairSummary:
        """Refuse to report below the resolved sample count."""
        require(self.n_deals, min_deals, f"{self.agent_a} vs {self.agent_b}")
        return self


def summarize(
    agent_a: Agent, agent_b: Agent, pairs: list[PairResult], deal_seeds: list[int]
) -> PairSummary:
    assert len(pairs) == len(deal_seeds)
    return PairSummary(
        agent_a=agent_a.name, agent_b=agent_b.name, pairs=pairs, deal_seeds=deal_seeds
    )


@dataclass
class Arena:
    config: HandConfig = field(default_factory=HandConfig)
    seeds: Seeds = field(default_factory=Seeds)

    def play_game(self, agent_a: Agent, agent_b: Agent, seed: int) -> GameResult:
        return play_game(agent_a, agent_b, seed, HandEnv(self.config, self.seeds))

    def duplicate(
        self, agent_a: Agent, agent_b: Agent, seed: int, n_deals: int
    ) -> list[PairResult]:
        """Every deal twice, seats swapped, same seed: common random numbers."""
        return self.duplicate_summary(agent_a, agent_b, seed, n_deals).pairs

    def duplicate_summary(
        self, agent_a: Agent, agent_b: Agent, seed: int, n_deals: int
    ) -> PairSummary:
        """Duplicate deals with their seeds, bundled for summaries and records."""
        pairs = []
        deal_seeds = []
        for i in range(n_deals):
            deal_seed = seed * 1_000_003 + i
            leg1 = self.play_game(agent_a, agent_b, deal_seed)
            leg2 = self.play_game(agent_b, agent_a, deal_seed)
            # A is seat 0 in leg 1, seat 1 in leg 2.
            pairs.append(
                PairResult(
                    score_a=leg1.returns[0] + leg2.returns[1],
                    leg1=leg1,
                    leg2=leg2,
                )
            )
            deal_seeds.append(deal_seed)
        return summarize(agent_a, agent_b, pairs, deal_seeds)
