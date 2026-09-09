"""Single-hand gin rummy wrapper: the learned policy's view of the game.

The learned policy sees 56 actions (identity-mapped to engine actions
0..55: 52 discards, draw upcard, draw stock, pass, knock). Meld declaration,
the knock discard, layoffs and wall passes are resolved automatically by
meld_policy — deterministic optimisations, not strategic choices.

Chance is sampled through a seeded RNG owned by the env. reset(seed)
fully determines the card stream, so replays are exact: reset with the same
seed and the same shuffle is dealt. Duplicate evaluation replays the seed
with the agents' seats swapped (done by the driver, not here).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pyspiel
from pyspiel import gin_rummy as gr

from ginrl import spiel_facts
from ginrl.config import HandConfig, Seeds
from ginrl.env import meld_policy
from ginrl.env.features import EVENT_PHASES, BeliefTracker
from ginrl.env.melds import CardLayout

N_LEARNED_ACTIONS = gr.KNOCK_ACTION + 1  # engine actions 0..55, identity-mapped

# Notified before every engine transition (chance, learned and auto actions)
# with the pre-action state: listener(acting_seat, state, player, action).
# Chance transitions report acting_seat=-1. The arena uses this to route the
# inform_action stream to stateful engine bots (landmine 3).
TransitionListener = Callable[[int, pyspiel.State, int, int], None]

# Phases where the learned policy acts. Everything else (Knock declaration,
# Layoff, Wall) is auto-resolved by meld_policy.
LEARNED_PHASES = frozenset({"FirstUpcard", "Draw", "Discard"})


@dataclass
class StepResult:
    obs: dict[str, object]
    mask: list[bool]  # length 56
    reward: float  # always 0.0; credit assignment uses `returns` at hand end
    done: bool
    seat: int  # seat to act on obs (meaningless when done)
    returns: tuple[float, float] | None  # set at hand end


@dataclass
class HandEnv:
    config: HandConfig = field(default_factory=HandConfig)
    seeds: Seeds = field(default_factory=Seeds)

    def __post_init__(self) -> None:
        self._game = pyspiel.load_game("gin_rummy", self.config.game_params())
        self._state: pyspiel.State | None = None
        self._rng = self.seeds.spawn("hand")
        self._chance_record: list[int] = []
        self._in_deal = True
        self._stock_order: list[int] | None = None
        self._stock_next = 0
        self._trackers = self._new_trackers()
        self.listener: TransitionListener | None = None
        # Seats whose Knock/Layoff phases are driven externally (engine bots
        # playing natively) instead of the meld auto-policy. Wall stays auto.
        self.manual_phases: set[int] = set()

    # -- episode control -------------------------------------------------

    def reset(self, seed: int | None = None) -> StepResult:
        """Start a new hand. Same seed -> same full deck order, bit for bit.

        Duplicate evaluation replays the seed with the agents' seats swapped.
        """
        self._rng = self.seeds.spawn(f"hand:{seed}" if seed is not None else "hand")
        self._chance_record = []
        self._in_deal = True
        self._stock_order: list[int] | None = None
        self._stock_next = 0
        self._trackers = self._new_trackers()
        self._state = self._game.new_initial_state()
        self._advance()
        return self._observe(0.0, None)

    @property
    def trackers(self) -> list[BeliefTracker]:
        return self._trackers

    @property
    def _card_layout(self) -> CardLayout:
        return CardLayout(self.config.num_ranks, self.config.num_suits)

    def _new_trackers(self) -> list[BeliefTracker]:
        layout = self._card_layout
        return [
            BeliefTracker(0, self.config.knock_card, layout, self.config.hand_size),
            BeliefTracker(1, self.config.knock_card, layout, self.config.hand_size),
        ]

    def features_for(self, seat: int) -> list[float]:
        """BeliefTracker feature vector for a seat at the current state."""
        state = self._require_state()
        obs = state.to_observation_struct(seat).to_dict()
        learned = str(state.to_dict()["phase"]) in LEARNED_PHASES
        mask = self.legal_mask() if learned and seat == state.current_player() else None
        return self._trackers[seat].features(obs, mask).tolist()

    def tensor_for(self, seat: int) -> tuple[float, ...]:
        """Padded raw observation tensor for `seat` (actor input)."""
        return tuple(float(x) for x in self._require_state().observation_tensor(seat))

    def hidden_opp_hand(self, seat: int) -> tuple[int, ...]:
        """TRUE opponent hand in layout indices — LABEL NAMESPACE.

        Read from the underlying engine state, which (unlike the observation
        struct) does not mask hidden cards. Never an actor input; used only
        as the supervised target for the belief head and belief metrics.
        """
        hands = self._require_state().to_dict()["hands"][1 - seat]
        layout = self._card_layout
        return tuple(sorted(layout.card_to_index(c) for c in hands if c != "XX"))

    @property
    def chance_record(self) -> list[int]:
        return list(self._chance_record)

    # -- stepping --------------------------------------------------------

    def step(self, action: int) -> StepResult:
        """Apply one learned action (0..55), then auto-resolve and pump chance."""
        state = self._require_state()
        assert not state.is_terminal(), "step called on a finished hand"
        assert state.current_player() >= 0, "step called on a chance node"
        mask = self.legal_mask()
        assert 0 <= action < N_LEARNED_ACTIONS and mask[action], f"illegal learned action {action}"
        self._apply(state, action, learned=True)
        return self._finish_step()

    def _finish_step(self) -> StepResult:
        state = self._require_state()
        self._advance()
        if state.is_terminal():
            returns = tuple(float(r) for r in state.returns())
            assert len(returns) == spiel_facts.NUM_PLAYERS
            return StepResult(
                obs={},
                mask=[False] * N_LEARNED_ACTIONS,
                reward=0.0,
                done=True,
                seat=-1,
                returns=(returns[0], returns[1]),
            )
        return self._observe(0.0, None)

    def legal_mask(self) -> list[bool]:
        """56-bool mask. Engine actions >= 56 never leak into learned decisions."""
        state = self._require_state()
        mask = [False] * N_LEARNED_ACTIONS
        for action in state.legal_actions():
            assert action < N_LEARNED_ACTIONS, (
                f"engine action {action} at learned phase {state.to_dict()['phase']}"
            )
            mask[action] = True
        assert any(mask), "legal mask all-zero at a decision node"
        return mask

    def state(self) -> pyspiel.State:
        return self._require_state()

    # -- internals -------------------------------------------------------

    def _require_state(self) -> pyspiel.State:
        assert self._state is not None, "reset before step"
        return self._state

    def _advance(self) -> None:
        """Drain chance nodes and auto phases until terminal or learned phase."""
        state = self._require_state()
        while True:
            self._pump_chance()
            if state.is_terminal() or state.is_chance_node():
                return
            phase = state.to_dict()["phase"]
            if phase in LEARNED_PHASES:
                return
            if phase in ("Knock", "Layoff") and state.current_player() in self.manual_phases:
                return
            before = len(state.history())
            auto = meld_policy.next_auto_action(state)
            assert auto is not None, f"no auto action at non-learned phase {phase}"
            self._apply(state, auto, learned=False)
            assert len(state.history()) > before, f"auto phase made no progress at {phase}"

    def raw_step(self, action: int) -> StepResult:
        """Apply any legal engine action (bots use this for meld phases)."""
        state = self._require_state()
        assert not state.is_terminal(), "step called on a finished hand"
        assert action in state.legal_actions(), f"illegal engine action {action}"
        self._apply(state, action, learned=False)
        return self._finish_step()

    def _apply(self, state: pyspiel.State, action: int, learned: bool) -> None:
        """Apply one engine action, publishing the public event to both trackers."""
        info = state.to_dict()
        seat = state.current_player()
        upcard = info["upcard"]
        upcard_idx = self._card_layout.card_to_index(upcard) if isinstance(upcard, str) else -1
        if self.listener is not None:
            self.listener(seat, state, seat, action)
        state.apply_action(action)
        self._in_deal = False
        phase = str(info["phase"])
        for tracker in self._trackers:
            self._publish(tracker, seat, phase, action, upcard_idx)
            # Chance never passes through here (only learned/auto moves do),
            # so the logged (phase, seat, action) stream is public information.
            if phase in EVENT_PHASES:
                tracker.observe_event(phase, seat, action)
        if learned:
            for tracker in self._trackers:
                tracker.observe_turn()

    @staticmethod
    def _publish(tracker: BeliefTracker, seat: int, phase: str, action: int, upcard: int) -> None:
        # The takeable card is the `upcard` field; `discard_pile` excludes it
        # (verified against the engine: taking the upcard leaves the pile
        # untouched and delivers the upcard field's card).
        handler = {
            "FirstUpcard": HandEnv._publish_first_upcard,
            "Draw": HandEnv._publish_draw,
            "Discard": HandEnv._publish_discard_like,
            "Knock": HandEnv._publish_discard_like,
            "Layoff": HandEnv._publish_layoff,
        }.get(phase)
        if handler is not None:
            handler(tracker, seat, action, upcard)

    @staticmethod
    def _publish_first_upcard(tracker: BeliefTracker, seat: int, action: int, upcard: int) -> None:
        if action == gr.DRAW_UPCARD_ACTION:
            tracker.observe_take(seat, upcard)
        elif action == gr.PASS_ACTION:
            tracker.observe_decline(seat, upcard)

    @staticmethod
    def _publish_draw(tracker: BeliefTracker, seat: int, action: int, upcard: int) -> None:
        if action == gr.DRAW_UPCARD_ACTION:
            tracker.observe_take(seat, upcard)
        elif action == gr.DRAW_STOCK_ACTION:
            tracker.observe_draw_stock(seat)
            tracker.observe_decline(seat, upcard)

    @staticmethod
    def _publish_discard_like(tracker: BeliefTracker, seat: int, action: int, _upcard: int) -> None:
        if action < gr.DRAW_UPCARD_ACTION:
            tracker.observe_discard(seat, action)

    @staticmethod
    def _publish_layoff(tracker: BeliefTracker, seat: int, action: int, _upcard: int) -> None:
        if action < gr.DRAW_UPCARD_ACTION:
            tracker.observe_layoff(seat, action)

    def _pump_chance(self) -> None:
        state = self._require_state()
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions = [a for a, _ in outcomes]
            if self._in_deal:
                probs = [p for _, p in outcomes]
                action = self._rng.choices(actions, probs)[0]
            else:
                # Stock draw: force the next card of the predetermined order.
                # A uniform permutation revealed draw-by-draw is distributed
                # exactly like sequential uniform draws, and it keeps the
                # whole deck fixed for a seed, so duplicates stay duplicate
                # past the first divergent decision.
                if self._stock_order is None:
                    dealt = set(self._chance_record)
                    rest = [c for c in range(self.config.deck_size) if c not in dealt]
                    self._stock_order = self._rng.sample(rest, len(rest))
                action = self._stock_order[self._stock_next]
                self._stock_next += 1
                assert action in actions, f"predetermined stock card {action} not offered"
            self._chance_record.append(action)
            if self.listener is not None:
                self.listener(-1, state, state.current_player(), action)
            state.apply_action(action)

    def _observe(self, reward: float, returns: tuple[float, float] | None) -> StepResult:
        state = self._require_state()
        seat = state.current_player()
        assert seat in (0, 1)
        # Manual-phase stops (bots driving melds natively) have no learned
        # mask; the driver uses choose_raw and the raw legal list there.
        mask = (
            self.legal_mask()
            if str(state.to_dict()["phase"]) in LEARNED_PHASES
            else [False] * N_LEARNED_ACTIONS
        )
        return StepResult(
            obs=state.to_observation_struct(seat).to_dict(),
            mask=mask,
            reward=reward,
            done=False,
            seat=seat,
            returns=returns,
        )
