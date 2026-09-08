"""Engine-bot agents: SimpleGinRummyBot and ISMCTS behind the Agent interface.

Landmine 3: these bots are stateful. A fresh bot is constructed per game in
begin_game (never reused across games, never restart_at), and every engine
transition — opponent actions, chance, and our own auto-resolved meld/layoff
actions — is routed to the waiting bot via inform_action through the env's
transition listener, which the duel in ginrl.eval.arena installs.
"""

from __future__ import annotations

import pyspiel

from ginrl.config import HandConfig
from ginrl.env.game import N_LEARNED_ACTIONS, HandEnv


class SimpleGinRummyAgent:
    """Reference opponent. Almost certainly the EAAI reference player port."""

    manual_phases = True

    def __init__(self, name: str = "simple_bot", config: HandConfig | None = None) -> None:
        self.name = name
        self._params = (config or HandConfig()).game_params()
        self._seat = 0
        self._bot: pyspiel.Bot | None = None

    def begin_game(self, seat: int) -> None:
        self._seat = seat
        self._bot = pyspiel.make_simple_gin_rummy_bot(dict(self._params), seat)

    def _step(self, state: pyspiel.State) -> int:
        assert self._bot is not None, "begin_game before choose"
        action = self._bot.step(state)
        assert action in state.legal_actions(), f"engine bot returned illegal action {action}"
        return action

    def choose(self, env: HandEnv, seat: int) -> int:
        assert seat == self._seat
        action = self._step(env.state())
        assert 0 <= action < N_LEARNED_ACTIONS, (
            f"engine bot returned non-learned action {action} at a learned phase"
        )
        assert env.legal_mask()[action], f"engine bot returned illegal action {action}"
        return action

    def choose_raw(self, state: pyspiel.State) -> int:
        return self._step(state)

    def inform(self, state: pyspiel.State, player: int, action: int) -> None:
        if self._bot is not None:
            self._bot.inform_action(state, player, action)


class ISMCTSAgent:
    """Determinized-search baseline. Budget (simulations) is the strength dial."""

    manual_phases = True

    def __init__(
        self,
        simulations: int = 100,
        *,
        seed: int = 0,
        uct_c: float = 1.0,
        rollout_count: int = 10,
        name: str | None = None,
    ) -> None:
        self.simulations = simulations
        self._seed = seed
        self._uct_c = uct_c
        self._rollout_count = rollout_count
        self.name = name or f"ismcts({simulations})"
        self._seat = 0
        self._bot: pyspiel.Bot | None = None
        self._games = 0

    def begin_game(self, seat: int) -> None:
        self._seat = seat
        evaluator = pyspiel.RandomRolloutEvaluator(self._rollout_count, self._seed + self._games)
        self._games += 1
        self._bot = pyspiel.ISMCTSBot(
            self._seed + self._games,
            evaluator,
            self._uct_c,
            self.simulations,
            -1,
            pyspiel.ISMCTSFinalPolicyType.NORMALIZED_VISIT_COUNT,
            False,
            False,
            -1.0,
        )

    def _step(self, state: pyspiel.State) -> int:
        assert self._bot is not None, "begin_game before choose"
        action = self._bot.step(state)
        assert action in state.legal_actions(), f"ISMCTS returned illegal action {action}"
        return action

    def choose(self, env: HandEnv, seat: int) -> int:
        assert seat == self._seat
        action = self._step(env.state())
        assert 0 <= action < N_LEARNED_ACTIONS, (
            f"ISMCTS returned non-learned action {action} at a learned phase"
        )
        assert env.legal_mask()[action], f"ISMCTS returned illegal action {action}"
        return action

    def choose_raw(self, state: pyspiel.State) -> int:
        return self._step(state)

    def inform(self, state: pyspiel.State, player: int, action: int) -> None:
        if self._bot is not None:
            self._bot.inform_action(state, player, action)
