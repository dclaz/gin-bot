"""Vectorised self-play driver for small OpenSpiel games (Kuhn, Leduc).

Both seats share one policy; seats alternate naturally as the game tree
dictates. Each sub-env owns a game state plus a dedicated chance RNG, so
`reset_all(base_seed)` fully determines every card stream and reruns are
bit-identical. Chance nodes are auto-sampled; the learner only sees decision
nodes with (infostate tensor, legal-action mask, acting seat).

Rewards are sparse: 0 on every non-terminal step, `returns()[seat]` at hand
end. A `Rollout` is one row per decision, ready for GAE/Monte-Carlo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyspiel

from ginrl.config import Seeds
from ginrl.nets.actor_critic import NetConfig

# Opponent-private-card head dims (deck sizes) for the aux belief target.
DECK_SIZES = {"kuhn_poker": 3, "leduc_poker": 6}


@dataclass(frozen=True)
class RolloutStep:
    """One decision node as the learner consumes it."""

    table: int  # sub-env id; groups rows into per-episode streams for GAE
    obs: tuple[float, ...]
    mask: tuple[bool, ...]
    action: int
    reward: float
    done: bool
    seat: int
    opp_card: int  # opponent private card index (aux belief label)


@dataclass
class Rollout:
    """One row per decision, concatenated over envs in env-id order."""

    steps: list[RolloutStep] = field(default_factory=list)


def net_config_for_game(name: str, hidden: int = 128, layers: int = 2) -> NetConfig:
    """Shape a MaskedActorCritic for a small game."""
    game = pyspiel.load_game(name)
    state = game.new_initial_state()
    obs_dim = len(state.information_state_tensor(0))
    return NetConfig(
        obs_dim=obs_dim,
        n_actions=game.num_distinct_actions(),
        aux_dim=DECK_SIZES[name],
        hidden=hidden,
        layers=layers,
    )


class SmallGameVecEnv:
    """N parallel small-game tables sharing one policy in self-play."""

    def __init__(self, game_name: str, n_envs: int, seeds: Seeds | None = None) -> None:
        if game_name not in DECK_SIZES:
            raise ValueError(f"driver supports {sorted(DECK_SIZES)}, got {game_name!r}")
        self.game_name = game_name
        self.n_envs = n_envs
        self.seeds = seeds if seeds is not None else Seeds()
        self._game = pyspiel.load_game(game_name)
        self._n_actions = self._game.num_distinct_actions()
        self._states: list[pyspiel.State] = []
        self._rngs = []
        self._dealt: list[list[int]] = []
        self._pending_returns: list[tuple[float, float] | None] = [None] * n_envs

    def reset_all(self, base_seed: int = 0) -> None:
        """Deal every table; table i is fully determined by base_seed + i."""
        self._states = []
        self._rngs = []
        self._dealt = []
        self._pending_returns = [None] * self.n_envs
        for i in range(self.n_envs):
            self._rngs.append(self.seeds.spawn(f"{self.game_name}-{base_seed + i}"))
            self._states.append(self._game.new_initial_state())
            self._dealt.append([])
            self._pump_chance(i)

    def _pump_chance(self, i: int) -> None:
        state = self._states[i]
        while state.is_chance_node():
            pairs = state.chance_outcomes()
            outcomes = [a for a, _ in pairs]
            probs = [p for _, p in pairs]
            roll = self._rngs[i].random()
            cumulative, chosen = 0.0, outcomes[-1]
            for action, prob in zip(outcomes, probs, strict=True):
                cumulative += prob
                if roll < cumulative:
                    chosen = action
                    break
            self._dealt[i].append(chosen)
            state.apply_action(chosen)

    def observe(self, i: int) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        """Current (obs, mask) for table i. Only valid on decision nodes."""
        obs, mask, _ = self._obs_for(i)
        return obs, mask

    def acting_seat(self, i: int) -> int | None:
        """Seat to act on table i, or None if the hand just ended."""
        state = self._states[i]
        if state.is_terminal():
            return None
        seat = state.current_player()
        return seat if seat >= 0 else None

    def _obs_for(self, i: int) -> tuple[tuple[float, ...], tuple[bool, ...], int]:
        state = self._states[i]
        seat = state.current_player()
        assert seat >= 0, "only call on decision nodes"
        obs = tuple(float(x) for x in state.information_state_tensor(seat))
        legal = set(state.legal_actions())
        mask = tuple(a in legal for a in range(self._n_actions))
        return obs, mask, seat

    def act(self, actions: list[int | None]) -> Rollout:
        """Step every table with the policy's actions (None = table done, skip).

        Returns this round's decision rows. Tables that reach terminal report
        one row with done=True carrying the acting seat's return; the table
        must be re-dealt with `reset_table` before stepping again.
        """
        assert len(actions) == self.n_envs
        rollout = Rollout()
        for i, action in enumerate(actions):
            if action is None:
                continue
            state = self._states[i]
            obs, mask, seat = self._obs_for(i)
            assert mask[action], f"illegal action {action} on {self.game_name}"
            opp = 1 - seat
            opp_card = self._dealt[i][opp] if opp < len(self._dealt[i]) else 0
            state.apply_action(action)
            self._pump_chance(i)
            if state.is_terminal():
                returns = state.returns()
                rollout.steps.append(
                    RolloutStep(i, obs, mask, action, float(returns[seat]), True, seat, opp_card)
                )
                self._pending_returns[i] = (float(returns[0]), float(returns[1]))
            else:
                rollout.steps.append(RolloutStep(i, obs, mask, action, 0.0, False, seat, opp_card))
        return rollout

    def reset_table(self, i: int, seed: int) -> None:
        """Re-deal a finished table under a new seed."""
        self._rngs[i] = self.seeds.spawn(f"{self.game_name}-{seed}")
        self._states[i] = self._game.new_initial_state()
        self._dealt[i] = []
        self._pending_returns[i] = None
        self._pump_chance(i)

    def pending_returns(self, i: int) -> tuple[float, float] | None:
        """Terminal returns waiting on table i (None while the hand runs)."""
        return self._pending_returns[i]
