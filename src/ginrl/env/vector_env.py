"""Batched stepping across N independent hands in one process.

Inference is batched; multi-process actors come later, only if Phase 5's
profile says so. Each sub-env owns its HandEnv (and BeliefTrackers); step()
takes one learned action per env. Finished envs report done with their
returns and must be reset individually before stepping again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ginrl.config import HandConfig, Seeds
from ginrl.env.features import feature_dim
from ginrl.env.game import N_LEARNED_ACTIONS, HandEnv, StepResult


@dataclass
class BatchResult:
    features: list[list[float]]  # per env; [] for done envs
    masks: list[list[bool]]
    seats: list[int]
    dones: list[bool]
    returns: list[tuple[float, float] | None]


@dataclass
class VectorEnv:
    n_envs: int
    config: HandConfig = field(default_factory=HandConfig)
    seeds: Seeds = field(default_factory=Seeds)

    def __post_init__(self) -> None:
        self._envs = [HandEnv(self.config, self.seeds) for _ in range(self.n_envs)]
        self._counts = [0] * self.n_envs

    def reset(self, env_id: int, seed: int | None = None) -> StepResult:
        self._counts[env_id] += 1
        return self._envs[env_id].reset(seed=seed)

    def reset_all(self, base_seed: int = 0) -> BatchResult:
        return self._batch([self.reset(i, seed=base_seed + i) for i in range(self.n_envs)])

    def step(self, actions: list[int | None]) -> BatchResult:
        """actions[i] is None for envs expected to be done (skipped)."""
        assert len(actions) == self.n_envs
        results: list[StepResult | None] = []
        for i, action in enumerate(actions):
            if action is None:
                results.append(None)
            else:
                results.append(self._envs[i].step(action))
        return self._batch(results)

    def features_for(self, env_id: int, seat: int) -> list[float]:
        return self._envs[env_id].features_for(seat)

    def tensor_for(self, env_id: int, seat: int) -> tuple[float, ...]:
        """Padded raw observation tensor (actor input)."""
        return self._envs[env_id].tensor_for(seat)

    def events_for(self, env_id: int, seat: int) -> tuple[int, ...]:
        """Bounded public-event window for `seat`'s tracker (actor input)."""
        return self._envs[env_id].trackers[seat].event_window()

    def hidden_opp_hand(self, env_id: int, seat: int) -> tuple[int, ...]:
        """TRUE opponent hand — LABEL NAMESPACE, never an actor input."""
        return self._envs[env_id].hidden_opp_hand(seat)

    def _batch(self, results: list[StepResult | None]) -> BatchResult:
        feats: list[list[float]] = []
        masks: list[list[bool]] = []
        seats: list[int] = []
        dones: list[bool] = []
        returns: list[tuple[float, float] | None] = []
        for i, result in enumerate(results):
            if result is None or result.done:
                feats.append([])
                masks.append([False] * N_LEARNED_ACTIONS)
                seats.append(-1)
                dones.append(True)
                returns.append(result.returns if result is not None else None)
            else:
                feats.append(self._envs[i].features_for(result.seat))
                assert len(feats[-1]) == feature_dim(self.config.deck_size)
                masks.append(result.mask)
                seats.append(result.seat)
                dones.append(False)
                returns.append(None)
        return BatchResult(features=feats, masks=masks, seats=seats, dones=dones, returns=returns)
