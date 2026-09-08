"""Runner-vs-native parity: identical chance streams must give identical play."""

from __future__ import annotations

from ginrl.agents.spiel_bots import SimpleGinRummyAgent
from ginrl.config import HandConfig, Seeds
from ginrl.env.game import HandEnv
from ginrl.eval import arena as arena_mod
from ginrl.eval.native import play_native

PARAMS = HandConfig().game_params()


def _runner(seed: int):  # type: ignore[no-untyped-def]
    env = HandEnv(HandConfig(), Seeds())
    g = arena_mod.play_game(SimpleGinRummyAgent(), SimpleGinRummyAgent(), seed, env)
    return g.returns, [(r.seat, r.action) for r in g.actions], list(env.chance_record)


def test_runner_matches_native_play_by_play() -> None:
    for seed in range(2000, 2010):
        returns_r, actions_r, chance = _runner(seed)
        returns_n, actions_n = play_native(PARAMS, chance)
        assert returns_r == returns_n, f"seed {seed}: {returns_r} != {returns_n}"
        assert actions_r == actions_n, f"seed {seed}: action streams diverge"
