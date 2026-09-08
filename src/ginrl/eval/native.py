"""Reference native driver: OpenSpiel's own game loop with engine bots.

Used only for parity checks against the arena harness. Fresh bot pair per
game, every action (including chance) informed to the waiting bot — the
protocol landmine 3 demands. Given identical chance actions, a correct
harness must produce identical player actions and returns.
"""

from __future__ import annotations

import pyspiel


def play_native(
    params: dict, chance_actions: list[int]
) -> tuple[tuple[float, float], list[tuple[int, int]]]:
    """Native self-play of two SimpleGinRummyBots through a fixed chance stream.

    Returns (returns, [(seat, action), ...]) over player actions only.
    """
    game = pyspiel.load_game("gin_rummy", params)
    state = game.new_initial_state()
    bots = [
        pyspiel.make_simple_gin_rummy_bot(dict(params), 0),
        pyspiel.make_simple_gin_rummy_bot(dict(params), 1),
    ]
    chance = list(chance_actions)
    player_actions: list[tuple[int, int]] = []
    while not state.is_terminal():
        if state.is_chance_node():
            action = chance.pop(0)
            assert action in [a for a, _ in state.chance_outcomes()]
            for bot in bots:
                bot.inform_action(state, -1, action)
            state.apply_action(action)
        else:
            seat = state.current_player()
            action = bots[seat].step(state)
            for other, bot in enumerate(bots):
                if other != seat:
                    bot.inform_action(state, seat, action)
            player_actions.append((seat, action))
            state.apply_action(action)
    assert not chance, "chance stream longer than the game consumed"
    returns = tuple(float(r) for r in state.returns())
    assert len(returns) == 2
    return returns, player_actions
