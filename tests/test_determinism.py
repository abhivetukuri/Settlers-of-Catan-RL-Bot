from catanatron.models.player import Color

from src.bots.factory import build_bot
from src.sim.adapter import run_game


def test_same_seed_same_outcome():
    players_a = [build_bot("random", Color.RED), build_bot("random", Color.BLUE)]
    players_b = [build_bot("random", Color.RED), build_bot("random", Color.BLUE)]
    first = run_game(players_a, seed=123)
    second = run_game(players_b, seed=123)
    assert first.winner == second.winner
    assert first.turns == second.turns
    assert [(p.color, p.victory_points) for p in first.players] == [
        (p.color, p.victory_points) for p in second.players
    ]
