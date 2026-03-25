from dataclasses import dataclass
from typing import Iterable

from catanatron.game import Game
from catanatron.models.player import Color, Player
from catanatron.state_functions import get_actual_victory_points

from src.sim.determinism import seed_all


@dataclass(frozen=True)
class PlayerResult:
    color: Color
    victory_points: int
    won: bool


@dataclass(frozen=True)
class GameResult:
    seed: int
    turns: int
    winner: Color | None
    players: list[PlayerResult]


def run_game(
    players: Iterable[Player],
    seed: int,
    vps_to_win: int = 10,
    discard_limit: int = 7,
) -> GameResult:
    seed_all(seed)
    game = Game(
        players=list(players),
        seed=seed,
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
    )
    winner = game.play()
    player_results: list[PlayerResult] = []
    for color in game.state.colors:
        vp = int(get_actual_victory_points(game.state, color))
        player_results.append(PlayerResult(color=color, victory_points=vp, won=color == winner))
    return GameResult(seed=seed, turns=game.state.num_turns, winner=winner, players=player_results)
