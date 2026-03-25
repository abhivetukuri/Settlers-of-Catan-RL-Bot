from catanatron.game import Game
from catanatron.models.player import Color

from src.bots.factory import build_bot
from src.representations.action_encoder import encode_legal_actions


def test_legal_action_encoding_is_stable():
    players = [build_bot("random", Color.RED), build_bot("greedy", Color.BLUE)]
    game = Game(players=players, seed=7)
    first = encode_legal_actions(game.state.playable_actions)
    second = encode_legal_actions(game.state.playable_actions)
    assert first.signatures == second.signatures
    assert first.action_to_index == second.action_to_index
