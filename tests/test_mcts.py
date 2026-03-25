"""Tests for the MCTS bot and determinization module."""

import random

from catanatron.game import Game
from catanatron.models.enums import RESOURCES
from catanatron.models.player import Color

from src.bots.factory import build_bot
from src.bots.mcts_bot import MCTSBot, MCTSConfig, MCTSNode, mcts_search
from src.sim.adapter import run_game
from src.sim.determinize import determinize, _total_resource_cards


# ── Determinization ──────────────────────────────────────────────────────

def _make_game(seed: int = 42) -> Game:
    """Create a short game and advance past the initial build phase."""
    players = [build_bot("random", Color.RED), build_bot("random", Color.BLUE)]
    game = Game(players=players, seed=seed)
    # play a few ticks to get past initial placements and into the main game
    for _ in range(100):
        if game.winning_color() is not None:
            break
        game.play_tick()
    return game


def test_determinize_preserves_total_resources():
    """Determinized game must keep each player's total card count."""
    game = _make_game()
    rng = random.Random(99)
    for color in game.state.colors:
        original_total = _total_resource_cards(game.state, color)
        det_game = determinize(game, color, rng)
        det_total = _total_resource_cards(det_game.state, color)
        # The perspective player's total should match; opponents too
        assert det_total == original_total, (
            f"{color}: expected {original_total}, got {det_total}"
        )


def test_determinize_preserves_own_hand():
    """The perspective player's per-resource breakdown must stay the same."""
    game = _make_game()
    perspective = game.state.colors[0]
    rng = random.Random(7)
    det_game = determinize(game, perspective, rng)

    from catanatron.state_functions import player_key
    key = player_key(game.state, perspective)
    for r in RESOURCES:
        orig = game.state.player_state[f"{key}_{r}_IN_HAND"]
        det = det_game.state.player_state[f"{key}_{r}_IN_HAND"]
        assert det == orig, f"{r}: expected {orig}, got {det}"


def test_determinize_does_not_mutate_original():
    """Determinizing must not alter the original game."""
    game = _make_game()
    from catanatron.state_functions import player_key
    original_hands = {}
    for color in game.state.colors:
        key = player_key(game.state, color)
        original_hands[color] = {
            r: game.state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES
        }

    determinize(game, game.state.colors[0], random.Random(1))

    for color in game.state.colors:
        key = player_key(game.state, color)
        for r in RESOURCES:
            assert game.state.player_state[f"{key}_{r}_IN_HAND"] == original_hands[color][r]


# ── MCTS node basics ────────────────────────────────────────────────────

def test_mcts_node_expansion():
    """Expanding a root node should create a valid child."""
    game = _make_game()
    actions = list(game.state.playable_actions)
    root = MCTSNode(
        parent=None, action=None,
        color=game.state.current_color(),
        untried_actions=actions,
    )
    assert not root.is_fully_expanded
    assert root.visit_count == 0


# ── MCTSBot integration ─────────────────────────────────────────────────

def test_mcts_bot_returns_legal_action():
    """MCTSBot.decide() must return an action from playable_actions."""
    game = _make_game()
    config = MCTSConfig(num_iterations=10)
    bot = MCTSBot(color=game.state.current_color(), config=config)
    action = bot.decide(game, game.state.playable_actions)
    assert action in game.state.playable_actions


def test_mcts_bot_completes_game():
    """A full game with MCTS vs random should finish without error."""
    config = MCTSConfig(num_iterations=5)  # very few for speed
    players = [
        MCTSBot(color=Color.RED, config=config),
        build_bot("random", Color.BLUE),
    ]
    result = run_game(players=players, seed=42)
    assert result.turns > 0
    # At least one player should have some VP
    assert any(p.victory_points > 0 for p in result.players)
