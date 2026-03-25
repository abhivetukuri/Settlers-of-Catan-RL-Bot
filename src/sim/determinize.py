"""Determinization utilities for information-set MCTS.

In Catan, the acting player cannot see opponents' resource hands or the
remaining development-card deck order.  A *determinization* creates a
plausible fully-observable clone of the game by randomly redistributing
hidden information while preserving public constraints.
"""

from __future__ import annotations

import random
from typing import List

from catanatron.game import Game
from catanatron.models.enums import RESOURCES
from catanatron.models.player import Color
from catanatron.state_functions import player_key


def _total_resource_cards(state, color: Color) -> int:
    """Return the total number of resource cards held by *color*."""
    key = player_key(state, color)
    return sum(state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES)


def _redistribute_hand(state, color: Color, rng: random.Random) -> None:
    """Replace *color*'s per-resource counts with a random redistribution
    that keeps the same total number of cards.

    This is an in-place mutation of ``state.player_state``.
    """
    key = player_key(state, color)
    total = _total_resource_cards(state, color)
    if total == 0:
        return

    # Zero out current hand
    for r in RESOURCES:
        state.player_state[f"{key}_{r}_IN_HAND"] = 0

    # Randomly assign *total* cards across the five resource types
    for _ in range(total):
        r = rng.choice(RESOURCES)
        state.player_state[f"{key}_{r}_IN_HAND"] += 1


def determinize(game: Game, perspective_color: Color, rng: random.Random | None = None) -> Game:
    """Return a determinized copy of *game* from *perspective_color*'s point
    of view.

    * The perspective player's hand is preserved exactly.
    * Every other player's resource hand is randomly redistributed (same
      total card count, random per-resource split).
    * The development-card deck is reshuffled.

    Parameters
    ----------
    game:
        The current game to determinize.  Not modified.
    perspective_color:
        The color of the player performing the search.  Their hand is
        kept intact.
    rng:
        Optional ``random.Random`` instance for reproducibility.

    Returns
    -------
    Game
        A deep-enough copy safe for simulation.
    """
    if rng is None:
        rng = random.Random()

    game_copy = game.copy()
    state = game_copy.state

    # Randomize opponent hands
    for color in state.colors:
        if color == perspective_color:
            continue
        _redistribute_hand(state, color, rng)

    # Reshuffle the development-card deck
    rng.shuffle(state.development_listdeck)

    return game_copy
