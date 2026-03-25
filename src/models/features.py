"""Enhanced feature extraction for the policy/value network.

Converts a Catanatron ``Game`` state into a normalised ``torch.Tensor``
suitable for neural-network input.  Also provides a mapping from
``ActionType`` → integer index for the policy head.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from catanatron.models.enums import ActionType, RESOURCES
from catanatron.models.player import Color
from catanatron.state_functions import player_key

# ── ActionType → index mapping ──────────────────────────────────────────

_ACTION_TYPES: list[ActionType] = sorted(ActionType, key=lambda a: a.name)
ACTION_TYPE_TO_INDEX: Dict[ActionType, int] = {
    at: i for i, at in enumerate(_ACTION_TYPES)
}
NUM_ACTION_TYPES: int = len(_ACTION_TYPES)

# ── Normalisation constants ─────────────────────────────────────────────
# Chosen so each feature lands roughly in [0, 1].

_NORM: dict[str, float] = {
    "ACTUAL_VICTORY_POINTS": 10.0,
    "VICTORY_POINTS": 10.0,
    "LONGEST_ROAD_LENGTH": 15.0,
    "SETTLEMENTS_AVAILABLE": 5.0,
    "CITIES_AVAILABLE": 4.0,
    "ROADS_AVAILABLE": 15.0,
    # Resource counts
    "BRICK_IN_HAND": 19.0,
    "ORE_IN_HAND": 19.0,
    "SHEEP_IN_HAND": 19.0,
    "WHEAT_IN_HAND": 19.0,
    "WOOD_IN_HAND": 19.0,
    # Dev cards in hand
    "KNIGHT_IN_HAND": 14.0,
    "MONOPOLY_IN_HAND": 2.0,
    "ROAD_BUILDING_IN_HAND": 2.0,
    "YEAR_OF_PLENTY_IN_HAND": 2.0,
    "VICTORY_POINT_IN_HAND": 5.0,
    # Played dev cards
    "PLAYED_KNIGHT": 14.0,
    "PLAYED_MONOPOLY": 2.0,
    "PLAYED_ROAD_BUILDING": 2.0,
    "PLAYED_YEAR_OF_PLENTY": 2.0,
    "PLAYED_VICTORY_POINT": 5.0,
}


def _norm_for_key(raw_key: str) -> float:
    """Return the normalisation divisor for a player-state key."""
    # raw_key looks like "P0_BRICK_IN_HAND" — strip the prefix
    suffix = raw_key.split("_", 1)[1] if "_" in raw_key else raw_key
    return _NORM.get(suffix, 1.0)


# ── Public API ──────────────────────────────────────────────────────────

def extract_features(game, perspective_color: Color) -> torch.Tensor:
    """Return a 1-D float tensor of normalised game-state features.

    The features are ordered so that the *perspective player*'s features
    come first, followed by each opponent (in the order they appear in
    ``game.state.colors``).  This canonical ordering lets the network
    learn position-agnostic representations.

    Parameters
    ----------
    game:
        A Catanatron ``Game`` instance.
    perspective_color:
        The color of the acting player.  Their features are placed in
        the first block.

    Returns
    -------
    torch.Tensor
        ``float32`` 1-D tensor of length ``state_dim()``.
    """
    state = game.state

    # Build ordered color list: perspective player first
    ordered_colors = [perspective_color] + [
        c for c in state.colors if c != perspective_color
    ]

    features: list[float] = []
    for color in ordered_colors:
        key = player_key(state, color)
        for raw_key in sorted(state.player_state.keys()):
            if not raw_key.startswith(f"{key}_"):
                continue
            val = state.player_state[raw_key]
            if isinstance(val, (int, float, bool)):
                features.append(float(val) / _norm_for_key(raw_key))

    return torch.tensor(features, dtype=torch.float32)


def state_dim(num_players: int = 2) -> int:
    """Return the expected feature-vector length for *num_players* players.

    In a 2-player game the Catanatron player_state dict contains 25
    numeric keys per player → 50 total.
    """
    return 25 * num_players


def action_type_index(action) -> int:
    """Map an ``Action`` to its policy-head index."""
    return ACTION_TYPE_TO_INDEX[action.action_type]
