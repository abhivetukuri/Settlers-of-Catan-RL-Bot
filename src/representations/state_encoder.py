from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncodedState:
    feature_names: list[str]
    features: np.ndarray


def _numeric_player_state(game) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for key, value in game.state.player_state.items():
        if isinstance(value, (int, float, bool)):
            values.append((key, float(value)))
    values.sort(key=lambda x: x[0])
    return values


def encode_state(game) -> EncodedState:
    numeric_items = _numeric_player_state(game)
    feature_names = [k for k, _ in numeric_items]
    feature_values = np.array([v for _, v in numeric_items], dtype=np.float32)
    return EncodedState(feature_names=feature_names, features=feature_values)
