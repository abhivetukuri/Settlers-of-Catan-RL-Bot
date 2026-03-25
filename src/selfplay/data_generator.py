"""Self-play data generator.

Runs games where the current MCTS agent (optionally augmented with a
learned policy/value network) plays against itself, and records
training examples of the form ``(state_tensor, mcts_action_probs,
game_outcome)``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from catanatron.game import Game
from catanatron.models.player import Color

from src.bots.mcts_bot import MCTSConfig, MCTSNode, _select, _expand, _rollout, _backpropagate
from src.models.features import (
    ACTION_TYPE_TO_INDEX,
    NUM_ACTION_TYPES,
    extract_features,
    action_type_index,
)
from src.models.policy_value_net import PolicyValueNet
from src.sim.determinize import determinize


# ── Training example ────────────────────────────────────────────────────

@dataclass
class TrainingExample:
    """A single self-play training example."""
    state: torch.Tensor          # normalised feature vector
    action_probs: torch.Tensor   # MCTS visit-count distribution over ActionTypes
    outcome: float               # 1.0 = win, 0.0 = loss


# ── Replay buffer ──────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size buffer with FIFO eviction."""

    def __init__(self, max_size: int = 50_000) -> None:
        self.max_size = max_size
        self.buffer: list[TrainingExample] = []

    def add(self, examples: list[TrainingExample]) -> None:
        self.buffer.extend(examples)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def sample(self, batch_size: int) -> list[TrainingExample]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ── MCTS with visit-count extraction ───────────────────────────────────

def _mcts_action_probs(
    game: Game,
    perspective_color: Color,
    config: MCTSConfig,
    rng: random.Random,
    temperature: float = 1.0,
) -> tuple:
    """Run MCTS and return ``(chosen_action, action_type_prob_tensor)``.

    ``temperature`` controls exploration:
      * 1.0 → proportional to visit counts (exploratory).
      * 0.0 → greedy (pick the most-visited action).
    """
    root_actions = list(game.state.playable_actions)
    if len(root_actions) == 1:
        probs = torch.zeros(NUM_ACTION_TYPES)
        probs[action_type_index(root_actions[0])] = 1.0
        return root_actions[0], probs

    root = MCTSNode(
        parent=None,
        action=None,
        color=perspective_color,
        untried_actions=root_actions,
    )

    for _ in range(config.num_iterations):
        sim_game = determinize(game, perspective_color, rng)
        node = _select(root, sim_game, config.exploration_constant)
        if node.untried_actions:
            expanded = _expand(node, sim_game, rng)
            if expanded is not None:
                node = expanded
        reward = _rollout(
            sim_game, perspective_color,
            game.vps_to_win, config.max_rollout_turns, rng,
        )
        _backpropagate(node, reward)

    # Aggregate visit counts by ActionType
    type_visits = torch.zeros(NUM_ACTION_TYPES)
    for child in root.children:
        idx = ACTION_TYPE_TO_INDEX[child.action.action_type]
        type_visits[idx] += child.visit_count

    # Apply temperature
    if temperature < 1e-6:
        # Greedy
        best = type_visits.argmax()
        probs = torch.zeros(NUM_ACTION_TYPES)
        probs[best] = 1.0
    else:
        # Proportional (with temperature)
        counts = type_visits ** (1.0 / temperature)
        total = counts.sum()
        probs = counts / total if total > 0 else torch.ones(NUM_ACTION_TYPES) / NUM_ACTION_TYPES

    # Pick the action with the most visits from root children
    best_child = max(root.children, key=lambda c: c.visit_count)
    return best_child.action, probs


# ── Self-play game runner ──────────────────────────────────────────────

@dataclass
class SelfPlayConfig:
    """Configuration for self-play data generation."""
    mcts_config: MCTSConfig = field(default_factory=lambda: MCTSConfig(num_iterations=50))
    num_games: int = 10
    temperature: float = 1.0
    base_seed: int = 0


def generate_self_play_data(
    config: SelfPlayConfig,
    *,
    net: PolicyValueNet | None = None,  # reserved for future integration
) -> list[TrainingExample]:
    """Play ``config.num_games`` self-play games and return training examples.

    Each game pits two MCTS bots against each other.  At every decision
    point we record the state and MCTS visit-count distribution.  After
    the game, outcomes are back-filled (1.0 for the winning colour,
    0.0 for the loser).
    """
    from src.bots.mcts_bot import MCTSBot

    all_examples: list[TrainingExample] = []

    for game_idx in range(config.num_games):
        seed = config.base_seed + game_idx
        rng = random.Random(seed)

        players = [
            MCTSBot(color=Color.RED, config=config.mcts_config),
            MCTSBot(color=Color.BLUE, config=config.mcts_config),
        ]
        game = Game(players=players, seed=seed)

        # Collect per-decision records: (color, state_tensor, action_probs)
        game_records: list[tuple[Color, torch.Tensor, torch.Tensor]] = []

        while game.winning_color() is None:
            current_color = game.state.current_color()
            state_tensor = extract_features(game, current_color)
            action, probs = _mcts_action_probs(
                game, current_color, config.mcts_config, rng, config.temperature,
            )
            game_records.append((current_color, state_tensor, probs))
            game.execute(action)

        winner = game.winning_color()

        # Convert records → training examples with outcome labels
        for color, state_tensor, probs in game_records:
            outcome = 1.0 if color == winner else 0.0
            all_examples.append(TrainingExample(
                state=state_tensor, action_probs=probs, outcome=outcome,
            ))

    return all_examples
