"""MCTS bot guided by a learned policy/value network.

Replaces random rollouts with value-head evaluation and uses the
policy head as a prior in a PUCT-style selection formula, following
the AlphaZero approach.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from catanatron.game import Game
from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Color

from src.bots.base import BaseBot
from src.bots.mcts_bot import MCTSConfig, MCTSNode, _backpropagate
from src.models.features import (
    ACTION_TYPE_TO_INDEX,
    NUM_ACTION_TYPES,
    extract_features,
)
from src.models.policy_value_net import PolicyValueNet
from src.sim.determinize import determinize


# ── Configuration ──────────────────────────────────────────────────────

@dataclass
class LearnedMCTSConfig:
    """Hyper-parameters for the learned MCTS variant."""
    num_iterations: int = 100
    cpuct: float = 1.5
    """PUCT exploration constant (controls prior weight vs visit count)."""
    max_rollout_turns: int = 500
    """Fallback: if no network, use random rollouts up to this many turns."""


# ── PUCT helpers ───────────────────────────────────────────────────────

class PUCTNode:
    """MCTS node augmented with a policy prior for PUCT selection."""

    __slots__ = (
        "parent", "action", "color", "children",
        "untried_actions", "visit_count", "total_value", "prior",
    )

    def __init__(
        self,
        parent: Optional["PUCTNode"],
        action: Optional[Action],
        color: Color,
        untried_actions: list[Action],
        prior: float = 0.0,
    ) -> None:
        self.parent = parent
        self.action = action
        self.color = color
        self.children: list[PUCTNode] = []
        self.untried_actions = list(untried_actions)
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prior = prior

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def puct_score(self, cpuct: float) -> float:
        """PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N)."""
        q = self.total_value / self.visit_count if self.visit_count > 0 else 0.0
        parent_visits = self.parent.visit_count if self.parent else 1
        u = cpuct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q + u

    def best_child_puct(self, cpuct: float) -> "PUCTNode":
        return max(self.children, key=lambda c: c.puct_score(cpuct))

    def best_action_child(self) -> "PUCTNode":
        return max(self.children, key=lambda c: c.visit_count)


# ── Learned MCTS search ───────────────────────────────────────────────

def _get_priors(
    net: PolicyValueNet,
    game: Game,
    perspective_color: Color,
    actions: list[Action],
) -> dict[ActionType, float]:
    """Get policy prior probabilities from the network."""
    state_tensor = extract_features(game, perspective_color)
    legal_mask = torch.zeros(NUM_ACTION_TYPES, dtype=torch.bool)
    for a in actions:
        legal_mask[ACTION_TYPE_TO_INDEX[a.action_type]] = True
    probs, _ = net.predict(state_tensor, legal_mask)
    return {at: probs[idx].item() for at, idx in ACTION_TYPE_TO_INDEX.items()}


def _value_estimate(
    net: PolicyValueNet,
    game: Game,
    perspective_color: Color,
) -> float:
    """Get value estimate from the network."""
    state_tensor = extract_features(game, perspective_color)
    _, value = net.predict(state_tensor)
    return value


def _select_puct(node: PUCTNode, game: Game, cpuct: float) -> PUCTNode:
    """Descend using PUCT until reaching an expandable or terminal node."""
    while node.is_fully_expanded and node.children:
        legal = set(game.state.playable_actions)
        valid_children = [c for c in node.children if c.action in legal]
        if not valid_children:
            break
        node = max(valid_children, key=lambda c: c.puct_score(cpuct))
        game.execute(node.action)
    return node


def _expand_puct(
    node: PUCTNode,
    game: Game,
    rng: random.Random,
    priors: dict[ActionType, float],
) -> PUCTNode | None:
    """Expand one untried action with its policy prior."""
    legal = set(game.state.playable_actions)
    valid_untried = [a for a in node.untried_actions if a in legal]
    if not valid_untried:
        return None

    action = rng.choice(valid_untried)
    node.untried_actions.remove(action)

    game.execute(action)

    child = PUCTNode(
        parent=node,
        action=action,
        color=game.state.current_color(),
        untried_actions=list(game.state.playable_actions),
        prior=priors.get(action.action_type, 1.0 / NUM_ACTION_TYPES),
    )
    node.children.append(child)
    return child


def _backprop_puct(node: PUCTNode, reward: float) -> None:
    """Walk back up updating visit counts and values."""
    while node is not None:
        node.visit_count += 1
        node.total_value += reward
        node = node.parent


def learned_mcts_search(
    game: Game,
    perspective_color: Color,
    net: PolicyValueNet,
    config: LearnedMCTSConfig,
    rng: random.Random | None = None,
) -> Action:
    """Run MCTS guided by the policy/value network.

    * **Selection** uses PUCT with network priors.
    * **Evaluation** uses the value head instead of random rollout.
    """
    if rng is None:
        rng = random.Random()

    root_actions = list(game.state.playable_actions)
    if len(root_actions) == 1:
        return root_actions[0]

    # Get policy priors for the root
    priors = _get_priors(net, game, perspective_color, root_actions)

    root = PUCTNode(
        parent=None,
        action=None,
        color=perspective_color,
        untried_actions=root_actions,
        prior=1.0,
    )

    for _ in range(config.num_iterations):
        sim_game = determinize(game, perspective_color, rng)

        # Select
        node = _select_puct(root, sim_game, config.cpuct)

        # Expand
        if node.untried_actions:
            expanded = _expand_puct(node, sim_game, rng, priors)
            if expanded is not None:
                node = expanded

        # Evaluate with value head (no rollout!)
        reward = _value_estimate(net, sim_game, perspective_color)

        # Backpropagate
        _backprop_puct(node, reward)

    return root.best_action_child().action


# ── Bot class ──────────────────────────────────────────────────────────

class LearnedMCTSBot(BaseBot):
    """MCTS bot guided by a learned policy/value network."""

    bot_name = "learned_mcts"

    def __init__(
        self,
        color: Color,
        is_bot: bool = True,
        config: LearnedMCTSConfig | None = None,
        net: PolicyValueNet | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__(color=color, is_bot=is_bot)
        self.config = config or LearnedMCTSConfig()

        if net is not None:
            self.net = net
        elif checkpoint_path is not None:
            self.net = PolicyValueNet.load(checkpoint_path)
        else:
            # Untrained network — random weights, still usable for testing
            self.net = PolicyValueNet()
            self.net.eval()

    def decide(self, game: Game, playable_actions):
        return learned_mcts_search(game, self.color, self.net, self.config)
