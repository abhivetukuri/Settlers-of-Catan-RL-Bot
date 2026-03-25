"""Monte Carlo Tree Search bot for Settlers of Catan.

Implements Information-Set MCTS (IS-MCTS) via determinization:
each ``decide()`` call samples a determinized game state, then runs
vanilla MCTS (UCB1 selection, random rollout, visit-count final pick).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

from catanatron.game import Game
from catanatron.models.enums import Action
from catanatron.models.player import Color

from src.bots.base import BaseBot
from src.sim.determinize import determinize


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MCTSConfig:
    """Tunable MCTS hyper-parameters."""

    num_iterations: int = 100
    """Number of select-expand-rollout-backprop cycles per decision."""

    exploration_constant: float = 1.41
    """UCB1 exploration constant (√2 ≈ 1.41 is a common default)."""

    max_rollout_turns: int = 500
    """Maximum additional game turns during a single rollout before
    terminating early and evaluating the position heuristically."""


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class MCTSNode:
    """A single node in the MCTS search tree."""

    __slots__ = (
        "parent",
        "action",
        "color",
        "children",
        "untried_actions",
        "visit_count",
        "total_value",
    )

    def __init__(
        self,
        parent: Optional["MCTSNode"],
        action: Optional[Action],
        color: Color,
        untried_actions: List[Action],
    ) -> None:
        self.parent = parent
        self.action = action  # action that led *to* this node
        self.color = color  # color of the player who acts from here
        self.children: List[MCTSNode] = []
        self.untried_actions = list(untried_actions)
        self.visit_count: int = 0
        self.total_value: float = 0.0

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self) -> bool:
        return len(self.untried_actions) == 0 and len(self.children) == 0

    def ucb1(self, exploration_constant: float) -> float:
        """Upper Confidence Bound for Trees."""
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.total_value / self.visit_count
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration

    def best_child(self, exploration_constant: float) -> "MCTSNode":
        """Return the child with the highest UCB1 score."""
        return max(self.children, key=lambda c: c.ucb1(exploration_constant))

    def best_action_child(self) -> "MCTSNode":
        """Return the child with the most visits (robust child)."""
        return max(self.children, key=lambda c: c.visit_count)


# ---------------------------------------------------------------------------
# MCTS logic
# ---------------------------------------------------------------------------

def _select(node: MCTSNode, game: Game, exploration_constant: float) -> MCTSNode:
    """Descend through fully-expanded nodes using UCB1 until reaching a
    node that still has untried actions or is terminal.

    Because we determinize each iteration independently, a child's stored
    action may not be legal in *this* determinization.  We filter children
    to only those whose actions are currently playable; if none are valid
    we stop descending."""
    while node.is_fully_expanded and node.children:
        legal = set(game.state.playable_actions)
        valid_children = [c for c in node.children if c.action in legal]
        if not valid_children:
            break
        node = max(valid_children, key=lambda c: c.ucb1(exploration_constant))
        game.execute(node.action)
    return node


def _expand(node: MCTSNode, game: Game, rng: random.Random) -> MCTSNode | None:
    """Pick a random untried action **that is legal in the current
    determinization**, create a child node, and return it.

    Returns ``None`` if no untried action is legal (the caller should
    fall through directly to rollout)."""
    legal = set(game.state.playable_actions)
    valid_untried = [a for a in node.untried_actions if a in legal]
    if not valid_untried:
        return None

    action = rng.choice(valid_untried)
    node.untried_actions.remove(action)

    game.execute(action)

    child = MCTSNode(
        parent=node,
        action=action,
        color=game.state.current_color(),
        untried_actions=list(game.state.playable_actions),
    )
    node.children.append(child)
    return child


def _rollout(
    game: Game,
    perspective_color: Color,
    vps_to_win: int,
    max_turns: int,
    rng: random.Random,
) -> float:
    """Play out the game with random moves and return a reward in [0, 1].

    The reward is the perspective player's victory points normalised by
    ``vps_to_win``.  A win yields at least 1.0.
    """
    start_turns = game.state.num_turns
    while game.winning_color() is None and (game.state.num_turns - start_turns) < max_turns:
        actions = list(game.state.playable_actions)
        if not actions:
            break
        action = rng.choice(actions)
        game.execute(action)

    # Evaluate from perspective player's viewpoint
    idx = game.state.color_to_index[perspective_color]
    vp = game.state.player_state[f"P{idx}_ACTUAL_VICTORY_POINTS"]
    return min(vp / vps_to_win, 1.0)


def _backpropagate(node: MCTSNode, reward: float) -> None:
    """Walk back up the tree, updating visit counts and values."""
    while node is not None:
        node.visit_count += 1
        node.total_value += reward
        node = node.parent


def mcts_search(
    game: Game,
    perspective_color: Color,
    config: MCTSConfig,
    rng: random.Random | None = None,
) -> Action:
    """Run MCTS from the current game state and return the best action.

    Parameters
    ----------
    game:
        The *real* (non-determinized) game.  Not modified.
    perspective_color:
        Color of the player deciding.
    config:
        MCTS hyper-parameters.
    rng:
        Optional RNG for reproducibility.

    Returns
    -------
    Action
        The action with the highest visit count at the root.
    """
    if rng is None:
        rng = random.Random()

    root_actions = list(game.state.playable_actions)
    if len(root_actions) == 1:
        return root_actions[0]

    root = MCTSNode(
        parent=None,
        action=None,
        color=perspective_color,
        untried_actions=root_actions,
    )

    for _ in range(config.num_iterations):
        # 1. Determinize from the real game each iteration
        sim_game = determinize(game, perspective_color, rng)

        # 2. Select
        node = _select(root, sim_game, config.exploration_constant)

        # 3. Expand (if not terminal)
        if node.untried_actions:
            expanded = _expand(node, sim_game, rng)
            if expanded is not None:
                node = expanded

        # 4. Rollout
        reward = _rollout(
            sim_game,
            perspective_color,
            game.vps_to_win,
            config.max_rollout_turns,
            rng,
        )

        # 5. Backpropagate
        _backpropagate(node, reward)

    return root.best_action_child().action


# ---------------------------------------------------------------------------
# Bot class
# ---------------------------------------------------------------------------

class MCTSBot(BaseBot):
    """MCTS-based Catan player."""

    bot_name = "mcts"

    def __init__(self, color: Color, is_bot: bool = True, config: MCTSConfig | None = None):
        super().__init__(color=color, is_bot=is_bot)
        self.config = config or MCTSConfig()

    def decide(self, game: Game, playable_actions):
        return mcts_search(game, self.color, self.config)
