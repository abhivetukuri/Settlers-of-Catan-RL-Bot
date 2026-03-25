from __future__ import annotations

from pathlib import Path

from catanatron.models.player import Color

from src.bots.greedy_bot import GreedyBot
from src.bots.heuristic_bot import HeuristicBot
from src.bots.learned_mcts_bot import LearnedMCTSBot, LearnedMCTSConfig
from src.bots.mcts_bot import MCTSBot, MCTSConfig
from src.bots.random_bot import RandomBot


BOT_REGISTRY = {
    "random": RandomBot,
    "greedy": GreedyBot,
    "heuristic": HeuristicBot,
    "mcts": MCTSBot,
    "learned_mcts": LearnedMCTSBot,
}


def build_bot(
    bot_name: str,
    color: Color,
    *,
    mcts_config: MCTSConfig | None = None,
    learned_mcts_config: LearnedMCTSConfig | None = None,
    checkpoint_path: str | Path | None = None,
):
    if bot_name not in BOT_REGISTRY:
        raise ValueError(f"Unknown bot '{bot_name}'. Choices: {sorted(BOT_REGISTRY.keys())}")
    if bot_name == "mcts":
        return MCTSBot(color=color, is_bot=True, config=mcts_config)
    if bot_name == "learned_mcts":
        return LearnedMCTSBot(
            color=color, is_bot=True,
            config=learned_mcts_config,
            checkpoint_path=checkpoint_path,
        )
    return BOT_REGISTRY[bot_name](color=color, is_bot=True)
