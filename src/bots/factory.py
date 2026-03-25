from catanatron.models.player import Color

from src.bots.greedy_bot import GreedyBot
from src.bots.heuristic_bot import HeuristicBot
from src.bots.random_bot import RandomBot


BOT_REGISTRY = {
    "random": RandomBot,
    "greedy": GreedyBot,
    "heuristic": HeuristicBot,
}


def build_bot(bot_name: str, color: Color):
    if bot_name not in BOT_REGISTRY:
        raise ValueError(f"Unknown bot '{bot_name}'. Choices: {sorted(BOT_REGISTRY.keys())}")
    return BOT_REGISTRY[bot_name](color=color, is_bot=True)
