import random

from src.bots.base import BaseBot


class RandomBot(BaseBot):
    bot_name = "random"

    def decide(self, game, playable_actions):
        return random.choice(list(playable_actions))
