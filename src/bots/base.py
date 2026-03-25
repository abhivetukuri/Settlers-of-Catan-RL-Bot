from catanatron.models.player import Player


class BaseBot(Player):
    bot_name = "base"

    def __repr__(self) -> str:
        return f"{self.bot_name}:{self.color.value}"
