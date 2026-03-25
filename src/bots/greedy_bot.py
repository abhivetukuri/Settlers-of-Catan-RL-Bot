from catanatron.models.actions import ActionType

from src.bots.base import BaseBot


GREEDY_SCORES = {
    ActionType.BUILD_CITY: 1000,
    ActionType.BUILD_SETTLEMENT: 800,
    ActionType.BUY_DEVELOPMENT_CARD: 220,
    ActionType.PLAY_MONOPOLY: 210,
    ActionType.PLAY_YEAR_OF_PLENTY: 180,
    ActionType.PLAY_ROAD_BUILDING: 170,
    ActionType.PLAY_KNIGHT_CARD: 140,
    ActionType.BUILD_ROAD: 120,
    ActionType.MOVE_ROBBER: 75,
    ActionType.MARITIME_TRADE: 40,
    ActionType.ROLL: 20,
    ActionType.DISCARD: -20,
    ActionType.END_TURN: -50,
}


class GreedyBot(BaseBot):
    bot_name = "greedy"

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        best_action = max(actions, key=lambda action: GREEDY_SCORES.get(action.action_type, 0))
        return best_action
