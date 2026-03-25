from catanatron.models.actions import ActionType

from src.bots.base import BaseBot


HEURISTIC_SCORES = {
    ActionType.BUILD_CITY: 1200,
    ActionType.BUILD_SETTLEMENT: 900,
    ActionType.BUILD_ROAD: 220,
    ActionType.BUY_DEVELOPMENT_CARD: 250,
    ActionType.PLAY_MONOPOLY: 260,
    ActionType.PLAY_YEAR_OF_PLENTY: 240,
    ActionType.PLAY_ROAD_BUILDING: 220,
    ActionType.PLAY_KNIGHT_CARD: 200,
    ActionType.MARITIME_TRADE: 120,
    ActionType.MOVE_ROBBER: 80,
    ActionType.ROLL: 10,
    ActionType.DISCARD: -10,
    ActionType.END_TURN: -100,
}


class HeuristicBot(BaseBot):
    bot_name = "heuristic"

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        my_vp = game.state.player_state[f"P{game.state.color_to_index[self.color]}_ACTUAL_VICTORY_POINTS"]
        adjusted_scores = []
        for action in actions:
            score = HEURISTIC_SCORES.get(action.action_type, 0)
            if my_vp >= 8 and action.action_type in (ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT):
                score += 500
            adjusted_scores.append((score, action))
        adjusted_scores.sort(key=lambda x: x[0], reverse=True)
        return adjusted_scores[0][1]
