from __future__ import annotations

from collections import defaultdict

from src.sim.adapter import GameResult


def compute_metrics(game_results: list[GameResult]) -> dict:
    total_games = len(game_results)
    wins = defaultdict(int)
    vp_totals = defaultdict(float)
    turns = []
    for result in game_results:
        turns.append(result.turns)
        for player in result.players:
            bot_key = player.color.value
            vp_totals[bot_key] += player.victory_points
            if player.won:
                wins[bot_key] += 1

    avg_turns = sum(turns) / total_games if total_games else 0.0
    win_rate = {k: v / total_games for k, v in wins.items()} if total_games else {}
    avg_vp = {k: v / total_games for k, v in vp_totals.items()} if total_games else {}
    return {
        "num_games": total_games,
        "avg_turns": avg_turns,
        "win_rate_by_color": win_rate,
        "avg_vp_by_color": avg_vp,
    }
