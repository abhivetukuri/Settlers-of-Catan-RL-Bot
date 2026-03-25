from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from catanatron.game import Game
from catanatron.models.player import Color

from src.bots.factory import build_bot
from src.bots.mcts_bot import MCTSConfig
from src.config import MatchConfig
from src.eval.metrics import compute_metrics
from src.representations.action_encoder import encode_legal_actions
from src.representations.state_encoder import encode_state
from src.sim.adapter import GameResult, run_game


@dataclass(frozen=True)
class RepresentationSnapshot:
    state_feature_names: list[str]
    state_feature_dim: int
    legal_action_signatures: list[str]


@dataclass(frozen=True)
class MatchRunOutput:
    game_results: list[GameResult]
    summary: dict
    representation_snapshot: RepresentationSnapshot


def _default_colors() -> tuple[Color, Color]:
    return (Color.RED, Color.BLUE)


def capture_representation_snapshot(bot_a: str, bot_b: str, seed: int) -> RepresentationSnapshot:
    red, blue = _default_colors()
    players = [build_bot(bot_a, red), build_bot(bot_b, blue)]
    game = Game(players=players, seed=seed)
    encoded_state = encode_state(game)
    encoded_actions = encode_legal_actions(game.state.playable_actions)
    return RepresentationSnapshot(
        state_feature_names=encoded_state.feature_names,
        state_feature_dim=int(encoded_state.features.shape[0]),
        legal_action_signatures=encoded_actions.signatures,
    )


def run_match_series(
    bot_a: str,
    bot_b: str,
    config: MatchConfig,
    *,
    mcts_config: MCTSConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> MatchRunOutput:
    red, blue = _default_colors()
    game_results: list[GameResult] = []
    for game_idx in range(config.num_games):
        seed = config.base_seed + game_idx
        players = [
            build_bot(bot_a, red, mcts_config=mcts_config, checkpoint_path=checkpoint_path),
            build_bot(bot_b, blue, mcts_config=mcts_config, checkpoint_path=checkpoint_path),
        ]
        result = run_game(
            players=players,
            seed=seed,
            vps_to_win=config.vps_to_win,
            discard_limit=config.discard_limit,
        )
        game_results.append(result)

    summary = compute_metrics(game_results)
    summary["bot_mapping"] = {red.value: bot_a, blue.value: bot_b}
    snapshot = capture_representation_snapshot(bot_a, bot_b, config.base_seed)
    return MatchRunOutput(game_results=game_results, summary=summary, representation_snapshot=snapshot)
