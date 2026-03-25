from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from src.bots.mcts_bot import MCTSConfig
from src.config import MatchConfig
from src.eval.match_runner import run_match_series
from src.logging_utils import run_directory, write_game_results_jsonl, write_summary_json

BOT_CHOICES = ["random", "greedy", "heuristic", "mcts", "learned_mcts"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seeded Catanatron bot matches.")
    parser.add_argument("--bot-a", choices=BOT_CHOICES, required=True)
    parser.add_argument("--bot-b", choices=BOT_CHOICES, required=True)
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--mcts-iterations", type=int, default=100,
                        help="Number of MCTS iterations per decision (default: 100)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint for learned_mcts bot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mcts_config = MCTSConfig(num_iterations=args.mcts_iterations)
    config = MatchConfig(
        num_games=args.num_games,
        base_seed=args.seed,
        artifacts_dir=Path(args.artifacts_dir),
    )
    output = run_match_series(
        args.bot_a, args.bot_b, config,
        mcts_config=mcts_config,
        checkpoint_path=args.checkpoint,
    )

    out_dir = run_directory(config.artifacts_dir)
    write_game_results_jsonl(out_dir / "games.jsonl", output.game_results)
    summary = dict(output.summary)
    summary["representation_snapshot"] = asdict(output.representation_snapshot)
    write_summary_json(out_dir / "summary.json", summary)

    print(f"Saved artifacts to: {out_dir}")
    print(summary)


if __name__ == "__main__":
    main()
