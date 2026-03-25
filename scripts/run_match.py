from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from src.config import MatchConfig
from src.eval.match_runner import run_match_series
from src.logging_utils import run_directory, write_game_results_jsonl, write_summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seeded Catanatron baseline bot matches.")
    parser.add_argument("--bot-a", choices=["random", "greedy", "heuristic"], required=True)
    parser.add_argument("--bot-b", choices=["random", "greedy", "heuristic"], required=True)
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MatchConfig(
        num_games=args.num_games,
        base_seed=args.seed,
        artifacts_dir=Path(args.artifacts_dir),
    )
    output = run_match_series(args.bot_a, args.bot_b, config)

    out_dir = run_directory(config.artifacts_dir)
    write_game_results_jsonl(out_dir / "games.jsonl", output.game_results)
    summary = dict(output.summary)
    summary["representation_snapshot"] = asdict(output.representation_snapshot)
    write_summary_json(out_dir / "summary.json", summary)

    print(f"Saved artifacts to: {out_dir}")
    print(summary)


if __name__ == "__main__":
    main()
