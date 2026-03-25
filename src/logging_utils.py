from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, UTC
from pathlib import Path

from src.sim.adapter import GameResult


def run_directory(artifacts_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = artifacts_dir / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _serialize_game_result(result: GameResult) -> dict:
    payload = asdict(result)
    payload["winner"] = result.winner.value if result.winner is not None else None
    for player in payload["players"]:
        player["color"] = player["color"].value
    return payload


def write_game_results_jsonl(output_path: Path, game_results: list[GameResult]) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for result in game_results:
            f.write(json.dumps(_serialize_game_result(result)))
            f.write("\n")


def write_summary_json(output_path: Path, summary: dict) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
