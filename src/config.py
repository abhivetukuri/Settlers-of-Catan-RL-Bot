from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MatchConfig:
    num_games: int = 20
    base_seed: int = 42
    vps_to_win: int = 10
    discard_limit: int = 7
    artifacts_dir: Path = Path("artifacts")
