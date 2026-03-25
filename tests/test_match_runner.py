from src.config import MatchConfig
from src.eval.match_runner import run_match_series


def test_match_runner_smoke():
    config = MatchConfig(num_games=2, base_seed=11)
    result = run_match_series("random", "heuristic", config)
    assert len(result.game_results) == 2
    assert result.summary["num_games"] == 2
    assert result.representation_snapshot.state_feature_dim > 0
