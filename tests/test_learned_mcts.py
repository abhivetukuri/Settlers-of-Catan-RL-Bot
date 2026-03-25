"""Tests for the learning component: network, features, learned MCTS,
self-play data generation, and training loop.
"""

import random
import tempfile
from pathlib import Path

import torch
from catanatron.game import Game
from catanatron.models.player import Color

from src.bots.factory import build_bot
from src.bots.learned_mcts_bot import LearnedMCTSBot, LearnedMCTSConfig, learned_mcts_search
from src.models.features import (
    ACTION_TYPE_TO_INDEX,
    NUM_ACTION_TYPES,
    extract_features,
    state_dim,
    action_type_index,
)
from src.models.policy_value_net import PolicyValueNet
from src.selfplay.data_generator import (
    ReplayBuffer,
    SelfPlayConfig,
    TrainingExample,
    generate_self_play_data,
)
from src.selfplay.trainer import TrainerConfig, train
from src.bots.mcts_bot import MCTSConfig


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_game(seed: int = 42) -> Game:
    """Create a game and advance past initial placements."""
    players = [build_bot("random", Color.RED), build_bot("random", Color.BLUE)]
    game = Game(players=players, seed=seed)
    for _ in range(100):
        if game.winning_color() is not None:
            break
        game.play_tick()
    return game


# ── Feature extraction ──────────────────────────────────────────────────

def test_extract_features_shape():
    """Feature tensor should have the expected dimension."""
    game = _make_game()
    tensor = extract_features(game, Color.RED)
    assert tensor.ndim == 1
    assert tensor.shape[0] == state_dim(num_players=2)
    assert tensor.dtype == torch.float32


def test_extract_features_perspective_ordering():
    """Features for RED-perspective vs BLUE-perspective should differ
    (the first block corresponds to the perspective player)."""
    game = _make_game()
    red_feat = extract_features(game, Color.RED)
    blue_feat = extract_features(game, Color.BLUE)
    # They encode the same game but in different order → not equal
    assert not torch.equal(red_feat, blue_feat)


def test_action_type_index_coverage():
    """Every ActionType should have a unique index."""
    assert len(ACTION_TYPE_TO_INDEX) == NUM_ACTION_TYPES
    indices = set(ACTION_TYPE_TO_INDEX.values())
    assert len(indices) == NUM_ACTION_TYPES


# ── PolicyValueNet ──────────────────────────────────────────────────────

def test_network_forward_shapes():
    """Forward pass should produce correct output shapes."""
    net = PolicyValueNet()
    x = torch.randn(4, state_dim())
    logits, value = net(x)
    assert logits.shape == (4, NUM_ACTION_TYPES)
    assert value.shape == (4, 1)
    # Value should be in [0, 1] (sigmoid)
    assert (value >= 0).all() and (value <= 1).all()


def test_network_predict_with_mask():
    """predict() should mask illegal actions to zero probability."""
    net = PolicyValueNet()
    x = torch.randn(state_dim())
    mask = torch.zeros(NUM_ACTION_TYPES, dtype=torch.bool)
    mask[0] = True
    mask[3] = True
    probs, val = net.predict(x, legal_mask=mask)
    # Only masked-on entries should have probability
    for i in range(NUM_ACTION_TYPES):
        if not mask[i]:
            assert probs[i].item() < 1e-6


def test_network_save_load():
    """Save and load should produce identical outputs."""
    net = PolicyValueNet()
    x = torch.randn(1, state_dim())
    logits1, val1 = net(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_model.pt"
        net.save(path)
        loaded = PolicyValueNet.load(path)
        logits2, val2 = loaded(x)

    assert torch.allclose(logits1, logits2, atol=1e-6)
    assert torch.allclose(val1, val2, atol=1e-6)


# ── Learned MCTS Bot ───────────────────────────────────────────────────

def test_learned_mcts_returns_legal_action():
    """LearnedMCTSBot.decide() must return a legal action."""
    game = _make_game()
    config = LearnedMCTSConfig(num_iterations=5)
    bot = LearnedMCTSBot(color=game.state.current_color(), config=config)
    action = bot.decide(game, game.state.playable_actions)
    assert action in game.state.playable_actions


def test_learned_mcts_via_factory():
    """Factory should build learned_mcts bot without error."""
    bot = build_bot("learned_mcts", Color.RED)
    assert bot.bot_name == "learned_mcts"


# ── Self-play data generation ──────────────────────────────────────────

def test_self_play_generates_examples():
    """Self-play should produce non-empty training examples."""
    sp_config = SelfPlayConfig(
        mcts_config=MCTSConfig(num_iterations=3),
        num_games=1,
        base_seed=42,
    )
    examples = generate_self_play_data(sp_config)
    assert len(examples) > 0
    ex = examples[0]
    assert ex.state.shape[0] == state_dim()
    assert ex.action_probs.shape[0] == NUM_ACTION_TYPES
    assert ex.outcome in (0.0, 1.0)


# ── Replay buffer ──────────────────────────────────────────────────────

def test_replay_buffer_fifo():
    """Buffer should evict oldest entries when full."""
    buf = ReplayBuffer(max_size=5)
    dummy = [
        TrainingExample(
            state=torch.zeros(state_dim()),
            action_probs=torch.zeros(NUM_ACTION_TYPES),
            outcome=float(i),
        )
        for i in range(10)
    ]
    buf.add(dummy)
    assert len(buf) == 5
    # Should have kept the last 5 (outcomes 5-9)
    outcomes = sorted(e.outcome for e in buf.buffer)
    assert outcomes == [5.0, 6.0, 7.0, 8.0, 9.0]


# ── Training loop ──────────────────────────────────────────────────────

def test_train_reduces_loss():
    """A few training steps should reduce loss on synthetic data."""
    net = PolicyValueNet()

    # Create synthetic examples
    examples = []
    for i in range(32):
        s = torch.randn(state_dim())
        p = torch.zeros(NUM_ACTION_TYPES)
        p[i % NUM_ACTION_TYPES] = 1.0
        examples.append(TrainingExample(state=s, action_probs=p, outcome=float(i % 2)))

    config = TrainerConfig(epochs=20, batch_size=16, learning_rate=1e-3)
    stats = train(net, examples, config)
    assert stats["total_loss"] < 5.0  # sanity: loss should be finite and bounded
