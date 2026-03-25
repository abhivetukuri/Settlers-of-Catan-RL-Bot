"""Self-play training loop CLI.

Usage::

    python -m scripts.train --rounds 5 --games-per-round 20 \\
        --mcts-iters 50 --checkpoint-dir artifacts/checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.bots.mcts_bot import MCTSConfig
from src.models.policy_value_net import PolicyValueNet
from src.selfplay.data_generator import ReplayBuffer, SelfPlayConfig, generate_self_play_data
from src.selfplay.trainer import TrainerConfig, train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run self-play training rounds.")
    p.add_argument("--rounds", type=int, default=5,
                   help="Number of generate→train rounds (default: 5)")
    p.add_argument("--games-per-round", type=int, default=20,
                   help="Self-play games per round (default: 20)")
    p.add_argument("--mcts-iters", type=int, default=50,
                   help="MCTS iterations per decision during self-play (default: 50)")
    p.add_argument("--epochs", type=int, default=10,
                   help="Training epochs per round (default: 10)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Mini-batch size (default: 64)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--buffer-size", type=int, default=50_000,
                   help="Replay buffer max size (default: 50000)")
    p.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints",
                   help="Directory for saving model checkpoints")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed for self-play games (default: 0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    net = PolicyValueNet()
    buffer = ReplayBuffer(max_size=args.buffer_size)

    mcts_config = MCTSConfig(num_iterations=args.mcts_iters)
    trainer_config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=ckpt_dir,
    )

    for round_idx in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"  Round {round_idx}/{args.rounds}")
        print(f"{'='*60}")

        # ── Generate self-play data ──
        sp_config = SelfPlayConfig(
            mcts_config=mcts_config,
            num_games=args.games_per_round,
            base_seed=args.seed + (round_idx - 1) * args.games_per_round,
        )
        print(f"  Generating {sp_config.num_games} self-play games "
              f"({mcts_config.num_iterations} MCTS iters each)...")
        examples = generate_self_play_data(sp_config)
        buffer.add(examples)
        print(f"  Collected {len(examples)} examples (buffer: {len(buffer)})")

        # ── Train ──
        train_examples = buffer.sample(min(len(buffer), 4096))
        print(f"  Training on {len(train_examples)} examples "
              f"for {trainer_config.epochs} epochs...")
        stats = train(net, train_examples, trainer_config)
        print(f"  Loss: total={stats['total_loss']:.4f}  "
              f"policy={stats['policy_loss']:.4f}  "
              f"value={stats['value_loss']:.4f}")

        # ── Save checkpoint ──
        ckpt_path = ckpt_dir / f"model_round_{round_idx:03d}.pt"
        net.save(ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

    # Save final model with a canonical name
    final_path = ckpt_dir / "model_latest.pt"
    net.save(final_path)
    print(f"\nTraining complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
