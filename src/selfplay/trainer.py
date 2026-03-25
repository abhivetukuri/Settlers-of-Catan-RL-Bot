"""Training loop for the policy/value network.

Ingests ``TrainingExample`` objects produced by
:mod:`src.selfplay.data_generator` and trains the network with a
combined cross-entropy (policy) + MSE (value) loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.policy_value_net import PolicyValueNet
from src.selfplay.data_generator import TrainingExample


@dataclass
class TrainerConfig:
    """Hyper-parameters for the training loop."""
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    checkpoint_dir: Path = field(default_factory=lambda: Path("artifacts/checkpoints"))


def _build_tensors(
    examples: list[TrainingExample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a list of examples into batch tensors."""
    states = torch.stack([e.state for e in examples])
    probs = torch.stack([e.action_probs for e in examples])
    outcomes = torch.tensor([e.outcome for e in examples], dtype=torch.float32).unsqueeze(1)
    return states, probs, outcomes


def train(
    net: PolicyValueNet,
    examples: list[TrainingExample],
    config: TrainerConfig | None = None,
) -> dict:
    """Run one round of training and return loss statistics.

    Returns
    -------
    dict
        Keys: ``total_loss``, ``policy_loss``, ``value_loss`` (averages
        over the final epoch).
    """
    if config is None:
        config = TrainerConfig()

    states, target_probs, target_values = _build_tensors(examples)
    dataset = TensorDataset(states, target_probs, target_values)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )

    net.train()
    last_epoch_stats = {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

    for epoch in range(config.epochs):
        epoch_total = 0.0
        epoch_policy = 0.0
        epoch_value = 0.0
        num_batches = 0

        for batch_states, batch_probs, batch_values in loader:
            policy_logits, pred_values = net(batch_states)

            # Policy loss: cross-entropy with soft targets
            log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_loss = -torch.mean(torch.sum(batch_probs * log_probs, dim=-1))

            # Value loss: MSE
            value_loss = F.mse_loss(pred_values, batch_values)

            loss = policy_loss + config.value_loss_weight * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_policy += policy_loss.item()
            epoch_value += value_loss.item()
            num_batches += 1

        if num_batches > 0:
            last_epoch_stats = {
                "total_loss": epoch_total / num_batches,
                "policy_loss": epoch_policy / num_batches,
                "value_loss": epoch_value / num_batches,
            }

    return last_epoch_stats
