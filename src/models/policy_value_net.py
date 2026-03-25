"""Dual-head policy/value network for Catan.

The network takes a flat state-feature vector and produces:
  • **policy logits** — one per ``ActionType`` (13 total).
  • **value** — scalar in [0, 1] estimating the win probability.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.features import NUM_ACTION_TYPES, state_dim


class PolicyValueNet(nn.Module):
    """Simple MLP with shared trunk and two heads."""

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int = 128,
        num_actions: int = NUM_ACTION_TYPES,
    ) -> None:
        super().__init__()
        if input_dim is None:
            input_dim = state_dim(num_players=2)

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of state feature vectors, shape ``(B, input_dim)``.

        Returns
        -------
        policy_logits : torch.Tensor
            Raw logits, shape ``(B, num_actions)``.
        value : torch.Tensor
            Win-probability estimate, shape ``(B, 1)``, range [0, 1].
        """
        h = self.trunk(x)
        policy_logits = self.policy_head(h)
        value = torch.sigmoid(self.value_head(h))
        return policy_logits, value

    # -----------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------

    def predict(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Single-sample inference (no grad).

        Parameters
        ----------
        x : torch.Tensor
            1-D feature vector.
        legal_mask : torch.Tensor, optional
            Boolean tensor of shape ``(num_actions,)``.  If provided,
            illegal actions are masked to ``-inf`` before softmax.

        Returns
        -------
        action_probs : torch.Tensor
            Probability distribution over action types (1-D).
        value : float
            Scalar win-probability estimate.
        """
        self.eval()
        with torch.no_grad():
            logits, val = self.forward(x.unsqueeze(0))
            logits = logits.squeeze(0)
            if legal_mask is not None:
                logits = logits.masked_fill(~legal_mask, float("-inf"))
            probs = F.softmax(logits, dim=-1)
        return probs, val.item()

    # -----------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model weights to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "PolicyValueNet":
        """Load model weights from *path*."""
        net = cls(**kwargs)
        net.load_state_dict(torch.load(path, weights_only=True))
        net.eval()
        return net
