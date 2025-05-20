import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class Critic(nn.Module):
    """Critic network for DDPG."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (256, 256)):
        """Initialize critic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Dimensions of hidden layers
        """
        super(Critic, self).__init__()

        # First layer processes state
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])

        # Hidden layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.LayerNorm(hidden_dims[i + 1]), nn.ReLU()])

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize network weights using orthogonal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state: Input state tensor
            action: Input action tensor

        Returns:
            Q-value tensor
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.ln1(self.l1(x)))
        x = self.hidden_layers(x)
        return self.output_layer(x)
