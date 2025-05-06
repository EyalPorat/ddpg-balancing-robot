import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class SimNet(nn.Module):
    """Neural network for simulating system dynamics using delta predictions."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (128, 128), learning_rate: float = 1e-3
    ):
        """Initialize simulation network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Dimensions of hidden layers
            learning_rate: Learning rate for optimizer
        """
        super(SimNet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Network to predict delta state (change in state)
        layers = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output delta state (change in state)
        self.output_layer = nn.Linear(prev_dim, state_dim)

        # Initialize weights
        self.apply(self._init_weights)

        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def _init_weights(self, m):
        """Initialize network weights using orthogonal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state: Input state tensor [batch_size, state_dim]
            action: Input action tensor [batch_size, action_dim]

        Returns:
            Next state predictions [batch_size, state_dim]
        """
        # Predict delta (change in state)
        delta_state = self.predict_delta(state, action)

        # Calculate next state by adding delta to current state
        next_state = state.clone()
        # Only apply delta to the first two elements (theta, theta_dot)
        next_state[:, :2] = state[:, :2] + delta_state[:, :2]

        # The last element of the state is the previous action
        # We need to replace it with the current action for the next step
        if self.state_dim > 2:  # If we're using the expanded state space
            # Replace the last element (prev_action) with the current action
            next_state[:, -1] = action.squeeze()

        return next_state

    def predict_delta(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict delta state directly, useful for analyzing model behavior.

        Args:
            state: Input state tensor [batch_size, state_dim]
            action: Input action tensor [batch_size, action_dim]

        Returns:
            Delta state predictions [batch_size, state_dim]
        """
        x = torch.cat([state, action], dim=1)
        x = self.hidden_layers(x)
        delta_state = self.output_layer(x)
        return delta_state

    def update(self, states: torch.Tensor, actions: torch.Tensor, target_states: torch.Tensor) -> Dict[str, float]:
        """Update network weights using supervised learning.

        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            target_states: Target state [batch_size, state_dim]

        Returns:
            Dictionary containing training metrics
        """
        # Calculate target deltas (difference between target state and current state)
        target_deltas = torch.zeros_like(target_states)
        target_deltas[:, :2] = target_states[:, :2] - states[:, :2]

        # Get delta predictions directly
        pred_deltas = self.predict_delta(states, actions)

        # Compute loss - only on the first two dimensions (angle, angular velocity)
        # The third dimension (previous action) is determined by the current action
        if self.state_dim > 2:
            loss = F.mse_loss(pred_deltas[:, :2], target_deltas[:, :2])
        else:
            loss = F.mse_loss(pred_deltas, target_deltas)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "pred_delta_mean": pred_deltas[:, :2].mean().item(),
            "pred_delta_std": pred_deltas[:, :2].std().item(),
            "target_delta_mean": target_deltas[:, :2].mean().item(),
            "target_delta_std": target_deltas[:, :2].std().item(),
        }
