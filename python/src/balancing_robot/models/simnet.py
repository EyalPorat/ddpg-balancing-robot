import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class SimNet(nn.Module):
    """Neural network for simulating system dynamics using time series."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128, 128),
        time_steps: int = 10,
        single_state_dim: int = 3,
        learning_rate: float = 1e-3,
    ):
        """Initialize simulation network.

        Args:
            state_dim: Dimension of time series state (time_steps * single_state_dim)
            action_dim: Dimension of action space
            hidden_dims: Dimensions of hidden layers
            time_steps: Number of time steps in state history
            single_state_dim: Dimension of single state sample
            learning_rate: Learning rate for optimizer
        """
        super(SimNet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_steps = time_steps
        self.single_state_dim = single_state_dim

        # Calculate expected input dimension
        self.expected_input_dim = time_steps * single_state_dim

        # Network to predict next time series
        layers = []
        prev_dim = self.expected_input_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer predicts delta for the newest state
        self.output_layer = nn.Linear(prev_dim, single_state_dim)

        # Initialize weights
        self.apply(self._init_weights)

        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Print model configuration for debugging
        print(f"SimNet initialized with:")
        print(f"  time_steps: {time_steps}")
        print(f"  single_state_dim: {single_state_dim}")
        print(f"  expected_input_dim: {self.expected_input_dim}")
        print(f"  First layer input size: {self.expected_input_dim + action_dim}")

    def _init_weights(self, m):
        """Initialize network weights using orthogonal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state: Input time series state tensor [batch_size, state_dim]
            action: Input action tensor [batch_size, action_dim]

        Returns:
            Next time series state prediction [batch_size, state_dim]
        """
        # Check input dimensions
        batch_size = state.shape[0]
        state_size = state.shape[1]

        if state_size != self.expected_input_dim:
            raise ValueError(
                f"Input state dimension mismatch. Got {state_size}, expected {self.expected_input_dim} "
                f"(time_steps={self.time_steps} × single_state_dim={self.single_state_dim}). "
                f"Check time_steps and single_state_dim configuration."
            )

        # Handle input reshaping if necessary (ensure 2D)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        # Ensure action has correct shape [batch_size, action_dim]
        if action.shape[1] != self.action_dim:
            raise ValueError(f"Action dimension mismatch. Got {action.shape[1]}, expected {self.action_dim}")

        # Predict delta for newest state
        delta_state = self.predict_delta(state, action)

        # Create next time series state by shifting and updating
        next_state = state.clone()

        # Shift the time series (roll old states out, new state in)
        # Each sample is single_state_dim elements
        for i in range(0, state.shape[1] - self.single_state_dim, self.single_state_dim):
            next_state[:, i : i + self.single_state_dim] = state[
                :, i + self.single_state_dim : i + 2 * self.single_state_dim
            ]

        # Apply delta to most recent state to get new state
        current_newest = state[:, -self.single_state_dim :]
        new_sample = current_newest.clone()

        # Apply delta to physical variables
        new_sample[:, :2] = current_newest[:, :2] + delta_state[:, :2]

        # Update motor command directly
        new_sample[:, 2] = action.squeeze(1)

        # Place new state at the end of the time series
        next_state[:, -self.single_state_dim :] = new_sample

        return next_state

    def predict_delta(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict delta state directly.

        Args:
            state: Input time series state tensor [batch_size, state_dim]
            action: Input action tensor [batch_size, action_dim]

        Returns:
            Delta state predictions [batch_size, single_state_dim]
        """
        x = torch.cat([state, action], dim=1)
        x = self.hidden_layers(x)
        delta_state = self.output_layer(x)
        return delta_state

    def update(self, states: torch.Tensor, actions: torch.Tensor, target_states: torch.Tensor) -> Dict[str, float]:
        """Update network weights using supervised learning.

        Args:
            states: Batch of time series states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            target_states: Target time series states [batch_size, state_dim]

        Returns:
            Dictionary containing training metrics
        """
        # Extract newest samples from current and target states
        current_newest = states[:, -self.single_state_dim :]
        target_newest = target_states[:, -self.single_state_dim :]

        # Calculate target deltas for physical variables
        target_deltas = torch.zeros_like(current_newest)
        target_deltas[:, :2] = target_newest[:, :2] - current_newest[:, :2]

        # Get delta predictions directly
        pred_deltas = self.predict_delta(states, actions)

        # Compute loss - only on the physical variables (angle, angular velocity)
        loss = F.mse_loss(pred_deltas[:, :2], target_deltas[:, :2])

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
