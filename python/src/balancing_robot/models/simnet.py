import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class SimNet(nn.Module):
    """Neural network for simulating system dynamics."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (128, 128), learning_rate: float = 1e-3
    ):
        """Initialize simulation network.

        Args:
            state_dim: Dimension of state space (including enhanced features)
            action_dim: Dimension of action space
            hidden_dims: Dimensions of hidden layers
            learning_rate: Learning rate for optimizer
        """
        super(SimNet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.basic_state_dim = 3  # The core state variables: theta, theta_dot, prev_action

        # Network to predict delta state (change in state)
        layers = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output delta state (change in the full state, not just basic components)
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
            state: Input state tensor [batch_size, state_dim] including enhanced features
            action: Input action tensor [batch_size, action_dim]

        Returns:
            Next state predictions [batch_size, state_dim]
        """
        # Predict delta (change in state)
        delta_state = self.predict_delta(state, action)

        # Calculate next state by adding delta to current state
        next_state = state.clone()

        # Apply predicted deltas to the entire state
        next_state = next_state + delta_state

        # The action component should be replaced with the current action
        next_state[:, 2] = action.squeeze()

        # Update action history (shift actions)
        # Assuming action history starts at index 5 and has length action_history_size
        action_history_start = 5

        # If there's an action history to update
        if state.shape[1] > action_history_start:
            action_history_size = state.shape[1] - action_history_start

            # Add the new action to the beginning of history
            if action_history_size > 0:
                # Shift the action history, add new action at front
                next_state[:, action_history_start] = action.squeeze()
                if action_history_size > 1:
                    next_state[:, action_history_start + 1 : action_history_start + action_history_size] = state[
                        :, action_history_start : action_history_start + action_history_size - 1
                    ]

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
        target_deltas = target_states - states

        # Get delta predictions directly
        pred_deltas = self.predict_delta(states, actions)

        # Compute loss - Create weights for different components
        # Weights vector: 1.0 for physics components (theta, theta_dot), 0.5 for others
        component_weights = torch.ones_like(pred_deltas)
        component_weights[:, 2:] = 0.5  # Lower weight for non-physics components

        # Weighted MSE loss on all components
        squared_errors = (pred_deltas - target_deltas) ** 2
        weighted_errors = component_weights * squared_errors
        loss = weighted_errors.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Calculate individual component metrics for monitoring
        physics_loss = F.mse_loss(pred_deltas[:, :2], target_deltas[:, :2])
        cmd_loss = F.mse_loss(pred_deltas[:, 2], target_deltas[:, 2]) if pred_deltas.shape[1] > 2 else 0.0
        history_loss = F.mse_loss(pred_deltas[:, 3:], target_deltas[:, 3:]) if pred_deltas.shape[1] > 3 else 0.0

        return {
            "loss": loss.item(),
            "physics_loss": physics_loss.item(),
            "cmd_loss": cmd_loss.item() if not isinstance(cmd_loss, int) else 0.0,
            "history_loss": history_loss.item() if not isinstance(history_loss, int) else 0.0,
            "pred_delta_mean": pred_deltas.mean().item(),
            "pred_delta_std": pred_deltas.std().item(),
            "target_delta_mean": target_deltas.mean().item(),
            "target_delta_std": target_deltas.std().item(),
        }
