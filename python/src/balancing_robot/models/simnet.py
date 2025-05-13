import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class SimNet(nn.Module):
    """Neural network for simulating system dynamics with LSTM for time series processing."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64, 64),
        lstm_hidden_size: int = 32,
        lstm_num_layers: int = 1,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
    ):
        """Initialize simulation network with LSTM components for time series.

        Args:
            state_dim: Dimension of state space (enhanced state with history components)
            action_dim: Dimension of action space
            hidden_dims: Dimensions of hidden layers for final prediction network
            lstm_hidden_size: Hidden size of LSTM units
            lstm_num_layers: Number of LSTM layers
            dropout_rate: Dropout probability for regularization
            learning_rate: Learning rate for optimizer
        """
        super(SimNet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.basic_state_dim = 2  # The core state variables: theta, theta_dot

        # Extract dimensions for different components based on configuration
        # Assume the first 2 elements are basic state (theta, theta_dot)
        # The rest is divided into action history, theta history, and theta_dot history
        remaining_dim = state_dim - self.basic_state_dim

        # Default history dimensions if not explicitly provided
        self.action_history_size = 4
        self.theta_history_size = 3
        self.theta_dot_history_size = 3

        # Ensure the sum adds up to the remaining_dim
        if self.action_history_size + self.theta_history_size + self.theta_dot_history_size != remaining_dim:
            # Adjust sizes proportionally to match remaining_dim
            total = self.action_history_size + self.theta_history_size + self.theta_dot_history_size
            self.action_history_size = int(self.action_history_size * remaining_dim / total)
            self.theta_history_size = int(self.theta_history_size * remaining_dim / total)
            self.theta_dot_history_size = remaining_dim - self.action_history_size - self.theta_history_size

        print(
            f"Initialized LSTMSimNet with history sizes: action={self.action_history_size}, "
            f"theta={self.theta_history_size}, theta_dot={self.theta_dot_history_size}"
        )

        # LSTM for processing action history (sequence length = action_history_size)
        self.action_lstm = nn.LSTM(
            input_size=1,  # Each action is a scalar
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
        )

        # LSTM for processing combined theta and theta_dot history
        # We combine these as they are physically related
        self.state_lstm = nn.LSTM(
            input_size=2,  # Each state point has theta and theta_dot
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
        )

        # Calculate combined representation size
        combined_size = self.basic_state_dim + action_dim + lstm_hidden_size * 2

        # Fully connected layers for final prediction
        layers = []
        prev_dim = combined_size

        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout_rate), nn.ReLU()]
            )
            prev_dim = hidden_dim

        self.fc_layers = nn.Sequential(*layers)

        # Output delta state (change in the full state)
        self.output_layer = nn.Linear(prev_dim, state_dim)

        # Initialize weights
        self.apply(self._init_weights)

        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.orthogonal_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)

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

        # Update action history (shift actions)
        # Assuming action history starts at index 2 and has length action_history_size
        action_history_start = 2

        # Add the new action at the beginning of history
        next_state[:, action_history_start] = action.squeeze()
        if self.action_history_size > 1:
            next_state[:, action_history_start + 1 : action_history_start + self.action_history_size] = state[
                :, action_history_start : action_history_start + self.action_history_size - 1
            ]

        return next_state

    def predict_delta(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict delta state using LSTM processing for historical components.

        Args:
            state: Input state tensor [batch_size, state_dim]
            action: Input action tensor [batch_size, action_dim]

        Returns:
            Delta state predictions [batch_size, state_dim]
        """
        batch_size = state.shape[0]

        # Extract current state components (theta, theta_dot)
        current_state = state[:, : self.basic_state_dim]

        # Extract history components
        idx = self.basic_state_dim

        # Action history is next action_history_size elements
        action_history = state[:, idx : idx + self.action_history_size]
        idx += self.action_history_size

        # Theta history is next theta_history_size elements
        theta_history = state[:, idx : idx + self.theta_history_size]
        idx += self.theta_history_size

        # Theta_dot history is next theta_dot_history_size elements
        theta_dot_history = state[:, idx : idx + self.theta_dot_history_size]

        # Reshape history components for LSTM processing
        # For action history: [batch_size, action_history_size, 1]
        action_history = action_history.unsqueeze(-1)

        # Process action history with LSTM
        action_lstm_out, _ = self.action_lstm(action_history)
        # Take the output from the last time step
        action_lstm_features = action_lstm_out[:, -1, :]

        # Create state history sequence by combining theta and theta_dot histories
        # We need to align the sequences to have the same length
        min_history_length = min(self.theta_history_size, self.theta_dot_history_size)

        # Create a combined tensor [batch_size, min_history_length, 2]
        state_history = torch.zeros(batch_size, min_history_length, 2, device=state.device)
        state_history[:, :, 0] = theta_history[:, :min_history_length]
        state_history[:, :, 1] = theta_dot_history[:, :min_history_length]

        # Process state history with LSTM
        state_lstm_out, _ = self.state_lstm(state_history)
        # Take the output from the last time step
        state_lstm_features = state_lstm_out[:, -1, :]

        # Concatenate all features: current state, current action, LSTM features
        combined_features = torch.cat(
            [
                current_state,  # Basic state (theta, theta_dot)
                action,  # Current action
                action_lstm_features,  # Features from action history
                state_lstm_features,  # Features from state history
            ],
            dim=1,
        )

        # Process through fully connected layers
        x = self.fc_layers(combined_features)

        # Predict delta state
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

        # Get delta predictions
        pred_deltas = self.predict_delta(states, actions)

        # Compute component weights - prioritize physics components
        component_weights = torch.ones_like(pred_deltas)
        component_weights[:, 2:] = 0.5  # Lower weight for non-physics components

        # Compute weighted MSE loss
        squared_errors = (pred_deltas - target_deltas) ** 2
        weighted_errors = component_weights * squared_errors
        loss = weighted_errors.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Calculate individual component metrics for monitoring
        physics_loss = F.mse_loss(pred_deltas[:, :2], target_deltas[:, :2])

        # Calculate history losses separately
        action_history_loss = (
            F.mse_loss(
                pred_deltas[:, 2 : 2 + self.action_history_size], target_deltas[:, 2 : 2 + self.action_history_size]
            )
            if pred_deltas.shape[1] > 2 + self.action_history_size
            else 0.0
        )

        theta_history_loss = (
            F.mse_loss(
                pred_deltas[:, 2 + self.action_history_size : 2 + self.action_history_size + self.theta_history_size],
                target_deltas[:, 2 + self.action_history_size : 2 + self.action_history_size + self.theta_history_size],
            )
            if pred_deltas.shape[1] > 2 + self.action_history_size + self.theta_history_size
            else 0.0
        )

        idx_start = 2 + self.action_history_size + self.theta_history_size
        idx_end = idx_start + self.theta_dot_history_size
        theta_dot_history_loss = (
            F.mse_loss(pred_deltas[:, idx_start:idx_end], target_deltas[:, idx_start:idx_end])
            if pred_deltas.shape[1] > idx_end
            else 0.0
        )

        return {
            "loss": loss.item(),
            "physics_loss": physics_loss.item(),
            "action_history_loss": action_history_loss.item() if not isinstance(action_history_loss, float) else 0.0,
            "theta_history_loss": theta_history_loss.item() if not isinstance(theta_history_loss, float) else 0.0,
            "theta_dot_history_loss": (
                theta_dot_history_loss.item() if not isinstance(theta_dot_history_loss, float) else 0.0
            ),
            "pred_delta_mean": pred_deltas.mean().item(),
            "pred_delta_std": pred_deltas.std().item(),
            "target_delta_mean": target_deltas.mean().item(),
            "target_delta_std": target_deltas.std().item(),
        }
