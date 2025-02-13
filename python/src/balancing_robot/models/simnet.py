import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

class SimNet(nn.Module):
    """Neural network for simulating system dynamics."""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dims: Tuple[int, ...] = (128, 128),
                 learning_rate: float = 1e-3):
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
        
        # Network to predict accelerations
        layers = []
        prev_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output accelerations for [theta_ddot, x_ddot, phi_ddot]
        self.output_layer = nn.Linear(prev_dim, 3)
        
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
            Acceleration predictions [batch_size, 3]
        """
        x = torch.cat([state, action], dim=1)
        x = self.hidden_layers(x)
        return self.output_layer(x)
    
    def update(self, 
              states: torch.Tensor, 
              actions: torch.Tensor, 
              target_accels: torch.Tensor) -> Dict[str, float]:
        """Update network weights using supervised learning.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            target_accels: Target accelerations [batch_size, 3]
            
        Returns:
            Dictionary containing training metrics
        """
        # Get predictions
        pred_accels = self(states, actions)
        
        # Compute loss
        loss = F.mse_loss(pred_accels, target_accels)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'pred_mean': pred_accels.mean().item(),
            'pred_std': pred_accels.std().item(),
            'target_mean': target_accels.mean().item(),
            'target_std': target_accels.std().item()
        }
    
    def get_accelerations(self, 
                         state: np.ndarray, 
                         action: np.ndarray) -> np.ndarray:
        """Get acceleration predictions for a single state-action pair.
        
        Args:
            state: Current state array
            action: Current action array
            
        Returns:
            Predicted accelerations [theta_ddot, x_ddot, phi_ddot]
        """
        # Convert to tensors
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.FloatTensor(action).unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            accelerations = self(state, action)
            
        return accelerations.squeeze(0).numpy()
