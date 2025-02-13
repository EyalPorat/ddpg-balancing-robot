import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class Actor(nn.Module):
    """Actor network for DDPG."""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 max_action: float,
                 hidden_dims: Tuple[int, ...] = (8, 8)):
        """Initialize actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum action value
            hidden_dims: Dimensions of hidden layers
        """
        super(Actor, self).__init__()
        self.max_action = max_action
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        # Output layer
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize network weights using orthogonal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
            
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Action tensor scaled by max_action
        """
        x = self.network(state)
        return self.max_action * torch.tanh(self.output_layer(x))
