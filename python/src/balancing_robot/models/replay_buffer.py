from collections import deque
import numpy as np
import random
from typing import Tuple, List, Optional

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity: int):
        """Initialize buffer with given capacity.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, 
             action: np.ndarray, 
             reward: float, 
             next_state: np.ndarray, 
             done: bool) -> None:
        """Store a transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay buffer."""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        """Initialize buffer with prioritized sampling.
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: How much prioritization to use (0 = uniform, 1 = full priority)
        """
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.epsilon = 1e-6  # Small constant to prevent zero priorities
        
    def push(self, state: np.ndarray, 
             action: np.ndarray, 
             reward: float, 
             next_state: np.ndarray, 
             done: bool) -> None:
        """Store a transition with maximum priority."""
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)
        super().push(state, action, reward, next_state, done)
        
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[np.ndarray, ...]:
        """Sample a batch of transitions with importance sampling weights.
        
        Args:
            batch_size: Number of transitions to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        if len(self.buffer) == 0:
            return None
            
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights
        
        # Get samples
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        
        return states, actions, rewards, next_states, dones, weights, indices
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """Update priorities for transitions.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
