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

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
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

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store a transition with maximum priority."""
        # Get maximum priority (or default if empty)
        max_priority = max([abs(p) for p in self.priorities]) if self.priorities else 1.0

        # Ensure max_priority is positive and non-zero
        max_priority = max(max_priority, self.epsilon)

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

        # Calculate sampling probabilities, ensuring no NaNs or zeros
        priorities = np.array([abs(p) for p in self.priorities])
        # Add epsilon to all priorities to avoid zeros and numerical instability
        priorities = priorities + self.epsilon

        # Clip extremely large values to prevent overflow
        max_value = 1e6
        priorities = np.clip(priorities, 0, max_value)

        # Calculate probabilities - alpha controls the amount of prioritization
        probabilities = priorities**self.alpha

        # Safety check for NaNs and normalize
        if np.any(np.isnan(probabilities)):
            # Fallback to uniform sampling if NaNs are detected
            print("Warning: NaN detected in priorities, using uniform sampling")
            probabilities = np.ones_like(probabilities)

        # Normalize probabilities to sum to 1
        sum_probs = np.sum(probabilities)
        if sum_probs <= 0 or np.isnan(sum_probs):
            # Another safety check - should never happen with the above checks
            print("Warning: Invalid probability sum, using uniform sampling")
            probabilities = np.ones_like(probabilities) / len(probabilities)
        else:
            probabilities = probabilities / sum_probs

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Calculate importance sampling weights
        # N * P(i) where N is buffer size
        sampling_weights = (len(self.buffer) * probabilities[indices]) ** (-beta)

        # Normalize weights to have max weight = 1
        max_weight = np.max(sampling_weights)
        if max_weight > 0 and not np.isnan(max_weight):
            weights = sampling_weights / max_weight
        else:
            # Fallback if max_weight is invalid
            weights = np.ones_like(sampling_weights)

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
            # Handle NaN or negative priorities
            if np.isnan(priority) or priority < 0:
                priority = self.epsilon

            # Ensure priority is at least epsilon
            priority = max(priority, self.epsilon)

            # Update the priority in our deque
            # Convert deque to list for indexed access
            priorities_list = list(self.priorities)
            if 0 <= idx < len(priorities_list):
                priorities_list[idx] = priority

                # Convert back to deque
                self.priorities = deque(priorities_list, maxlen=self.priorities.maxlen)
            else:
                print(f"Warning: Index {idx} out of range for priorities list of length {len(self.priorities)}")
