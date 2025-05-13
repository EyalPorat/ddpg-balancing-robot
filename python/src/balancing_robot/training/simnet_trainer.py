import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import collections
from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer

from ..models import SimNet
from ..environment import BalancerEnv
from ..training.utils import TrainingLogger, save_model


class SimNetTrainer:
    """Trainer for the dynamics simulation network with enhanced state support."""

    def __init__(
        self,
        env: BalancerEnv,
        config_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize SimNet trainer.

        Args:
            env: Environment for collecting physics-based data
            config_path: Path to SimNet config file
            device: Device to use for training
        """
        self.env = env
        self.device = device

        # Load config
        if config_path:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()

        # Get enhanced state parameters from env config
        if hasattr(env, "config") and env.config:
            self.theta_history_size = env.config["observation"].get("theta_history_size", 3)
            self.theta_dot_history_size = env.config["observation"].get("theta_dot_history_size", 3)
            self.action_history_size = env.config["observation"].get("action_history_size", 4)
            self.motor_delay_steps = env.config["physics"].get("motor_delay_steps", 2)
        else:
            self.theta_history_size = 3
            self.theta_dot_history_size = 3
            self.action_history_size = 4
            self.motor_delay_steps = 2

        # Calculate enhanced state dimension
        # Basic state (2) + action history (4) + theta history (3) + theta_dot history (3)
        enhanced_state_dim = 2 + self.action_history_size + self.theta_history_size + self.theta_dot_history_size

        # Initialize SimNet from config with enhanced state dimension
        model_config = self.config["model"]
        self.simnet = SimNet(
            state_dim=enhanced_state_dim,
            action_dim=env.action_space.shape[0],
            hidden_dims=model_config["hidden_dims"],
        ).to(device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.simnet.parameters(), lr=self.config["training"]["pretrain"].get("learning_rate", 0.001)
        )

        # Set up scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config["training"]["pretrain"]["reduce_lr_factor"],
            patience=self.config["training"]["pretrain"]["reduce_lr_patience"],
            min_lr=self.config["training"]["pretrain"]["min_lr"],
            verbose=True,
        )

        print(f"Initialized SimNetTrainer with enhanced state (dim={enhanced_state_dim})")
        print(f"- theta_history_size: {self.theta_history_size}")
        print(f"- theta_dot_history_size: {self.theta_dot_history_size}")
        print(f"- action_history_size: {self.action_history_size}")
        print(f"- motor_delay_steps: {self.motor_delay_steps}")

    def _get_default_config(self) -> Dict:
        """Return default configuration if none provided."""
        return {
            "model": {
                "hidden_dims": [32, 32, 32],
                "dropout_rate": 0.25,
                "activation": "relu",
                "layer_norm": True,
            },
            "data_collection": {
                "physics_samples": 100000,
                "real_samples": 10000,
                "noise_std": 0.1,
                "observation_noise_std": 0.01,
                "validation_split": 0.2,
                "random_seed": 42,
            },
            "training": {
                "pretrain": {
                    "learning_rate": 0.00005,
                    "epochs": 30,
                    "batch_size": 512,
                    "early_stopping_patience": 10,
                    "reduce_lr_patience": 4,
                    "reduce_lr_factor": 0.5,
                    "min_lr": 0.0000001,
                },
                "finetune": {
                    "learning_rate": 0.00001,
                    "epochs": 50,
                    "batch_size": 128,
                    "early_stopping_patience": 5,
                    "reduce_lr_patience": 3,
                    "reduce_lr_factor": 0.5,
                    "min_lr": 0.0000001,
                },
                "class_balancing": {
                    "enabled": False,
                    "num_bins": 15,
                    "strategy": "kmeans",
                    "sample_weights": False,
                    "oversample": False,
                    "thresholds": {
                        "angle_deg": 20,
                        "angular_velocity_dps": 120,
                        "max_abs_action": 0.9,
                        "motor_angle_deg": 10,
                    },
                },
            },
            "hybrid": {
                "initial_ratio": 0.0,
                "target_ratio": 1.0,
                "adaptation_steps": 1000,
                "adaptation_schedule": "linear",
            },
            "logging": {
                "log_frequency": 1,
                "validation_frequency": 5,
                "save_best": True,
                "save_frequency": 10,
            },
        }

    def _create_enhanced_state(
        self,
        theta: float,
        theta_dot: float,
        theta_history: collections.deque,
        theta_dot_history: collections.deque,
        action_history: collections.deque,
    ) -> np.ndarray:
        """Create enhanced state representation."""
        # Create enhanced state
        return np.concatenate(
            [[theta, theta_dot], np.array(action_history), np.array(theta_history), np.array(theta_dot_history)]
        )

    def collect_physics_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Collect data with enhanced state representation by:
        1. Resetting the environment to its initial state
        2. Taking no action Episodes (letting the system fall on its own)
        3. Taking random action episodes (with noise)
        4. Adding observation noise to increase robustness
        5. Breaking on 'done'
        """
        config = self.config["data_collection"]
        num_samples = config["physics_samples"]
        action_noise_std = config["noise_std"]
        observation_noise_std = config.get("observation_noise_std", 0.01)

        # Arrays to store transitions
        states, actions, next_states = [], [], []

        # Steps per episode
        steps_per_episode = 500

        # Compute number of episodes for each type
        total_episodes = num_samples // steps_per_episode
        no_action_episodes = total_episodes // 3
        random_action_episodes = total_episodes - no_action_episodes

        # (1) "No action" episodes
        for _ in tqdm(range(no_action_episodes), desc="Collecting no-action physics data"):
            # Reset environment - already provides enhanced state
            state, _ = self.env.reset(should_zero_previous_action=True)

            # Add initial observation noise
            state[:2] = state[:2] + np.random.normal(0, observation_noise_std, size=2)

            for _ in range(steps_per_episode):
                s_t = state.copy()
                # Zero action for all motors, letting it fall
                a_t = np.zeros(self.env.action_space.shape, dtype=np.float32)
                next_state, _, done, _, _ = self.env.step(a_t)

                states.append(s_t)
                actions.append(a_t)
                next_states.append(next_state.copy())

                # break if angle is too large
                if abs(state[0]) > np.pi / 2:
                    break

                state = next_state

        # (2) Random actions episodes
        for _ in tqdm(range(random_action_episodes), desc="Collecting random-action physics data"):
            # Reset environment
            state, _ = self.env.reset()

            # Add initial observation noise
            state[:2] = state[:2] + np.random.normal(0, observation_noise_std, size=2)

            for _ in range(steps_per_episode):
                s_t = state.copy()

                # Sample random action
                a_t = np.random.uniform(-1, 1, size=self.env.action_space.shape)

                # Add action noise to increase exploration
                a_t = np.clip(a_t + np.random.normal(0, action_noise_std, size=self.env.action_space.shape), -1, 1)

                next_state, _, done, _, _ = self.env.step(a_t, action_as_actual_output=True)

                states.append(s_t)
                actions.append(a_t)
                next_states.append(next_state.copy())

                # break if angle is too large
                if abs(state[0]) > np.pi / 2:
                    break

                state = next_state

        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)

        # Randomly split train & val
        num_val = int(len(states) * config["validation_split"])
        indices = np.random.permutation(len(states))
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        train_data = {
            "states": states[train_indices],
            "actions": actions[train_indices],
            "next_states": next_states[train_indices],
        }

        val_data = {
            "states": states[val_indices],
            "actions": actions[val_indices],
            "next_states": next_states[val_indices],
        }

        return train_data, val_data

    def process_real_data(self, log_data: List[Dict[str, Any]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Process real robot log data with proper history handling for sequential prediction."""
        states = []
        actions = []
        next_states = []

        # Process each episode
        for episode in log_data:
            episode_states = episode["states"]

            # Skip episodes that are too short
            min_length_needed = max(self.action_history_size, self.theta_history_size, self.theta_dot_history_size) + 2
            if len(episode_states) < min_length_needed:
                continue

            # Initialize history buffers
            action_history = collections.deque([0.0] * self.action_history_size, maxlen=self.action_history_size)
            theta_history = collections.deque([0.0] * self.theta_history_size, maxlen=self.theta_history_size)
            theta_dot_history = collections.deque(
                [0.0] * self.theta_dot_history_size, maxlen=self.theta_dot_history_size
            )

            # Pre-fill history buffers with initial values
            for i in range(min_length_needed - 2):
                idx = min(i, len(episode_states) - 1)  # Avoid index out of bounds
                action = episode_states[idx]["motor_pwm"] / 127.0  # Normalize to [-1, 1]
                theta = episode_states[idx]["theta_global"]
                theta_dot = episode_states[idx]["theta_dot"]

                action_history.append(action)
                theta_history.append(theta)
                theta_dot_history.append(theta_dot)

            # Process state transitions
            for i in range(len(episode_states) - 1):
                current = episode_states[i]
                next_state_data = episode_states[i + 1]

                # Extract current values
                theta = current["theta_global"]
                theta_dot = current["theta_dot"]
                action_value = current["motor_pwm"] / 127.0  # Normalize to [-1, 1]
                action = np.array([action_value])

                # Get next state values
                next_theta = next_state_data["theta_global"]
                next_theta_dot = next_state_data["theta_dot"]

                # Create enhanced states using SAME history buffers for both
                enhanced_state = self._create_enhanced_state(
                    theta, theta_dot, theta_history, theta_dot_history, action_history
                )

                enhanced_next_state = self._create_enhanced_state(
                    next_theta, next_theta_dot, theta_history, theta_dot_history, action_history
                )

                # Store the transition
                states.append(enhanced_state)
                actions.append(action)
                next_states.append(enhanced_next_state)

                # ONLY NOW update history buffers for the next iteration
                action_history.append(action_value)
                theta_history.append(theta)
                theta_dot_history.append(theta_dot)

        # Convert to arrays
        if not states:  # Check if we have any data
            raise ValueError("No valid sequences found in log data")

        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)

        print(f"Processed {len(states)} state transitions from real data")

        # Split into train/validation
        val_split = self.config["data_collection"]["validation_split"]
        num_samples = len(states)
        num_val = int(num_samples * val_split)
        indices = np.random.permutation(num_samples)
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        train_data = {
            "states": states[train_indices],
            "actions": actions[train_indices],
            "next_states": next_states[train_indices],
        }

        val_data = {
            "states": states[val_indices],
            "actions": actions[val_indices],
            "next_states": next_states[val_indices],
        }

        return train_data, val_data

    @staticmethod
    def calculate_class_weights(states: np.ndarray, num_bins: int = 10, strategy: str = "uniform") -> np.ndarray:
        """Calculate class weights for balanced sampling based on state distribution."""
        # Only use the basic state components (theta and theta_dot) for binning
        states_2d = states[:, :2]  # Extract just angle and angular velocity

        # Create discretizer for state binning
        discretizer = KBinsDiscretizer(n_bins=num_bins, encode="ordinal", strategy=strategy)

        # Fit on angle and angular velocity separately
        angle_bins = discretizer.fit_transform(states_2d[:, 0].reshape(-1, 1))
        angular_vel_bins = discretizer.fit_transform(states_2d[:, 1].reshape(-1, 1))

        # Combine the bins to create a 2D grid of state space
        combined_bins = angle_bins * num_bins + angular_vel_bins
        combined_bins = combined_bins.astype(int).flatten()

        # Count occurrences of each state class
        bin_counts = np.bincount(combined_bins)

        # Calculate inverse frequency weights (higher for rare states)
        inv_bin_freq = 1.0 / (bin_counts + 1e-8)  # Add small epsilon to avoid div by zero

        # Normalize weights
        norm_weights = inv_bin_freq / np.sum(inv_bin_freq) * len(inv_bin_freq)

        # Map weights back to samples
        sample_weights = norm_weights[combined_bins]

        return sample_weights

    @staticmethod
    def create_balanced_dataset(
        data: Dict[str, np.ndarray], num_bins: int = 10, strategy: str = "uniform"
    ) -> Dict[str, np.ndarray]:
        """Create a balanced dataset using oversampling of minority classes."""
        states = data["states"]
        actions = data["actions"]
        next_states = data["next_states"]

        # Create discretizer (using just theta and theta_dot)
        discretizer = KBinsDiscretizer(n_bins=num_bins, encode="ordinal", strategy=strategy)

        # Discretize states to identify classes (use basic state elements)
        angle_bins = discretizer.fit_transform(states[:, 0].reshape(-1, 1))
        angular_vel_bins = discretizer.fit_transform(states[:, 1].reshape(-1, 1))

        # Combine the bins to create a 2D grid of state space
        combined_bins = (angle_bins * num_bins + angular_vel_bins).astype(int).flatten()

        # Count occurrences of each bin
        unique_bins, bin_counts = np.unique(combined_bins, return_counts=True)

        # Find the class with maximum number of samples
        max_samples = np.max(bin_counts)

        # Initialize arrays for balanced data
        balanced_states = []
        balanced_actions = []
        balanced_next_states = []

        # For each bin, either keep all samples or oversample
        for bin_idx in unique_bins:
            # Get indices belonging to this bin
            bin_indices = np.where(combined_bins == bin_idx)[0]

            # If we need to oversample
            if len(bin_indices) < max_samples:
                # Number of additional samples needed
                additional_samples = max_samples - len(bin_indices)

                # Randomly sample with replacement
                oversample_indices = np.random.choice(bin_indices, size=additional_samples, replace=True)

                # Combine original and oversampled indices
                all_indices = np.concatenate([bin_indices, oversample_indices])
            else:
                # Keep original samples if this bin has enough
                all_indices = bin_indices

            # Add to balanced dataset
            balanced_states.append(states[all_indices])
            balanced_actions.append(actions[all_indices])
            balanced_next_states.append(next_states[all_indices])

        # Concatenate all balanced data
        balanced_data = {
            "states": np.vstack(balanced_states),
            "actions": np.vstack(balanced_actions),
            "next_states": np.vstack(balanced_next_states),
        }

        return balanced_data

    @staticmethod
    def clean_data_exceptions(
        data: Dict[str, np.ndarray], std_threshold_multiplier: float = 3
    ) -> Dict[str, np.ndarray]:
        """Remove large standard deviation samples from data."""
        states = data["states"]
        actions = data["actions"]
        next_states = data["next_states"]

        # Calculate standard deviation of states
        state_std = np.std(states, axis=0)
        state_mean = np.mean(states, axis=0)
        # Define threshold for outliers (e.g., std_threshold_multiplier standard deviations from the mean)
        threshold = std_threshold_multiplier * state_std
        # Identify outliers
        outliers = np.any(np.abs(states - state_mean) > threshold, axis=1)
        # Filter out outliers
        cleaned_states = states[~outliers]

        cleaned_actions = actions[~outliers]
        cleaned_next_states = next_states[~outliers]

        # Create cleaned data dictionary
        cleaned_data = {
            "states": cleaned_states,
            "actions": cleaned_actions,
            "next_states": cleaned_next_states,
        }

        return cleaned_data

    @staticmethod
    def prepare_weighted_batch_indices(data_size: int, batch_size: int, weights: np.ndarray) -> List[np.ndarray]:
        """Prepare batch indices with weighted sampling."""
        # Normalize weights for sampling
        weights = weights / np.sum(weights)

        # Number of complete batches
        num_batches = data_size // batch_size

        batches = []
        for _ in range(num_batches):
            # Sample indices according to weights
            batch_indices = np.random.choice(
                data_size, size=batch_size, replace=True, p=weights  # Allow replacement for balanced sampling
            )
            batches.append(batch_indices)

        return batches

    def train_epoch(
        self,
        train_data: Dict[str, np.ndarray],
        batch_size: int,
        class_balancing_thresholds: Optional[Dict[str, float]],
        is_finetuning: bool = False,
    ) -> Dict[str, float]:
        """Train for one epoch with full state prediction."""
        self.simnet.train()
        total_loss = 0
        total_physics_loss = 0
        total_action_history_loss = 0
        total_theta_history_loss = 0
        total_theta_dot_history_loss = 0
        total_weighted_samples = 0

        # Pull out arrays for convenience
        states_arr = train_data["states"]
        actions_arr = train_data["actions"]
        next_states_arr = train_data["next_states"]

        num_samples = len(states_arr)
        num_batches = num_samples // batch_size

        for i in range(num_batches):
            idx = slice(i * batch_size, (i + 1) * batch_size)
            states = torch.FloatTensor(states_arr[idx]).to(self.device)
            actions = torch.FloatTensor(actions_arr[idx]).to(self.device)
            target_next_states = torch.FloatTensor(next_states_arr[idx]).to(self.device)

            # Compute target deltas (the differences between target state and current state)
            target_deltas = target_next_states - states

            # Predict deltas directly
            pred_deltas = self.simnet.predict_delta(states, actions)

            # Create weight tensor, default weight = 1.0
            weights = torch.ones(states.shape[0], device=self.device)

            # Calculate sample weights based on thresholds if in fine-tuning mode
            if class_balancing_thresholds is not None and is_finetuning:
                # Convert angles from radians to degrees (for thresholds specified in degrees)
                angles_degrees = torch.abs(states[:, 0] * 180.0 / np.pi) + 9
                angular_vels_degrees = torch.abs(states[:, 1] * 180.0 / np.pi)

                # Safely get action values - avoid squeeze() which can cause dimension issues
                action_values = torch.abs(actions[:, 0])  # Explicitly get first dimension

                # Apply higher weights for samples where angle or angular velocity exceeds thresholds
                extreme_samples = (
                    (angles_degrees > class_balancing_thresholds["angle_deg"])
                    | (angular_vels_degrees > class_balancing_thresholds["angular_velocity_dps"])
                    | (
                        (angles_degrees > class_balancing_thresholds["motor_angle_deg"])
                        & (action_values < class_balancing_thresholds["max_abs_action"])
                    )
                )

                non_extreme_weight = 0.1

                # Apply weighting
                weights[~extreme_samples] = non_extreme_weight

                # Count weighted samples for logging
                total_weighted_samples += extreme_samples.sum().item()

            # Create component weights - prioritize physics components
            component_weights = torch.ones_like(pred_deltas)
            # component_weights[:, 2:] = 0.5  # Lower weight for non-physics components

            # Compute weighted MSE loss on all delta components
            squared_errors = (pred_deltas - target_deltas) ** 2

            # Apply both sample weights and component weights
            # Explicitly reshape weights for broadcasting with 2D error tensors
            sample_weights = weights.view(-1, 1)  # [batch_size, 1]
            weighted_errors = sample_weights * component_weights * squared_errors

            loss = weighted_errors.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Track individual component losses for monitoring
            physics_loss = torch.nn.functional.mse_loss(pred_deltas[:, :2], target_deltas[:, :2])

            # Calculate history losses separately with explicit indexing
            idx_start = 2
            idx_end = idx_start + self.action_history_size
            action_history_loss = torch.nn.functional.mse_loss(
                pred_deltas[:, idx_start:idx_end], target_deltas[:, idx_start:idx_end]
            )

            idx_start = 2 + self.action_history_size
            idx_end = idx_start + self.theta_history_size
            theta_history_loss = torch.nn.functional.mse_loss(
                pred_deltas[:, idx_start:idx_end], target_deltas[:, idx_start:idx_end]
            )

            idx_start = 2 + self.action_history_size + self.theta_history_size
            idx_end = idx_start + self.theta_dot_history_size
            theta_dot_history_loss = torch.nn.functional.mse_loss(
                pred_deltas[:, idx_start:idx_end], target_deltas[:, idx_start:idx_end]
            )

            total_physics_loss += physics_loss.item()
            total_action_history_loss += action_history_loss.item()
            total_theta_history_loss += theta_history_loss.item()
            total_theta_dot_history_loss += theta_dot_history_loss.item()

        return {
            "train_loss": total_loss / num_batches,
            "physics_loss": total_physics_loss / num_batches,
            "action_history_loss": total_action_history_loss / num_batches,
            "theta_history_loss": total_theta_history_loss / num_batches,
            "theta_dot_history_loss": total_theta_dot_history_loss / num_batches,
            "extreme_samples_percent": (
                (total_weighted_samples / (num_batches * batch_size)) * 100 if total_weighted_samples > 0 else 0
            ),
        }

    def validate(self, val_data: Dict[str, np.ndarray], batch_size: int) -> Dict[str, float]:
        """Validate the model on full state delta predictions."""
        self.simnet.eval()
        total_loss = 0
        total_physics_loss = 0
        total_action_history_loss = 0
        total_theta_history_loss = 0
        total_theta_dot_history_loss = 0

        states_arr = val_data["states"]
        actions_arr = val_data["actions"]
        next_states_arr = val_data["next_states"]

        num_samples = len(states_arr)
        num_batches = max(1, num_samples // batch_size)

        with torch.no_grad():
            for i in range(num_batches):
                idx = slice(i * batch_size, min((i + 1) * batch_size, num_samples))
                states = torch.FloatTensor(states_arr[idx]).to(self.device)
                actions = torch.FloatTensor(actions_arr[idx]).to(self.device)
                target_next_states = torch.FloatTensor(next_states_arr[idx]).to(self.device)

                # Calculate target deltas
                target_deltas = target_next_states - states

                # Get delta predictions
                pred_deltas = self.simnet.predict_delta(states, actions)

                # Compute weighted MSE loss
                component_weights = torch.ones_like(pred_deltas)
                component_weights[:, 2:] = 0.5  # Lower weight for non-physics components

                squared_errors = (pred_deltas - target_deltas) ** 2
                weighted_errors = component_weights * squared_errors
                loss = weighted_errors.mean()

                total_loss += loss.item()

                # Track individual component losses for monitoring
                physics_loss = torch.nn.functional.mse_loss(pred_deltas[:, :2], target_deltas[:, :2])

                # Calculate history losses separately
                action_history_loss = (
                    torch.nn.functional.mse_loss(pred_deltas[:, 2:6], target_deltas[:, 2:6])
                    if pred_deltas.shape[1] > 5
                    else 0.0
                )
                theta_history_loss = (
                    torch.nn.functional.mse_loss(pred_deltas[:, 6:9], target_deltas[:, 6:9])
                    if pred_deltas.shape[1] > 8
                    else 0.0
                )
                theta_dot_history_loss = (
                    torch.nn.functional.mse_loss(pred_deltas[:, 9:12], target_deltas[:, 9:12])
                    if pred_deltas.shape[1] > 11
                    else 0.0
                )

                total_physics_loss += physics_loss.item()
                total_action_history_loss += (
                    action_history_loss.item() if not isinstance(action_history_loss, float) else 0.0
                )
                total_theta_history_loss += (
                    theta_history_loss.item() if not isinstance(theta_history_loss, float) else 0.0
                )
                total_theta_dot_history_loss += (
                    theta_dot_history_loss.item() if not isinstance(theta_dot_history_loss, float) else 0.0
                )

        return {
            "val_loss": total_loss / num_batches,
            "val_physics_loss": total_physics_loss / num_batches,
            "val_action_history_loss": total_action_history_loss / num_batches,
            "val_theta_history_loss": total_theta_history_loss / num_batches,
            "val_theta_dot_history_loss": total_theta_dot_history_loss / num_batches,
        }

    @staticmethod
    def analyze_class_distribution(
        data: Dict[str, np.ndarray], num_bins: int = 10, strategy: str = "uniform"
    ) -> Dict[str, Any]:
        """Analyze and visualize the class distribution in data."""
        states = data["states"]

        # Calculate statistics
        angle_range = (np.min(states[:, 0]), np.max(states[:, 0]))
        angle_mean = np.mean(states[:, 0])
        angle_std = np.std(states[:, 0])

        angular_vel_range = (np.min(states[:, 1]), np.max(states[:, 1]))
        angular_vel_mean = np.mean(states[:, 1])
        angular_vel_std = np.std(states[:, 1])

        # Calculate sample weights (inverse frequency)
        sample_weights = SimNetTrainer.calculate_class_weights(states, num_bins, strategy)
        weight_stats = {
            "min_weight": np.min(sample_weights),
            "max_weight": np.max(sample_weights),
            "mean_weight": np.mean(sample_weights),
            "std_weight": np.std(sample_weights),
        }

        # Return distribution statistics
        return {
            "angle_range": angle_range,
            "angle_mean": angle_mean,
            "angle_std": angle_std,
            "angular_vel_range": angular_vel_range,
            "angular_vel_mean": angular_vel_mean,
            "angular_vel_std": angular_vel_std,
            "num_samples": len(states),
            **weight_stats,
        }

    def train(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
        is_finetuning: bool = False,
        log_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train the SimNet model."""
        # Get appropriate training config
        train_config = self.config["training"]["finetune"] if is_finetuning else self.config["training"]["pretrain"]

        num_epochs = train_config["epochs"]
        batch_size = train_config["batch_size"]
        early_stopping_patience = train_config["early_stopping_patience"]

        # Set learning rate
        lr = train_config.get("learning_rate", 0.001)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        logger = TrainingLogger(log_dir) if log_dir else None
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            class_balancing_thresholds = (
                None if not is_finetuning else self.config["training"]["class_balancing"]["thresholds"]
            )
            # Train
            train_metrics = self.train_epoch(train_data, batch_size, class_balancing_thresholds, is_finetuning)

            # Validate
            val_metrics = self.validate(val_data, batch_size)

            # Update learning rate
            self.scheduler.step(val_metrics["val_loss"])

            # Log metrics
            metrics = {**train_metrics, **val_metrics, "epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"]}

            if logger:
                logger.log(metrics)

            # Early stopping check
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0

                if log_dir:
                    save_model(
                        self.simnet, Path(log_dir) / "best_simnet.pt", {"epoch": epoch, "val_loss": best_val_loss}
                    )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.6f} - "
                f"Val Loss: {val_metrics['val_loss']:.6f}"
            )

        if logger:
            logger.save()
            if log_dir:
                save_model(self.simnet, Path(log_dir) / "simnet_final.pt")

        return logger.metrics if logger else {}
