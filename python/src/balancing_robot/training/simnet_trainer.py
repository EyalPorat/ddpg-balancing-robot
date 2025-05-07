import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer

from ..models import SimNet
from ..environment import BalancerEnv
from .utils import TrainingLogger, save_model, load_model


class SimNetTrainer:
    """Trainer for the dynamics simulation network."""

    def __init__(
        self,
        env: BalancerEnv,
        config_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize SimNet trainer with time series support.

        Args:
            env: Environment for collecting physics-based data
            config_path: Path to SimNet config file
            device: Device to use for training
        """
        self.env = env
        self.device = device

        # Get time series parameters from environment
        self.time_steps = getattr(env, "time_steps", 10)
        self.single_state_dim = getattr(env, "single_state_dim", 3)

        # Calculate full state dimension for time series
        self.full_state_dim = self.time_steps * self.single_state_dim

        # Load config
        if config_path:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()

        # Initialize SimNet from config
        model_config = self.config["model"]

        # Initialize SimNet with time series dimensions
        self.simnet = SimNet(
            state_dim=self.full_state_dim,
            action_dim=env.action_space.shape[0],
            hidden_dims=model_config["hidden_dims"],
            time_steps=self.time_steps,
            single_state_dim=self.single_state_dim,
        ).to(device)

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.simnet.parameters(), lr=model_config.get("learning_rate", 0.001))
        self.simnet.optimizer = self.optimizer  # Attach optimizer to model for convenience

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config["training"]["pretrain"]["reduce_lr_factor"],
            patience=self.config["training"]["pretrain"]["reduce_lr_patience"],
            min_lr=self.config["training"]["pretrain"]["min_lr"],
            verbose=True,
        )

    def _get_default_config(self) -> Dict:
        """Return default configuration if none provided."""
        return {
            "model": {
                "hidden_dims": [128, 128, 128],
                "learning_rate": 0.001,
                "dropout_rate": 0.1,
                "activation": "relu",
                "layer_norm": True,
            },
            "data_collection": {
                "num_episodes": 500,
                "max_steps_per_episode": 200,
                "noise_std": 0.1,
                "validation_split": 0.1,
                "random_seed": 42,
            },
            "training": {
                "pretrain": {
                    "epochs": 50,
                    "batch_size": 512,
                    "early_stopping_patience": 10,
                    "reduce_lr_patience": 5,
                    "reduce_lr_factor": 0.5,
                    "min_lr": 0.00001,
                },
                "finetune": {
                    "epochs": 20,
                    "batch_size": 128,
                    "early_stopping_patience": 5,
                    "reduce_lr_patience": 3,
                    "reduce_lr_factor": 0.5,
                    "min_lr": 0.00001,
                },
                "class_balancing": {
                    "enabled": False,
                    "num_bins": 10,
                    "strategy": "uniform",
                    "sample_weights": False,
                    "oversample": False,
                    "thresholds": {
                        "angle_deg": 25,
                        "angular_velocity_dps": 120,
                        "motor_angle_deg": 10,
                        "max_abs_action": 0.9,
                    },
                },
            },
            "hybrid": {
                "initial_ratio": 0.0,
                "target_ratio": 1.0,
                "adaptation_steps": 1000,
                "adaptation_schedule": "linear",
            },
        }

    def collect_physics_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Collect data by:
        1. Resetting the environment to its initial state
        2. Taking no action Episodes (letting the system fall on its own)
        3. Taking random action episodes (with noise)
        4. Adding observation noise to increase robustness
        5. Breaking on 'done'
        """
        config = self.config["data_collection"]
        num_samples = config["physics_samples"]
        action_noise_std = config["noise_std"]
        observation_noise_std = config.get("observation_noise_std", 0.01)  # Default observation noise

        # Arrays to store transitions
        time_series_states = []
        actions = []
        next_time_series_states = []

        # Decide how many steps to record per episode
        steps_per_episode = 500

        # Compute number of episodes for each type
        # We'll have equal numbers of no-action and random-action episodes
        total_episodes = num_samples // steps_per_episode
        no_action_episodes = total_episodes // 3
        random_action_episodes = total_episodes - no_action_episodes

        print(f"Collecting {no_action_episodes} no-action episodes and {random_action_episodes} random-action episodes")

        # (1) "No action" episodes
        for _ in tqdm(range(no_action_episodes), desc="Collecting no-action physics data"):
            # Reset environment with zero previous action
            state, _ = self.env.reset(should_zero_previous_action=True)

            # Current time series state is the complete state from reset
            current_time_series = state.copy()

            for _ in range(steps_per_episode):
                # Store current time series state
                current_time_series_copy = current_time_series.copy()

                # Zero action for all motors, letting it fall
                a_t = np.zeros(self.env.action_space.shape, dtype=np.float32)

                # Execute step - will return time series state
                next_state, _, done, _, _ = self.env.step(a_t)

                # Next time series state
                next_time_series = next_state.copy()

                # Store transition
                time_series_states.append(current_time_series_copy)
                actions.append(a_t)
                next_time_series_states.append(next_time_series)

                # Update for next step
                current_time_series = next_time_series

                # Break if angle is too large
                if done:
                    break

        # (2) Random actions episodes
        for _ in tqdm(range(random_action_episodes), desc="Collecting random-action physics data"):
            # Reset environment
            state, _ = self.env.reset()

            # Current time series state is the complete state from reset
            current_time_series = state.copy()

            for _ in range(steps_per_episode):
                # Store current time series state
                current_time_series_copy = current_time_series.copy()

                # Sample random action
                a_t = np.random.uniform(-1, 1, size=self.env.action_space.shape)

                # Add action noise to increase exploration
                a_t = np.clip(a_t + np.random.normal(0, action_noise_std, size=self.env.action_space.shape), -1, 1)

                # Execute step with random action - will return time series state
                next_state, _, done, _, _ = self.env.step(a_t, action_as_actual_output=True)

                # Next time series state
                next_time_series = next_state.copy()

                # Store transition
                time_series_states.append(current_time_series_copy)
                actions.append(a_t)
                next_time_series_states.append(next_time_series)

                # Update for next step
                current_time_series = next_time_series

                # Break if episode done
                if done:
                    break

        # Convert to numpy arrays
        time_series_states = np.array(time_series_states)
        actions = np.array(actions)
        next_time_series_states = np.array(next_time_series_states)

        # Log collection statistics
        print(f"Collected {len(time_series_states)} total transitions")
        print(f"State shape: {time_series_states.shape}")
        print(f"Action shape: {actions.shape}")

        # Randomly split train & val
        num_val = int(len(time_series_states) * config["validation_split"])
        indices = np.random.permutation(len(time_series_states))
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        train_data = {
            "states": time_series_states[train_indices],
            "actions": actions[train_indices],
            "next_states": next_time_series_states[train_indices],
        }

        val_data = {
            "states": time_series_states[val_indices],
            "actions": actions[val_indices],
            "next_states": next_time_series_states[val_indices],
        }

        return train_data, val_data

    def process_real_data(self, log_data: List[Dict[str, Any]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Process real robot log data for training with time series format."""
        # Initialize containers for time series data
        time_series_states = []
        actions = []
        next_time_series_states = []
        dt_factors = []  # To store the time difference ratio

        # Environment timestep (target)
        env_dt = 0.01  # 100Hz - standard timestep

        # Debug counters
        total_episodes = len(log_data)
        processed_episodes = 0
        skipped_episodes = 0

        print("Sample state format:")
        if log_data and "states" in log_data[0] and log_data[0]["states"]:
            sample_state = log_data[0]["states"][0]
            for key, value in sample_state.items():
                print(f"  {key}: {value}")

        # Process each episode
        for episode_idx, episode in enumerate(log_data):
            print(f"Processing episode {episode_idx+1}/{total_episodes}")

            # Check if episode has states
            if "states" not in episode or not episode["states"]:
                print(f"  Episode {episode_idx+1} has no states")
                skipped_episodes += 1
                continue

            episode_states = episode["states"]
            print(f"  Episode has {len(episode_states)} states")

            # Skip episodes that are too short for time series
            if len(episode_states) <= self.time_steps:
                print(f"  Skipping episode {episode_idx+1}: too short ({len(episode_states)} < {self.time_steps})")
                skipped_episodes += 1
                continue

            # Verify key fields exist in states
            required_fields = ["theta_global", "theta_dot", "motor_pwm", "timestamp"]
            if not all(field in episode_states[0] for field in required_fields):
                missing = [f for f in required_fields if f not in episode_states[0]]
                print(f"  Episode {episode_idx+1} missing required fields: {missing}")
                continue

            # Pre-validate all timestamps in the episode to ensure they're monotonically increasing
            valid_timestamps = True
            for i in range(1, len(episode_states)):
                time_diff = (episode_states[i]["timestamp"] - episode_states[i - 1]["timestamp"]) / 1000.0
                if time_diff <= 0:
                    print(f"  Invalid time difference at index {i}: {time_diff} seconds")
                    valid_timestamps = False
                    break

            if not valid_timestamps:
                print(f"  Skipping episode {episode_idx+1} due to invalid timestamps")
                continue

            # Keep track of valid transitions for this episode
            episode_transitions = 0

            # Create sliding windows of time_steps length
            for i in range(len(episode_states) - self.time_steps):
                # Current window
                current_window = episode_states[i : i + self.time_steps]
                # Next window (shifted by one step)
                next_window = episode_states[i + 1 : i + self.time_steps + 1]

                # Check all required fields exist in all states
                valid_window = True
                for state in current_window + next_window:
                    if not all(field in state for field in required_fields):
                        valid_window = False
                        break

                if not valid_window:
                    continue

                # Calculate average time difference and dt_factor across all pairs in the window
                time_diffs = []
                for j in range(self.time_steps - 1):
                    time_diff = (current_window[j + 1]["timestamp"] - current_window[j]["timestamp"]) / 1000.0
                    time_diffs.append(time_diff)
                # Also include transition between windows
                time_diff = (next_window[0]["timestamp"] - current_window[-1]["timestamp"]) / 1000.0
                time_diffs.append(time_diff)

                # Use average dt_factor for adjustment
                avg_time_diff = sum(time_diffs) / len(time_diffs)
                dt_factor = avg_time_diff / env_dt

                # Build time series state from current window
                try:
                    current_time_series = []
                    for state in current_window:
                        current_time_series.extend(
                            [
                                float(state["theta_global"]),
                                float(state["theta_dot"]),
                                float(state["motor_pwm"]) / 127.0,  # Normalize to [-1, 1]
                            ]
                        )

                    # Extract states from next window
                    next_time_series_raw = []
                    for state in next_window:
                        next_time_series_raw.extend(
                            [float(state["theta_global"]), float(state["theta_dot"]), float(state["motor_pwm"]) / 127.0]
                        )
                except (ValueError, TypeError) as e:
                    print(f"  Error converting state values: {e}")
                    continue

                # Calculate state changes
                state_changes = np.array(next_time_series_raw) - np.array(current_time_series)

                # Adjust physical state components based on dt_factor
                adjusted_next_time_series = np.copy(current_time_series)
                for j in range(0, len(current_time_series), self.single_state_dim):
                    # Only adjust physical variables, not motor commands
                    adjusted_next_time_series[j : j + 2] += state_changes[j : j + 2] / dt_factor
                    # Motor commands aren't adjusted
                    adjusted_next_time_series[j + 2] = next_time_series_raw[j + 2]

                # Action is the motor command from the most recent state
                action = np.array([float(current_window[-1]["motor_pwm"]) / 127.0])

                # Store the transition
                time_series_states.append(current_time_series)
                actions.append(action)
                next_time_series_states.append(adjusted_next_time_series)
                dt_factors.append(dt_factor)

                episode_transitions += 1

            if episode_transitions > 0:
                print(f"  Collected {episode_transitions} transitions from episode {episode_idx+1}")
                processed_episodes += 1
            else:
                print(f"  No valid transitions found in episode {episode_idx+1}")

        # Convert to arrays
        if len(time_series_states) > 0:
            time_series_states = np.array(time_series_states)
            actions = np.array(actions)
            next_time_series_states = np.array(next_time_series_states)
            dt_factors = np.array(dt_factors)

            # Log statistics
            print(f"\nData Processing Summary:")
            print(f"Total episodes: {total_episodes}")
            print(f"Processed episodes: {processed_episodes}")
            print(f"Skipped episodes: {skipped_episodes}")
            print(f"Collected transitions: {len(time_series_states)}")
            print(f"Time series state shape: {time_series_states.shape}")
            print(f"Time difference statistics:")
            print(
                f"  Average real dt: {np.mean(dt_factors * env_dt):.4f}s ({1.0/(np.mean(dt_factors * env_dt)):.1f}Hz)"
            )
            print(f"  Min dt factor: {np.min(dt_factors):.2f}, Max dt factor: {np.max(dt_factors):.2f}")

            # Split into train/validation
            val_split = self.config["data_collection"]["validation_split"]
            num_samples = len(time_series_states)
            num_val = max(1, int(num_samples * val_split)) if num_samples > 1 else 0

            indices = np.random.permutation(num_samples)
            val_indices = indices[:num_val]
            train_indices = indices[num_val:]

            train_data = {
                "states": (
                    time_series_states[train_indices]
                    if train_indices.size > 0
                    else np.zeros((0, self.time_steps * self.single_state_dim))
                ),
                "actions": actions[train_indices] if train_indices.size > 0 else np.zeros((0, 1)),
                "next_states": (
                    next_time_series_states[train_indices]
                    if train_indices.size > 0
                    else np.zeros((0, self.time_steps * self.single_state_dim))
                ),
            }

            val_data = {
                "states": (
                    time_series_states[val_indices]
                    if val_indices.size > 0
                    else np.zeros((0, self.time_steps * self.single_state_dim))
                ),
                "actions": actions[val_indices] if val_indices.size > 0 else np.zeros((0, 1)),
                "next_states": (
                    next_time_series_states[val_indices]
                    if val_indices.size > 0
                    else np.zeros((0, self.time_steps * self.single_state_dim))
                ),
            }
        else:
            print("\nWARNING: No valid transitions found in log data!")
            empty_state = np.zeros((0, self.time_steps * self.single_state_dim))
            empty_action = np.zeros((0, 1))

            train_data = {
                "states": empty_state,
                "actions": empty_action,
                "next_states": empty_state,
            }

            val_data = {
                "states": empty_state,
                "actions": empty_action,
                "next_states": empty_state,
            }

        return train_data, val_data

    @staticmethod
    def calculate_class_weights(states: np.ndarray, num_bins: int = 10, strategy: str = "uniform") -> np.ndarray:
        """Calculate class weights for balanced sampling based on state distribution."""
        # Handle empty array
        if len(states) == 0:
            return np.array([])

        # Only use the first two dimensions (theta and theta_dot) for binning
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
        """Create a balanced dataset using oversampling of minority classes.

        Args:
            data: Dictionary containing training data
            num_bins: Number of bins for discretizing the state space
            strategy: Binning strategy ('uniform', 'quantile', or 'kmeans')

        Returns:
            Dictionary containing balanced training data
        """
        states = data["states"]
        actions = data["actions"]
        next_states = data["next_states"]

        # Create discretizer
        discretizer = KBinsDiscretizer(n_bins=num_bins, encode="ordinal", strategy=strategy)

        # Discretize states to identify classes
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
        """Remove large standard deviation samples from data.

        Args:
            data: Dictionary containing training data
            std_threshold_multiplier: Multiplier for standard deviation threshold

        Returns:
            Dictionary containing cleaned training data
        """
        states = data["states"]
        actions = data["actions"]
        next_states = data["next_states"]

        # Print shape info for debugging
        print(f"States shape: {states.shape}")
        print(f"Actions shape: {actions.shape}")
        print(f"Next states shape: {next_states.shape}")

        # Check dimensions of states array
        if len(states.shape) == 1:
            print("Warning: States array is 1D, cannot process outliers")
            return data  # Return original data unchanged

        # Calculate standard deviation and mean
        state_std = np.std(states, axis=0)
        state_mean = np.mean(states, axis=0)

        # Define threshold for outliers
        threshold = std_threshold_multiplier * state_std

        # Identify outliers - any feature value that exceeds its threshold
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

        print(f"Removed {np.sum(outliers)} outliers from {len(states)} samples")
        return cleaned_data

    @staticmethod
    def prepare_weighted_batch_indices(data_size: int, batch_size: int, weights: np.ndarray) -> List[np.ndarray]:
        """Prepare batch indices with weighted sampling.

        Args:
            data_size: Size of the dataset
            batch_size: Size of each batch
            weights: Sample weights for weighting the distribution

        Returns:
            List of batches with indices
        """
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
    ) -> Dict[str, float]:
        """Train for one epoch with time series data."""
        self.simnet.train()
        total_loss = 0
        total_weighted_samples = 0

        # Pull out arrays for convenience
        states_arr = train_data["states"]
        actions_arr = train_data["actions"]
        next_states_arr = train_data["next_states"]

        num_samples = len(states_arr)

        # Handle empty dataset or not enough samples for a batch
        if num_samples < batch_size:
            print(f"Warning: Not enough samples for a batch. Have {num_samples}, need {batch_size}")
            return {
                "train_loss": 0.0,
                "extreme_samples_percent": 0.0,
            }

        num_batches = num_samples // batch_size

        # Handle case where num_batches is 0
        if num_batches == 0:
            print("Warning: Calculated 0 batches, using 1 instead")
            num_batches = 1

        for i in range(num_batches):
            idx = slice(i * batch_size, (i + 1) * batch_size)
            states = torch.FloatTensor(states_arr[idx]).to(self.device)
            actions = torch.FloatTensor(actions_arr[idx]).to(self.device)
            target_next_states = torch.FloatTensor(next_states_arr[idx]).to(self.device)

            # Forward pass with SimNet
            pred_next_states = self.simnet(states, actions)

            # Calculate sample weights based on thresholds if provided
            weights = torch.ones(states.shape[0], device=self.device)

            if class_balancing_thresholds is not None:
                # Get most recent state from time series
                recent_states = states[:, -self.single_state_dim :]

                # Convert angles from radians to degrees
                angles_degrees = torch.abs(recent_states[:, 0] * 180.0 / np.pi)
                angular_vels = torch.abs(recent_states[:, 1])

                # Apply higher weights for challenging samples
                extreme_samples = (
                    (angles_degrees > class_balancing_thresholds["angle_deg"])
                    | (angular_vels > class_balancing_thresholds["angular_velocity_dps"])
                    | (
                        (angles_degrees > class_balancing_thresholds["motor_angle_deg"])
                        & (torch.abs(actions).squeeze() < class_balancing_thresholds["max_abs_action"])
                    )
                )

                # Ensure boolean tensor format
                extreme_samples = extreme_samples.bool()

                # Adjust weights
                weights[~extreme_samples] = 0.3  # Decrease weight for non-extreme samples

                # Count weighted samples for logging
                total_weighted_samples += extreme_samples.sum().item()

            # Compute MSE loss focused on physical state variables
            # Extract only the newest physical state from both predictions and targets
            pred_newest_physics = pred_next_states[:, -self.single_state_dim :][:, :2]
            target_newest_physics = target_next_states[:, -self.single_state_dim :][:, :2]

            # Weighted MSE loss
            squared_errors = (pred_newest_physics - target_newest_physics) ** 2
            weighted_errors = weights.unsqueeze(1) * squared_errors
            loss = weighted_errors.mean()

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return {
            "train_loss": total_loss / max(1, num_batches),  # Avoid division by zero
            "extreme_samples_percent": (total_weighted_samples / max(1, num_samples)) * 100,  # Avoid division by zero
        }

    def validate(self, val_data: Dict[str, np.ndarray], batch_size: int) -> Dict[str, float]:
        """Validate the model on time series data."""
        self.simnet.eval()
        total_loss = 0

        states_arr = val_data["states"]
        actions_arr = val_data["actions"]
        next_states_arr = val_data["next_states"]

        num_samples = len(states_arr)

        # Handle empty validation set
        if num_samples == 0:
            print("Warning: Empty validation set")
            return {"val_loss": 0.0}

        # Adjust batch size if needed
        if num_samples < batch_size:
            print(f"Warning: Not enough validation samples for a batch. Have {num_samples}, need {batch_size}")
            batch_size = max(1, num_samples)

        num_batches = max(1, num_samples // batch_size)  # Ensure at least 1 batch

        with torch.no_grad():
            for i in range(num_batches):
                idx = slice(i * batch_size, (i + 1) * batch_size)
                states = torch.FloatTensor(states_arr[idx]).to(self.device)
                actions = torch.FloatTensor(actions_arr[idx]).to(self.device)
                target_next_states = torch.FloatTensor(next_states_arr[idx]).to(self.device)

                # Forward pass
                pred_next_states = self.simnet(states, actions)

                # Compare only the physical variables (theta, theta_dot) of the newest state
                pred_newest_physics = pred_next_states[:, -self.single_state_dim :][:, :2]
                target_newest_physics = target_next_states[:, -self.single_state_dim :][:, :2]

                # Compute loss
                loss = torch.nn.functional.mse_loss(pred_newest_physics, target_newest_physics)
                total_loss += loss.item()

        return {"val_loss": total_loss / max(1, num_batches)}  # Avoid division by zero

    @staticmethod
    def analyze_class_distribution(
        data: Dict[str, np.ndarray], num_bins: int = 10, strategy: str = "uniform"
    ) -> Dict[str, Any]:
        """Analyze and visualize the class distribution in data.

        Args:
            data: Dictionary containing training data
            num_bins: Number of bins for discretizing the state space
            strategy: Binning strategy ('uniform', 'quantile', or 'kmeans')

        Returns:
            Dictionary with distribution statistics
        """
        states = data["states"]

        # Check if data is empty
        if len(states) == 0:
            print("Warning: Empty dataset provided to analyze_class_distribution")
            return {
                "angle_range": (0, 0),
                "angle_mean": 0,
                "angle_std": 0,
                "angular_vel_range": (0, 0),
                "angular_vel_mean": 0,
                "angular_vel_std": 0,
                "num_samples": 0,
                "min_weight": 0,
                "max_weight": 0,
                "mean_weight": 0,
                "std_weight": 0,
            }

        # Handle time series data format
        if states.shape[1] > 3:  # Time series data
            # Extract the most recent state from each time series
            single_state_dim = 3
            latest_states = np.array([s[-single_state_dim:] for s in states])

            # Calculate statistics on latest states
            angle_range = (np.min(latest_states[:, 0]), np.max(latest_states[:, 0]))
            angle_mean = np.mean(latest_states[:, 0])
            angle_std = np.std(latest_states[:, 0])

            angular_vel_range = (np.min(latest_states[:, 1]), np.max(latest_states[:, 1]))
            angular_vel_mean = np.mean(latest_states[:, 1])
            angular_vel_std = np.std(latest_states[:, 1])

            # Calculate sample weights
            sample_weights = SimNetTrainer.calculate_class_weights(latest_states, num_bins, strategy)
        else:
            # Regular state format
            angle_range = (np.min(states[:, 0]), np.max(states[:, 0]))
            angle_mean = np.mean(states[:, 0])
            angle_std = np.std(states[:, 0])

            angular_vel_range = (np.min(states[:, 1]), np.max(states[:, 1]))
            angular_vel_mean = np.mean(states[:, 1])
            angular_vel_std = np.std(states[:, 1])

            # Calculate sample weights
            sample_weights = SimNetTrainer.calculate_class_weights(states, num_bins, strategy)

        weight_stats = {
            "min_weight": np.min(sample_weights) if len(sample_weights) > 0 else 0,
            "max_weight": np.max(sample_weights) if len(sample_weights) > 0 else 0,
            "mean_weight": np.mean(sample_weights) if len(sample_weights) > 0 else 0,
            "std_weight": np.std(sample_weights) if len(sample_weights) > 0 else 0,
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
        """Train the SimNet model with time series data."""
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
            train_metrics = self.train_epoch(train_data, batch_size, class_balancing_thresholds)

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

        return logger.metrics if logger else {}
