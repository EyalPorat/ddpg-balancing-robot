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

        # Initialize SimNet from config
        model_config = self.config["model"]
        self.simnet = SimNet(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_dims=model_config["hidden_dims"],
        ).to(device)

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.simnet.parameters())

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
                "hidden_dims": [128, 128],
                "learning_rate": 0.001,
                "dropout_rate": 0.1,
                "activation": "relu",
                "layer_norm": True,
            },
            "data_collection": {
                "physics_samples": 100000,
                "real_samples": 10000,
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
        states, actions, next_states = [], [], []

        # Decide how many steps to record per episode
        steps_per_episode = 500

        # Compute number of episodes for each type
        # We'll have equal numbers of no-action and random-action episodes
        total_episodes = num_samples // steps_per_episode
        no_action_episodes = total_episodes // 3
        random_action_episodes = total_episodes - no_action_episodes

        # (1) "No action" episodes
        for _ in tqdm(range(no_action_episodes), desc="Collecting no-action physics data"):
            # Reset environment
            state, _ = self.env.reset(should_zero_previous_action=True)

            # Add initial observation noise only to physical components (theta, theta_dot)
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
            state = state + np.random.normal(0, observation_noise_std, size=state.shape)

            for _ in range(steps_per_episode):
                s_t = state.copy()

                # Sample random action
                a_t = np.random.uniform(-1, 1, size=self.env.action_space.shape)

                # Add action noise to increase exploration
                a_t = np.clip(a_t + np.random.normal(0, action_noise_std, size=self.env.action_space.shape), -1, 1)

                next_state, _, done, _, _ = self.env.step(a_t, action_as_actual_output=True)

                # Add observation noise to next_state (only to physical components)
                next_state[:2] = next_state[:2] + np.random.normal(0, observation_noise_std, size=2)

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
        """Process real robot log data for training.
        Args:
            log_data: List of logged data dictionaries
        Returns:
            Tuple of (train_data, val_data) dictionaries
        """
        states = []
        actions = []
        next_states = []
        dt_factors = []  # To store the time difference ratio
        # Environment timestep (target)
        env_dt = 0.01  # 100Hz - the timestep used in simulation and control

        # For each episode in the log_data
        for episode in log_data:
            episode_states = episode["states"]

            # Process state transitions within each episode
            for i in range(len(episode_states) - 1):
                # Extract current state
                current = episode_states[i]
                # Extract next state
                next_state = episode_states[i + 1]

                # Calculate the real time difference between states
                time_diff = (next_state["timestamp"] - current["timestamp"]) / 1000.0  # Convert ms to seconds

                # Skip invalid time differences (e.g., if there's a logging error causing negative time)
                if time_diff <= 0:
                    continue

                # Calculate the ratio between real time difference and environment timestep
                dt_factor = time_diff / env_dt

                # Skip if the time difference is too large (indicating potential gap in data)
                if dt_factor > 10.0:  # Arbitrary threshold, adjust as needed
                    continue

                # Get the previous motor command for current state
                # For the first state in the sequence, we can't know what came before
                # If i > 0, use the motor command from the previous timestep
                # Otherwise, use the current motor command (not ideal but better than nothing)
                if i > 0:
                    prev_motor_command = float(episode_states[i - 1]["motor_pwm"]) / 127.0  # Normalize to [-1, 1]
                else:
                    # For the first state, we don't have a previous command
                    # We could either skip this state or use the current command as an approximation
                    prev_motor_command = float(current["motor_pwm"]) / 127.0  # This is a compromise

                # Current state includes theta, theta_dot, and PREVIOUS motor command
                state = np.array([current["theta"], current["theta_dot"], prev_motor_command])

                # Current action is the motor command applied at the current timestep
                action = np.array([current["motor_pwm"]]) / 127.0  # Normalize to [-1, 1]

                # Next state includes next theta, next theta_dot, and CURRENT motor command
                # (which becomes the "previous" command for the next timestep)
                current_motor_command = float(current["motor_pwm"]) / 127.0
                next_state_array = np.array([next_state["theta"], next_state["theta_dot"], current_motor_command])

                states.append(state)
                actions.append(action)
                next_states.append(next_state_array)
                dt_factors.append(dt_factor)

        # Convert to arrays
        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        dt_factors = np.array(dt_factors)

        # Adjust next_states based on dt_factors
        if len(states) > 0:
            # Calculate expected state change per environment timestep
            state_changes = next_states - states

            # Scale down the state changes by the dt_factor to get consistent rate of change
            # Only adjust the physical state components (theta, theta_dot), not the motor command
            adjusted_next_states = np.copy(states)
            adjusted_next_states[:, :2] += state_changes[:, :2] / dt_factors[:, np.newaxis]

            # Keep the motor command update as is (it's not affected by dt)
            adjusted_next_states[:, 2] = next_states[:, 2]

            # Log statistics about time differences
            print(f"Time difference statistics:")
            print(
                f"  Average real dt: {np.mean(dt_factors * env_dt):.4f}s ({1.0/(np.mean(dt_factors * env_dt)):.1f}Hz)"
            )
            print(f"  Min dt factor: {np.min(dt_factors):.2f}, Max dt factor: {np.max(dt_factors):.2f}")
            print(f"  Adjustment applied to {len(states)} state transitions")

            # Update next_states with adjusted values
            next_states = adjusted_next_states

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
        """Calculate class weights for balanced sampling based on state distribution.

        Args:
            states: Array of states to classify
            num_bins: Number of bins for discretizing the state space
            strategy: Binning strategy ('uniform', 'quantile', or 'kmeans')

        Returns:
            Array of sample weights corresponding to each state
        """
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

    def train_epoch(self, train_data: Dict[str, np.ndarray], batch_size: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.simnet.train()
        total_loss = 0

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

            # Predict next state
            pred_next_states = self.simnet(states, actions)

            # Compute MSE
            loss = torch.nn.functional.mse_loss(pred_next_states, target_next_states)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return {"train_loss": total_loss / num_batches}

    def validate(self, val_data: Dict[str, np.ndarray], batch_size: int) -> Dict[str, float]:
        self.simnet.eval()
        total_loss = 0

        states_arr = val_data["states"]
        actions_arr = val_data["actions"]
        next_states_arr = val_data["next_states"]

        num_samples = len(states_arr)
        num_batches = num_samples // batch_size

        with torch.no_grad():
            for i in range(num_batches):
                idx = slice(i * batch_size, (i + 1) * batch_size)
                states = torch.FloatTensor(states_arr[idx]).to(self.device)
                actions = torch.FloatTensor(actions_arr[idx]).to(self.device)
                target_next_states = torch.FloatTensor(next_states_arr[idx]).to(self.device)

                pred_next_states = self.simnet(states, actions)
                loss = torch.nn.functional.mse_loss(pred_next_states, target_next_states)
                total_loss += loss.item()

        return {"val_loss": total_loss / num_batches}

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
            # Train
            train_metrics = self.train_epoch(train_data, batch_size)

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
