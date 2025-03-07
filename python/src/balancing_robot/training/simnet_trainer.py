import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

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
        self.optimizer = torch.optim.Adam(self.simnet.parameters(), lr=model_config["learning_rate"])

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
        4. Not breaking on 'done' so we gather the full episode
        """
        config = self.config["data_collection"]
        num_samples = config["physics_samples"]
        noise_std = config["noise_std"]
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
            state, _ = self.env.reset()

            for _ in range(steps_per_episode):
                s_t = state.copy()
                # Zero action for all motors, letting it fall
                a_t = np.zeros(self.env.action_space.shape, dtype=np.float32)
                next_state, _, done, _, _ = self.env.step(a_t)

                states.append(s_t)
                actions.append(a_t)
                next_states.append(next_state.copy())

                # Keep going even if done == True (we are ignoring terminal conditions)
                state = next_state

        # (2) Random actions episodes
        for _ in tqdm(range(random_action_episodes), desc="Collecting random-action physics data"):
            # Reset environment
            state, _ = self.env.reset()

            for _ in range(steps_per_episode):
                s_t = state.copy()
                # Sample random action with noise
                a_t = np.random.uniform(-1, 1, size=self.env.action_space.shape)
                a_t = np.clip(a_t + np.random.normal(0, noise_std, size=self.env.action_space.shape), -1, 1)
                next_state, _, done, _, _ = self.env.step(a_t)

                states.append(s_t)
                actions.append(a_t)
                next_states.append(next_state.copy())

                # do not break on done
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
        states, actions = [], []

        for entry in log_data:
            # Extract state
            state = np.array(
                [entry["theta"], entry["theta_dot"], entry["x"], entry["x_dot"], entry["phi"], entry["phi_dot"]]
            )

            # Extract action and scale to [-1, 1]
            action = np.array([entry["motor_pwm"]]) / 127.0

            if len(states) > 0:
                states.append(state)
                actions.append(action)

        # Convert to arrays
        data = {"states": np.array(states), "actions": np.array(actions)}

        # Split into train/val
        num_samples = len(states)
        num_val = int(num_samples * self.validation_split)
        indices = np.random.permutation(num_samples)
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        train_data = {k: v[train_indices] for k, v in data.items()}
        val_data = {k: v[val_indices] for k, v in data.items()}

        return train_data, val_data

    def train_epoch(self, train_data: Dict[str, np.ndarray], batch_size: int) -> Dict[str, float]:
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
