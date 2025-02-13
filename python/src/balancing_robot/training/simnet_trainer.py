import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
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

        # Set up logging
        self.logger = logging.getLogger(__name__)

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
        """Collect data from physics-based simulation."""
        config = self.config["data_collection"]
        num_samples = config["physics_samples"]
        noise_std = config["noise_std"]

        states, actions, accels = [], [], []

        state = self.env.reset()
        for _ in tqdm(range(num_samples), desc="Collecting physics data"):
            # Random action with noise
            action = np.random.uniform(-1, 1, size=self.env.action_space.shape)
            action = np.clip(action + np.random.normal(0, noise_std), -1, 1)

            # Get accelerations from physics
            accel = self.env.physics.get_acceleration(state, action)

            states.append(state.copy())
            actions.append(action)
            accels.append(accel)

            # Step environment
            state, _, done, _ = self.env.step(action)
            if done:
                state = self.env.reset()

        # Convert to arrays
        data = {"states": np.array(states), "actions": np.array(actions), "accelerations": np.array(accels)}

        # Split train/val
        num_samples = len(states)
        num_val = int(num_samples * config["validation_split"])
        indices = np.random.permutation(num_samples)

        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        train_data = {k: v[train_indices] for k, v in data.items()}
        val_data = {k: v[val_indices] for k, v in data.items()}

        return train_data, val_data

    def process_real_data(self, log_data: List[Dict[str, Any]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Process real robot log data for training.

        Args:
            log_data: List of logged data dictionaries

        Returns:
            Tuple of (train_data, val_data) dictionaries
        """
        states, actions, accels = [], [], []

        for entry in log_data:
            # Extract state
            state = np.array(
                [entry["theta"], entry["theta_dot"], entry["x"], entry["x_dot"], entry["phi"], entry["phi_dot"]]
            )

            # Extract action and scale to [-1, 1]
            action = np.array([entry["motor_pwm"]]) / 127.0

            # Calculate accelerations from consecutive states
            if len(states) > 0:
                dt = entry["dt"]
                prev_state = states[-1]

                theta_ddot = (state[1] - prev_state[1]) / dt  # theta_dot difference
                x_ddot = (state[3] - prev_state[3]) / dt  # x_dot difference
                phi_ddot = (state[5] - prev_state[5]) / dt  # phi_dot difference

                accels.append([theta_ddot, x_ddot, phi_ddot])
                states.append(prev_state)
                actions.append(action)

        # Convert to arrays
        data = {"states": np.array(states), "actions": np.array(actions), "accelerations": np.array(accels)}

        # Split into train/val
        num_samples = len(states)
        num_val = int(num_samples * self.validation_split)
        indices = np.random.permutation(num_samples)
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        train_data = {k: v[train_indices] for k, v in data.items()}
        val_data = {k: v[val_indices] for k, v in data.items()}

        return train_data, val_data

    def train_epoch(
        self, train_data: Dict[str, np.ndarray], batch_size: int, is_finetuning: bool = False
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.simnet.train()
        total_loss = 0
        num_batches = len(train_data["states"]) // batch_size

        for i in range(num_batches):
            # Get batch
            idx = slice(i * batch_size, (i + 1) * batch_size)
            states = torch.FloatTensor(train_data["states"][idx]).to(self.device)
            actions = torch.FloatTensor(train_data["actions"][idx]).to(self.device)
            target_accels = torch.FloatTensor(train_data["accelerations"][idx]).to(self.device)

            # Forward pass
            pred_accels = self.simnet(states, actions)
            loss = torch.nn.functional.mse_loss(pred_accels, target_accels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return {"train_loss": total_loss / num_batches}

    def validate(self, val_data: Dict[str, np.ndarray], batch_size: int) -> Dict[str, float]:
        """Validate model."""
        self.simnet.eval()
        total_loss = 0
        num_batches = len(val_data["states"]) // batch_size

        with torch.no_grad():
            for i in range(num_batches):
                idx = slice(i * batch_size, (i + 1) * batch_size)
                states = torch.FloatTensor(val_data["states"][idx]).to(self.device)
                actions = torch.FloatTensor(val_data["actions"][idx]).to(self.device)
                target_accels = torch.FloatTensor(val_data["accelerations"][idx]).to(self.device)

                pred_accels = self.simnet(states, actions)
                loss = torch.nn.functional.mse_loss(pred_accels, target_accels)
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
            train_metrics = self.train_epoch(train_data, batch_size, is_finetuning)

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
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.6f} - "
                f"Val Loss: {val_metrics['val_loss']:.6f}"
            )

        if logger:
            logger.save()

        return logger.metrics if logger else {}
