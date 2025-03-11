import logging
import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from ..models import Actor, Critic, ReplayBuffer
from ..environment import BalancerEnv
from .utils import polyak_update, TrainingLogger, save_model, load_model
from tqdm import tqdm


class DDPGTrainer:
    """Trainer for DDPG algorithm."""

    def __init__(
        self,
        env: BalancerEnv,
        config_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize DDPG trainer.

        Args:
            env: Training environment
            config_path: Path to DDPG config file
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

        # Extract parameters from config
        model_config = self.config["model"]
        train_config = self.config["training"]

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            hidden_dims=model_config["actor"]["hidden_dims"],
        ).to(device)

        self.critic = Critic(
            state_dim=state_dim, action_dim=action_dim, hidden_dims=model_config["critic"]["hidden_dims"]
        ).to(device)

        # Create target networks
        self.actor_target = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            hidden_dims=model_config["actor"]["hidden_dims"],
        ).to(device)

        self.critic_target = Critic(
            state_dim=state_dim, action_dim=action_dim, hidden_dims=model_config["critic"]["hidden_dims"]
        ).to(device)

        # Copy weights
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=model_config["actor"]["learning_rate"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=model_config["critic"]["learning_rate"])

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(train_config["buffer_size"])

        # Training parameters
        self.gamma = train_config["gamma"]
        self.tau = train_config["tau"]
        self.action_noise = train_config["action_noise"]
        self.noise_decay = train_config["noise_decay"]
        self.min_noise = train_config["min_noise"]

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.training_steps = 0

    def _get_default_config(self) -> Dict:
        """Return default configuration if none provided."""
        return {
            "model": {
                "actor": {"hidden_dims": [8, 8], "learning_rate": 1e-4},
                "critic": {"hidden_dims": [256, 256], "learning_rate": 3e-4},
            },
            "training": {
                "total_episodes": 2000,
                "max_steps_per_episode": 500,
                "batch_size": 512,
                "gamma": 0.99,
                "tau": 0.005,
                "buffer_size": 1000000,
                "min_buffer_size": 5000,
                "action_noise": 0.1,
                "noise_decay": 0.9999,
                "min_noise": 0.01,
                "eval_frequency": 10,
                "eval_episodes": 5,
                "save_frequency": 100,
            },
        }

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using current policy."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state).cpu().numpy().flatten()

            if training:
                current_noise = max(self.action_noise * (self.noise_decay**self.training_steps), self.min_noise)
                noise = np.random.normal(0, current_noise, size=action.shape)
                action = np.clip(action + noise, -1.0, 1.0)

        return action

    def train_step(self, batch_size: int) -> Dict[str, float]:
        """Perform one training step."""
        self.training_steps += 1

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Update critic
        next_action = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_action)
        target_Q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_Q

        current_Q = self.critic(states, actions)
        critic_loss = torch.nn.functional.mse_loss(current_Q, target_Q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        polyak_update(self.actor_target, self.actor, self.tau)
        polyak_update(self.critic_target, self.critic, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_value": current_Q.mean().item(),
            "action_noise": self.action_noise * (self.noise_decay**self.training_steps),
        }

    def train(
        self,
        num_episodes: int,
        max_steps: int,
        batch_size: int,
        eval_freq: int = 10,
        save_freq: int = 100,
        log_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train the agent.

        Args:
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            batch_size: Batch size for training
            eval_freq: Episodes between evaluations
            save_freq: Episodes between saving models
            log_dir: Directory for saving logs and models

        Returns:
            Dictionary containing training history
        """
        logger = TrainingLogger(log_dir) if log_dir else None
        best_reward = float("-inf")

        progress_bar = tqdm(range(num_episodes), desc="Training", position=0, leave=True)

        for episode in progress_bar:
            state, _ = self.env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)

                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)

                episode_reward += reward
                state = next_state

                # Train if enough samples
                if len(self.replay_buffer) > batch_size:
                    metrics = self.train_step(batch_size)
                    if logger:
                        logger.log(metrics)

                # Stop episode if reached stable state
                if info.get("reached_stable", False):
                    break

                if done:
                    break

            if logger:
                logger.log({"episode_reward": episode_reward, "episode_length": step + 1})

            # Evaluation
            if (episode + 1) % eval_freq == 0:
                eval_reward = self.evaluate()
                if logger:
                    logger.log({"eval_reward": eval_reward, "episode_length": step + 1})

                progress_bar.set_postfix({"episode": episode + 1, "eval_reward": f"{eval_reward:.2f}"})

                # Save best model
                if eval_reward > best_reward and log_dir:
                    best_reward = eval_reward
                    save_model(self.actor, Path(log_dir) / "best_actor.pt", {"episode": episode, "reward": eval_reward})
                    save_model(
                        self.critic, Path(log_dir) / "best_critic.pt", {"episode": episode, "reward": eval_reward}
                    )

            # Regular saving
            if log_dir and (episode + 1) % save_freq == 0:
                save_model(self.actor, Path(log_dir) / f"actor_episode_{episode+1}.pt", {"episode": episode})
                save_model(self.critic, Path(log_dir) / f"critic_episode_{episode+1}.pt", {"episode": episode})

        if logger:
            logger.save()

        # Final model saving
            if log_dir:
                save_model(self.actor, Path(log_dir) / "actor_final.pt")
                save_model(self.critic, Path(log_dir) / "critic_final.pt")

        return logger.metrics if logger else {}

    def evaluate(self, num_episodes: int = 5, max_steps: int = 500) -> float:
        """Evaluate current policy."""
        total_reward = 0
        counter = 0

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done and counter < max_steps:
                counter += 1
                action = self.select_action(state, training=False)
                state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward

            total_reward += episode_reward

        return total_reward / num_episodes

    def print_model_info(self):
        print(self.actor)
        print(self.critic)
