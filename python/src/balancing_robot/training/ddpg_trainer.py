import logging
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from ..models import Actor, Critic, ReplayBuffer, PrioritizedReplayBuffer
from ..environment import BalancerEnv
from .utils import polyak_update, TrainingLogger, save_model, load_model
from tqdm import tqdm


class DDPGTrainer:
    """Enhanced DDPG trainer with separate prioritized replay buffers for actor and critic."""

    def __init__(
        self,
        env: BalancerEnv,
        config_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize DDPG trainer with dual prioritized buffers.

        Args:
            env: Training environment
            config_path: Path to DDPG config file
            device: Device to use for training
        """
        # Initialize evaluation tracking
        self.last_eval_reward = 0.0
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

        # We always use max_action=1.0 for the actor
        max_action = 1.0

        # Note: For delta-based control, the actor outputs changes in the range [-1, 1]
        # which are then scaled by the environment using max_delta parameter
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            hidden_dims=model_config["actor"]["hidden_dims"],
        ).to(device)

        self.critic = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=model_config["critic"]["hidden_dims"],
        ).to(device)

        # Create target networks
        self.actor_target = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            hidden_dims=model_config["actor"]["hidden_dims"],
        ).to(device)

        self.critic_target = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=model_config["critic"]["hidden_dims"],
        ).to(device)

        # Copy weights
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=model_config["actor"]["learning_rate"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=model_config["critic"]["learning_rate"])

        # Initialize dual prioritized replay buffers
        buffer_size = train_config["buffer_size"]
        self.critic_buffer = PrioritizedReplayBuffer(buffer_size, alpha=train_config.get("critic_alpha", 0.6))
        self.actor_buffer = PrioritizedReplayBuffer(buffer_size, alpha=train_config.get("actor_alpha", 0.6))

        # Buffer mode flags
        self.use_critic_prioritization = train_config.get("use_critic_prioritization", True)
        self.use_actor_prioritization = train_config.get("use_actor_prioritization", True)
        self.shared_transitions = train_config.get(
            "shared_transitions", True
        )  # Whether to store transitions in both buffers

        # Training parameters
        self.gamma = train_config["gamma"]
        self.tau = train_config["tau"]
        self.action_noise = train_config["action_noise"]
        self.noise_decay = train_config["noise_decay"]
        self.min_noise = train_config["min_noise"]

        # Actor prioritization method (options: 'q_value', 'advantage', 'policy_gradient')
        self.actor_priority_method = train_config.get("actor_priority_method", "q_value")

        # Parameter for advantage calculation if using that method
        self.advantage_samples = train_config.get("advantage_samples", 10)

        # Beta annealing for importance sampling (starts lower, approaches 1)
        self.critic_beta_start = train_config.get("critic_beta_start", 0.4)
        self.actor_beta_start = train_config.get("actor_beta_start", 0.4)
        self.beta_anneal_steps = train_config.get("beta_anneal_steps", 50000)

        # Curriculum learning parameters
        self.use_curriculum = train_config.get("use_curriculum", True)  # Default to using curriculum
        self.curriculum_epochs = train_config.get("curriculum_epochs", 200)
        self.curriculum_initial_angle_range_precent = train_config.get("curriculum_initial_angle_range_precent", 0.2)
        self.curriculum_initial_angular_velocity_range_precent = train_config.get(
            "curriculum_initial_angular_velocity_range_precent", 0.2
        )

        # Default initialization ranges in the environment
        self.final_theta_range = (-50, 50)  # degrees
        self.final_theta_dot_range = (-150, 150)  # degrees per second

        # Initial (easier) ranges for curriculum learning
        self.initial_theta_range = (
            self.final_theta_range[0] * self.curriculum_initial_angle_range_precent,
            self.final_theta_range[1] * self.curriculum_initial_angle_range_precent,
        )
        self.initial_theta_dot_range = (
            self.final_theta_dot_range[0] * self.curriculum_initial_angular_velocity_range_precent,
            self.final_theta_dot_range[1] * self.curriculum_initial_angular_velocity_range_precent,
        )

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.training_steps = 0
        self.last_eval_reward = 0.0  # Track the most recent evaluation reward

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
                "use_curriculum": True,
                "curriculum_epochs": 200,
                "curriculum_initial_angle_range_precent": 0.2,
                "curriculum_initial_angular_velocity_range_precent": 0.2,
                "use_critic_prioritization": True,
                "use_actor_prioritization": True,
                "critic_alpha": 0.6,
                "actor_alpha": 0.6,
                "critic_beta_start": 0.4,
                "actor_beta_start": 0.4,
                "beta_anneal_steps": 50000,
                "actor_priority_method": "q_value",
                "advantage_samples": 10,
                "shared_transitions": True,
            },
        }

    def _calculate_curriculum_ranges(self, episode: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Calculate initialization ranges based on curriculum progress.

        As episodes progress, the ranges gradually increase from initial to final values.

        Args:
            episode: Current episode number

        Returns:
            Tuple of (theta_range, theta_dot_range) to use for initialization
        """
        if not self.use_curriculum or episode >= self.curriculum_epochs:
            # Use final ranges after curriculum is complete
            return self.final_theta_range, self.final_theta_dot_range

        # Calculate progress ratio (0.0 to 1.0)
        progress = min(1.0, episode / self.curriculum_epochs)

        # Linearly interpolate between initial and final ranges
        theta_min = self.initial_theta_range[0] + progress * (self.final_theta_range[0] - self.initial_theta_range[0])
        theta_max = self.initial_theta_range[1] + progress * (self.final_theta_range[1] - self.initial_theta_range[1])

        theta_dot_min = self.initial_theta_dot_range[0] + progress * (
            self.final_theta_dot_range[0] - self.initial_theta_dot_range[0]
        )
        theta_dot_max = self.initial_theta_dot_range[1] + progress * (
            self.final_theta_dot_range[1] - self.initial_theta_dot_range[1]
        )

        return (theta_min, theta_max), (theta_dot_min, theta_dot_max)

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

    def store_transition(
        self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool
    ):
        """Store transition in replay buffers.

        When shared_transitions is True, both buffers get the same data.
        Otherwise, we randomly distribute transitions between buffers.
        """
        # Always store in critic buffer
        self.critic_buffer.push(state, action, reward, next_state, done)

        if self.shared_transitions:
            # Store the same transition in actor buffer
            self.actor_buffer.push(state, action, reward, next_state, done)
        elif np.random.random() < 0.5:
            # Randomly distribute transitions (50/50) between buffers when not sharing
            self.actor_buffer.push(state, action, reward, next_state, done)

    def _calculate_actor_priorities(self, states: torch.Tensor, actor_actions: torch.Tensor) -> np.ndarray:
        """Calculate priorities for the actor replay buffer.

        Args:
            states: Batch of states
            actor_actions: Actions from the current policy for those states

        Returns:
            Array of priority values for each state-action pair
        """
        if self.actor_priority_method == "q_value":
            # Prioritize states where the actor's action has a low Q-value
            # (meaning there's a lot of room for improvement)
            with torch.no_grad():
                q_values = self.critic(states, actor_actions)
                # Invert Q-values - lower Q-values get higher priority
                priorities = -q_values.detach().cpu().numpy().squeeze()
                return priorities + 1e-6  # Add small constant for numerical stability

        elif self.actor_priority_method == "advantage":
            # Prioritize states where the advantage (current action vs random actions) is large
            with torch.no_grad():
                # Get Q-value for current policy's action
                policy_q = self.critic(states, actor_actions)

                # Sample random actions and get their Q-values
                random_qs = []
                for _ in range(self.advantage_samples):
                    random_actions = torch.FloatTensor(
                        np.random.uniform(-1, 1, size=(states.size(0), actor_actions.size(1)))
                    ).to(self.device)
                    random_q = self.critic(states, random_actions)
                    random_qs.append(random_q)

                # Calculate the mean Q-value of random actions
                mean_random_q = torch.stack(random_qs).mean(dim=0)

                # Calculate advantage (how much better is the policy than random)
                advantage = policy_q - mean_random_q

                # Use the absolute advantage for prioritization
                # Large absolute advantage means this state is important for learning
                priorities = torch.abs(advantage).detach().cpu().numpy().squeeze()
                return priorities + 1e-6

        elif self.actor_priority_method == "policy_gradient":
            # Prioritize states with large policy gradient magnitudes
            self.actor_optimizer.zero_grad()
            q_values = self.critic(states, actor_actions)
            actor_loss = -q_values.mean()
            actor_loss.backward(retain_graph=True)

            # Calculate the norm of gradients for each state-action pair
            # This is a bit tricky since the gradients are for the network parameters
            # We'll use a proxy based on the sensitivity of Q-values to actions
            with torch.no_grad():
                # Make a small perturbation to actions
                epsilon = 1e-3
                perturbed_actions = actor_actions.clone() + epsilon
                perturbed_actions = torch.clamp(perturbed_actions, -1.0, 1.0)

                # Get Q-values for perturbed actions
                perturbed_q = self.critic(states, perturbed_actions)

                # Sensitivity is approximated by the change in Q-value
                sensitivity = torch.abs(perturbed_q - q_values) / epsilon
                priorities = sensitivity.detach().cpu().numpy().squeeze()
                return priorities + 1e-6

        else:
            # Default to uniform priorities if method not recognized
            return np.ones(states.size(0))

    def train_step(self, batch_size: int) -> Dict[str, float]:
        """Perform one training step with dual prioritized experience replay."""
        self.training_steps += 1

        # Calculate current beta values for importance sampling
        critic_beta = min(
            1.0,
            self.critic_beta_start + (1.0 - self.critic_beta_start) * (self.training_steps / self.beta_anneal_steps),
        )
        actor_beta = min(
            1.0, self.actor_beta_start + (1.0 - self.actor_beta_start) * (self.training_steps / self.beta_anneal_steps)
        )

        # CRITIC UPDATE
        # Sample from critic replay buffer
        if self.use_critic_prioritization:
            critic_sample = self.critic_buffer.sample(batch_size=batch_size, beta=critic_beta)
            states, actions, rewards, next_states, dones, critic_weights, critic_indices = critic_sample
            critic_weights = torch.FloatTensor(critic_weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.critic_buffer.sample(batch_size=batch_size)
            critic_weights = torch.ones(batch_size).to(self.device)
            critic_indices = None  # No need for indices if not updating priorities

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_Q

        # Compute current Q-values and TD errors
        current_Q = self.critic(states, actions)
        td_errors = torch.abs(current_Q - target_Q)

        # Apply importance sampling weights to critic loss
        critic_loss = (critic_weights.unsqueeze(1) * F.mse_loss(current_Q, target_Q, reduction="none")).mean()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update critic priorities
        if self.use_critic_prioritization and critic_indices is not None:
            new_critic_priorities = td_errors.detach().cpu().numpy().squeeze() + 1e-6  # Small constant for stability
            self.critic_buffer.update_priorities(critic_indices, new_critic_priorities)

        # ACTOR UPDATE
        # Sample from actor replay buffer (which may be different from critic's sample)
        if self.use_actor_prioritization:
            actor_sample = self.actor_buffer.sample(batch_size=batch_size, beta=actor_beta)
            actor_states, actor_actions_old, _, _, _, actor_weights, actor_indices = actor_sample
            actor_states = torch.FloatTensor(actor_states).to(self.device)
            actor_weights = torch.FloatTensor(actor_weights).to(self.device)
        else:
            # If not using separate actor prioritization, use the same states as critic
            actor_states = states
            actor_weights = torch.ones(batch_size).to(self.device)
            actor_indices = None

        # Get current policy's actions for these states
        actor_actions = self.actor(actor_states)

        # Calculate actor loss with importance sampling weights
        actor_q_values = self.critic(actor_states, actor_actions)
        actor_loss = -(actor_weights * actor_q_values.squeeze()).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update actor priorities based on the chosen method
        if self.use_actor_prioritization and actor_indices is not None:
            new_actor_priorities = self._calculate_actor_priorities(actor_states, actor_actions)
            self.actor_buffer.update_priorities(actor_indices, new_actor_priorities)

        # Update target networks
        polyak_update(self.actor_target, self.actor, self.tau)
        polyak_update(self.critic_target, self.critic, self.tau)

        # Collect metrics for logging
        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q_value": float(current_Q.mean().item()),
            "action_noise": float(self.action_noise * (self.noise_decay**self.training_steps)),
            "critic_beta": float(critic_beta),
            "actor_beta": float(actor_beta),
        }

        if self.use_critic_prioritization:
            metrics.update(
                {
                    "critic_priority_mean": float(np.mean(new_critic_priorities)),
                    "critic_priority_max": float(np.max(new_critic_priorities)),
                }
            )

        if self.use_actor_prioritization:
            metrics.update(
                {
                    "actor_priority_mean": float(np.mean(new_actor_priorities)),
                    "actor_priority_max": float(np.max(new_actor_priorities)),
                }
            )

        return metrics

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

        # Get minimum buffer size for training
        min_buffer_size = self.config["training"]["min_buffer_size"]

        progress_bar = tqdm(range(num_episodes), desc="Training", position=0, leave=True)

        for episode in progress_bar:
            # Calculate initialization ranges based on curriculum
            theta_range, theta_dot_range = self._calculate_curriculum_ranges(episode)

            # Modify environment's random initialization ranges
            # We need to monkey-patch the environment's reset method for this episode
            original_reset = self.env.reset

            def curriculum_reset(seed=None, should_zero_previous_action=False):
                """Override environment reset to use curriculum ranges."""
                super_reset = original_reset

                # Reset with default parameters
                state, info = super_reset(seed, should_zero_previous_action)

                # Override the first two state components (theta and theta_dot)
                # with values sampled from our curriculum ranges
                theta_deg = self.env.np_random.uniform(theta_range[0], theta_range[1])
                theta_dot_dps = self.env.np_random.uniform(theta_dot_range[0], theta_dot_range[1])

                # Convert to radians and update state
                self.env.state = np.array(
                    [
                        np.deg2rad(theta_deg),
                        np.deg2rad(theta_dot_dps),
                        self.env.state[2],  # Keep the previous motor command unchanged
                    ]
                )

                return self.env.state, info

            # Apply the patched reset method
            self.env.reset = curriculum_reset

            # Reset environment with curriculum ranges
            state, _ = self.env.reset()

            # Restore original reset method after initialization
            self.env.reset = original_reset

            episode_reward = 0

            for step in range(max_steps):
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, done, _, info = self.env.step(action)

                # Store transition in both replay buffers
                self.store_transition(state, action, reward, next_state, done)

                episode_reward += reward
                state = next_state

                # Train if enough samples are available in both buffers
                min_buffer_filled = (
                    len(self.critic_buffer) >= min_buffer_size and len(self.actor_buffer) >= min_buffer_size
                )

                if min_buffer_filled:
                    metrics = self.train_step(batch_size)
                    if logger:
                        logger.log(metrics)

                if done:
                    break

            # Log curriculum parameters
            if logger:
                logger.log(
                    {
                        "episode_reward": episode_reward,
                        "episode_length": step + 1,
                        "curriculum_theta_min": theta_range[0],
                        "curriculum_theta_max": theta_range[1],
                        "curriculum_theta_dot_min": theta_dot_range[0],
                        "curriculum_theta_dot_max": theta_dot_range[1],
                        "curriculum_progress": (
                            min(1.0, episode / self.curriculum_epochs) if self.use_curriculum else 1.0
                        ),
                        "critic_buffer_size": len(self.critic_buffer),
                        "actor_buffer_size": len(self.actor_buffer),
                    }
                )

                progress_bar.set_postfix(
                    {
                        "episode": episode + 1,
                        "reward": f"{episode_reward:.2f}",
                        "eval": f"{self.last_eval_reward:.2f}" if hasattr(self, "last_eval_reward") else "N/A",
                        "curriculum": (
                            f"{min(100, int(episode * 100 / self.curriculum_epochs))}%"
                            if self.use_curriculum and episode < self.curriculum_epochs
                            else "done"
                        ),
                        "actor_loss": (
                            f"{logger.get_latest('actor_loss'):.4f}"
                            if logger and logger.get_latest("actor_loss") != 0
                            else "0"
                        ),
                        "critic_loss": (
                            f"{logger.get_latest('critic_loss'):.4f}"
                            if logger and logger.get_latest("critic_loss") != 0
                            else "0"
                        ),
                    }
                )

            # Save best model
            if episode_reward > best_reward and log_dir:
                best_reward = episode_reward
                save_model(
                    self.actor,
                    Path(log_dir) / "best_actor.pt",
                    {"episode": int(episode), "reward": float(episode_reward)},
                )
                save_model(
                    self.critic,
                    Path(log_dir) / "best_critic.pt",
                    {"episode": int(episode), "reward": float(episode_reward)},
                )

            # Regular saving
            if log_dir and (episode + 1) % save_freq == 0:
                save_model(self.actor, Path(log_dir) / f"actor_episode_{episode+1}.pt", {"episode": int(episode)})
                save_model(self.critic, Path(log_dir) / f"critic_episode_{episode+1}.pt", {"episode": int(episode)})

            # Evaluate periodically
            if (episode + 1) % eval_freq == 0:
                eval_reward = self.evaluate(num_episodes=5, max_steps=max_steps)
                # Store the latest evaluation reward
                self.last_eval_reward = eval_reward

                if logger:
                    logger.log({"eval_reward": eval_reward, "episode": episode})

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

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            counter = 0
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
