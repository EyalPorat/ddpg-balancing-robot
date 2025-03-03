import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional

from .physics import PhysicsEngine, PhysicsParams
from ..models.simnet import SimNet


class BalancerEnv(gym.Env):
    """Gymnasium environment for a two-wheeled balancing robot."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        physics_params: Optional[PhysicsParams] = None,
        config_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        simnet: Optional[SimNet] = None,
    ):
        """Initialize environment.

        Args:
            physics_params: Custom physics parameters
            simnet: Optional SimNet model for dynamics prediction
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()

        # Load config if provided
        if config_path:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = None

        self.physics = PhysicsEngine(physics_params or PhysicsParams(config_path))
        self.simnet = simnet
        self.max_steps = self.config["termination"]["max_steps"] if self.config else 500
        self.render_mode = render_mode

        # Define action space (motor torque)
        # Maximum torque in N⋅m
        if self.config:
            self.max_torque = self.config["physics"]["max_torque"]
        else:
            self.max_torque = 0.23
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Define observation space
        # [theta, theta_dot, x, x_dot, phi, phi_dot]
        if self.config:
            obs_config = self.config["observation"]
            obs_high = np.array(
                [
                    obs_config["angle_limit"],
                    obs_config["angular_velocity_limit"],
                    float(obs_config["position_limit"]),
                    obs_config["velocity_limit"],
                    float(obs_config["wheel_angle_limit"]),
                    obs_config["wheel_velocity_limit"],
                ]
            )
        else:
            obs_high = np.array(
                [np.pi / 2, 8.0, np.inf, 5.0, np.inf, 20.0]  # theta  # theta_dot  # x  # x_dot  # phi  # phi_dot
            )

        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Initialize state and render setup
        self.state = None
        self.steps = 0
        self.fig = None
        self.ax = None

        if self.config:
            reward_config = self.config["reward"]
            self.reward_weights = {
                "angle": reward_config["angle_weight"],
                "angular_velocity": reward_config["angular_velocity_weight"],
                "position": reward_config["position_weight"],
                "velocity": reward_config["velocity_weight"],
                "angle_decay": reward_config["angle_decay"],
                "position_decay": reward_config["position_decay"],
            }
        else:
            self.reward_weights = {
                "angle": 2.0,
                "angular_velocity": 3.0,
                "position": 5.0,
                "velocity": 0.5,
                "angle_decay": 10.0,
                "position_decay": 5.0,
            }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options (unused)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Initialize with small random angle
        self.state = np.array(
            [
                self.np_random.uniform(-0.3, 0.3),  # theta
                0.0,  # theta_dot
                self.np_random.uniform(-0.15, 0.15),  # x
                0.0,  # x_dot
                0.0,  # phi
                0.0,  # phi_dot
            ]
        )

        self.steps = 0

        if self.render_mode == "human":
            self._render_frame()

        return self.state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step.

        Args:
            action: Action to take (scaled motor torque)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Scale action to actual torque
        torque = np.clip(action, -1.0, 1.0) * self.max_torque

        # Calculate accelerations (using SimNet if available)
        accelerations = self.physics.get_acceleration(self.state, torque, self.simnet)

        # Update state
        self.state = (
            self.physics.integrate_state(self.state, accelerations)
            if self.simnet is None
            else self.simnet(self.state, torque)
        ).flatten()
        self.steps += 1

        # Calculate rewards
        reward = self._compute_reward()

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.steps >= self.max_steps

        # Additional info
        info = {
            "state_of_interest": {
                "angle": self.state[0],
                "position": self.state[2],
                "energy": self.physics.get_energy(self.state),
            }
        }

        if self.render_mode == "human":
            self._render_frame()

        return self.state, reward, terminated, truncated, info

    def _compute_reward(self) -> float:
        """Compute reward based on current state."""
        w = self.reward_weights

        theta = self.state[0]
        theta_dot = self.state[1]
        x = self.state[2]
        x_dot = self.state[3]

        # Angle reward (exponential decay with angle)
        angle_reward = np.exp(-w["angle_decay"] * theta**2)

        # Angular velocity penalty
        angular_vel_penalty = -0.5 * theta_dot**2

        # Position reward (keep robot near origin)
        position_reward = np.exp(-w["position_decay"] * x**2)

        # Velocity penalty (discourage high speeds)
        velocity_penalty = -0.1 * x_dot**2

        # Combine rewards with weights
        return (
            w["angle"] * angle_reward
            + w["angular_velocity"] * angular_vel_penalty
            + w["position"] * position_reward
            + w["velocity"] * velocity_penalty
        )

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        theta = self.state[0]
        x = self.state[2]

        # Terminate if angle too large or robot too far from origin
        if self.config:
            term_config = self.config["termination"]
            return abs(theta) > term_config["max_angle"] or abs(x) > term_config["max_position"]
        else:
            return abs(theta) > np.pi / 3 or abs(x) > 0.5

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            return self._render_frame()

        return None

    def _render_frame(self):
        """Render one frame of the environment."""
        import matplotlib.pyplot as plt

        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111)

        self.ax.clear()

        # Extract state
        theta = self.state[0]
        x = self.state[2]

        # Robot dimensions
        wheel_radius = self.physics.params.r
        body_length = self.physics.params.l * 2  # Double for visualization

        # Draw ground
        self.ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Draw wheel
        wheel = plt.Circle((x, wheel_radius), wheel_radius, fill=False, color="black")
        self.ax.add_patch(wheel)

        # Draw body
        body_x = x + body_length * np.sin(theta)
        body_y = wheel_radius + body_length * np.cos(theta)
        self.ax.plot([x, body_x], [wheel_radius, body_y], "b-", linewidth=3)

        # Configure view
        self.ax.set_xlim(x - 0.5, x + 0.5)
        self.ax.set_ylim(-0.1, 0.3)
        self.ax.set_aspect("equal")

        # Add title with state information
        self.ax.set_title(f"θ: {theta * 180/np.pi:.1f}°, " f"x: {x:.2f}m\n" f"Steps: {self.steps}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self.render_mode == "rgb_array":
            # Convert plot to RGB array
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data

    def close(self):
        """Clean up environment resources."""
        if self.fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self.fig)
            self.fig = None
