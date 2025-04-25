import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define action space (always normalized to [-1, 1])
        # The actor network always outputs values in this range
        if self.config:
            # max_torque is only used in physics simulation to scale normalized actions to actual torque values
            self.max_torque = self.config["physics"]["max_torque"]
            # max_delta is the maximum change allowed in the normalized action space [-1, 1]
            self.max_delta = self.config["physics"].get("max_delta", 0.1)  # Default to 10% of normalized range
        else:
            self.max_torque = 0.23  # Used only for physics simulation
            self.max_delta = 0.1  # 10% of normalized range

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Define observation space - [theta, theta_dot, prev_motor_command]
        if self.config:
            obs_config = self.config["observation"]
            obs_high = np.array(
                [
                    obs_config["angle_limit"],
                    obs_config["angular_velocity_limit"],
                    1.0,  # Previous motor command is normalized to [-1, 1]
                ]
            )
        else:
            obs_high = np.array(
                [
                    np.pi / 2,  # theta
                    8.0,  # theta_dot
                    1.0,  # prev_motor_command
                ]
            )

        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Initialize state and render setup
        self.state = None
        self.steps = 0
        self.prev_action = 0.0
        self.fig = None
        self.ax = None

        if self.config:
            reward_config = self.config["reward"]
            self.reward_weights = {
                "angle": reward_config["angle"],
                "direction": reward_config["direction"],
                "angular_velocity": reward_config["angular_velocity"],
                "angle_decay": reward_config["angle_decay"],
                "reached_stable_bonus": reward_config["reached_stable_bonus"],
                "stillness": reward_config["stillness"],
                "far_from_center_penalty": reward_config["far_from_center_penalty"],
            }
        else:
            self.reward_weights = {
                "angle": 5.0,
                "direction": 3.0,
                "angular_velocity": 1.0,
                "angle_decay": 30.0,
                "reached_stable_bonus": 50.0,
                "stillness": 5.0,
                "far_from_center_penalty": 0.5,
            }

    def reset(self, seed: Optional[int] = None, should_zero_previous_action: bool = False, state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.

        Args:
            seed: Random seed
            should_zero_previous_action: If True, set previous action to zero
            state: Optional initial state for the environment

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if state is not None:
            # If a state is provided, use it directly
            self.state = state
        else:

            # Initialize with random angle and angular velocity
            theta_deg = self.np_random.uniform(-50, 50)  # theta in degrees
            theta_dot_dps = self.np_random.uniform(-150, 150)  # theta_dot in degrees per second
            random_motor_command = self.np_random.uniform(-1.0, 1.0)  # Random initial motor command
            prev_motor_command = 0.0 if should_zero_previous_action else random_motor_command

            self.state = np.array(
                [
                    np.deg2rad(theta_deg),  # Convert theta to radians
                    np.deg2rad(theta_dot_dps),  # Convert theta_dot to radians per second
                    prev_motor_command,  # Use random initial motor command
                ]
            )

        self.prev_action = self.state[2]  # Store the initial motor command
        self.steps = 0

        if self.render_mode == "human":
            self._render_frame()

        return self.state, {}

    def step(self, action: np.ndarray, action_as_actual_output: bool = False) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step with delta-based action.

        Args:
            action: Action to take (delta in motor command, scaled to [-1, 1])
            action_as_actual_output: If True, action is treated as the actual motor command (not delta)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get previous motor command from state
        prev_motor_command = self.state[2]

        # Calculate delta (scaled by max_delta)
        delta = np.clip(action, -1.0, 1.0) * self.max_delta

        # Apply delta to previous command and clip to valid range
        new_motor_command = np.clip(prev_motor_command + delta, -1.0, 1.0)

        # If action is treated as actual output, use it directly
        if action_as_actual_output:
            new_motor_command = np.clip(action, -1.0, 1.0)


        # Scale the motor command to actual torque
        # This is the only place where max_torque is used - to convert normalized [-1, 1] to physical torque
        torque = new_motor_command * self.max_torque

        # Extract theta and theta_dot for physics update
        theta, theta_dot, _ = self.state

        # Calculate angular acceleration with physics engine
        theta_ddot = self.physics.get_acceleration(self.state[:2], torque)

        # Update state, with SimNet if available
        if self.simnet is None:
            # Create physics state with just the angle components
            physics_state = np.array([theta, theta_dot])
            new_state = self.physics.integrate_state(physics_state, theta_ddot)
            # Combine with previous action
            self.state = np.array(
                [
                    new_state[0],  # Updated theta
                    new_state[1],  # Updated theta_dot
                    new_motor_command[0],  # Store the new motor command
                ]
            )
        else:
            s_tensor = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            # For SimNet, we need to provide the actual command, not the delta
            a_tensor = torch.tensor([[new_motor_command[0]]], dtype=torch.float32, device=self.device)
            self.state = self.simnet(s_tensor, a_tensor).cpu().detach().numpy()[0]
            self.state[2] = new_motor_command[0]  # Ensure the motor command is correctly stored

        # Add noise to state
        if self.config:
            noise_std = self.config["observation"]["noise_std"]
            # Add noise only to theta and theta_dot, not to the action
            noise = self.np_random.normal(0, noise_std, size=2)
            self.state[:2] += noise

        self.steps += 1
        self.prev_action = new_motor_command[0]  # Store current motor command for next step

        # Check if reached stable state
        min_angle_for_stable = np.deg2rad(6)
        min_angular_velocity_for_stable = np.deg2rad(6)
        reached_stable = (
            abs(self.state[0]) < min_angle_for_stable and abs(self.state[1]) < min_angular_velocity_for_stable
        )

        # Calculate rewards
        reward = self._compute_reward(reached_stable)

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.steps >= self.max_steps

        # Additional info
        info = {
            "state_of_interest": {
                "angle": self.state[0],
                "energy": self.physics.get_energy(self.state[:2]),
                "prev_motor_command": self.state[2],
                "delta": delta[0],  # Include delta in info for analysis
            },
            "reached_stable": reached_stable,
        }

        if self.render_mode == "human":
            self._render_frame()

        return self.state, reward, terminated, truncated, info

    # def _compute_reward(self, reached_stable: bool) -> float:
    #     """Compute reward based on current state."""
    #     w = self.reward_weights

    #     theta = self.state[0]
    #     theta_dot = self.state[1]

    #     # Angle reward (exponential decay with angle)
    #     angle_reward = np.exp(-w["angle_decay"] * theta**2)

    #     # Angular velocity penalty
    #     angular_vel_penalty = -0.5 * theta_dot**2

    #     reward = w["angle"] * angle_reward + w["angular_velocity"] * angular_vel_penalty

    #     if self._check_termination():
    #         reward -= 200

    #     if reached_stable:
    #         reward += w["reached_stable_bonus"]

    #     return float(reward)

    # def _compute_reward(self, reached_stable: bool) -> float:
    #     w = self.reward_weights
    #     theta = self.state[0]
    #     theta_dot = self.state[1]

    #     # Time penalty for unstable steps
    #     time_penalty = -1 if not reached_stable else 0

    #     # More gradual angle reward
    #     angle_reward = 1.0 / (1.0 + w["angle_decay"] * theta**2)

    #     # Directional component: reward corrective actions
    #     # Negative reward when angle and angular velocity have same sign
    #     # (robot is moving away from center)
    #     direction_reward = -np.sign(theta) * theta_dot

    #     # Angular velocity penalty with a small deadzone
    #     vel_deadzone = 0.05  # Radians/sec
    #     angular_vel_penalty = -0.5 * max(0, abs(theta_dot) - vel_deadzone)**2

    #     reward = (
    #         w["angle"] * angle_reward +
    #         w["direction"] * max(0, direction_reward) + # Only reward corrective motion
    #         w["angular_velocity"] * angular_vel_penalty +
    #         time_penalty  # Time penalty for unstable steps
    #     )

    #     # Smoother termination penalty
    #     if self._check_termination():
    #         reward -= 20  # Less harsh

    #     # Graduated stability bonus
    #     if abs(theta) < np.radians(5) and abs(theta_dot) < np.radians(5):
    #         stability_factor = 1.0 - (abs(theta) / np.radians(5) + abs(theta_dot) / np.radians(5)) / 2
    #         reward += w["reached_stable_bonus"] * stability_factor

    #     return float(reward)

    def _compute_reward(self, reached_stable: bool) -> float:
        w = self.reward_weights
        theta = self.state[0]
        theta_dot = self.state[1]

        # Directional component:
        # Reward corrective actions when angle and angular velocity have opposite signs
        # Negative reward when angle and angular velocity have same sign
        # (robot is moving away from center)
        # Scale theta_dot contribution exponentially based on angle magnitude
        direction_component = -np.sign(theta) * theta_dot

        # New stillness reward that activates near the balanced position
        stillness_reward = 0
        angle_threshold = np.deg2rad(3)  # Angle threshold for stillness reward
        if abs(theta) < angle_threshold:
            # Reward is highest when both angle and angular velocity are zero
            # and decreases as either increases
            velocity_factor = max(0, 1.0 - (abs(theta_dot) / np.deg2rad(40)))  # 0.5 rad/s threshold

            angle_factor = 1.0 - (abs(theta) / angle_threshold)
            stillness_reward = w["stillness"] * angle_factor * velocity_factor**2

        termination_penalty = -20 if self._check_termination() else 0

        far_from_center_penalty = -abs(theta) * w["far_from_center_penalty"]

        reward = w["direction"] * direction_component + stillness_reward + termination_penalty + far_from_center_penalty

        return float(reward)

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        theta = self.state[0]

        # Terminate if angle too large
        if self.config:
            term_config = self.config["termination"]
            return abs(theta) > term_config["max_angle"]
        else:
            return abs(theta) > np.pi / 3

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
        prev_action = self.state[2]

        # Robot dimensions
        wheel_radius = self.physics.params.r
        body_length = self.physics.params.l * 2  # Double for visualization

        # Draw ground
        self.ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Fixed position for visualization (assuming no movement)
        x = 0

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

        # Add title with state information including previous motor command
        self.ax.set_title(f"θ: {theta * 180/np.pi:.1f}°\n" f"Motor: {prev_action:.2f}\n" f"Steps: {self.steps}")

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
