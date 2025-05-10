import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import collections
from typing import Tuple, Dict, Any, Optional

from .physics import PhysicsEngine, PhysicsParams
from ..models.simnet import SimNet


class BalancerEnv(gym.Env):
    """Gymnasium environment for a two-wheeled balancing robot with motor delay compensation."""

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
        if self.config:
            # max_torque is only used in physics simulation to scale normalized actions to actual torque values
            self.max_torque = self.config["physics"]["max_torque"]
            # max_delta is the maximum change allowed in the normalized action space [-1, 1]
            self.max_delta = self.config["physics"].get("max_delta", 0.1)  # Default to 10% of normalized range
        else:
            self.max_torque = 0.23  # Used only for physics simulation
            self.max_delta = 0.1  # 10% of normalized range

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Define observation space with expanded state
        # [theta, theta_dot, prev_motor_command,
        #  theta_ma, theta_dot_ma,
        #  pwm_t-1, pwm_t-2, pwm_t-3, pwm_t-4]
        if self.config:
            obs_config = self.config["observation"]
            angle_limit = obs_config["angle_limit"]
            velocity_limit = obs_config["angular_velocity_limit"]
        else:
            angle_limit = np.pi / 2
            velocity_limit = 8.0

        # Create observation space bounds
        # Basic observation + moving averages + 4 PWM snapshots
        obs_high = np.array(
            [
                angle_limit,  # theta
                velocity_limit,  # theta_dot
                1.0,  # prev_motor_command
                angle_limit,  # theta moving average
                velocity_limit,  # theta_dot moving average
                1.0,
                1.0,
                1.0,
                1.0,  # 4 PWM snapshots
            ]
        )

        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Initialize state tracking variables
        self.state = None
        self.steps = 0
        self.prev_action = 0.0

        # Motor delay model - buffer of commands with configurable delay
        self.motor_delay_steps = (
            self.config["physics"].get("motor_delay_steps", 2) if self.config else 2
        )  # Default to ~80ms at 25Hz
        self.motor_command_buffer = collections.deque([0.0] * self.motor_delay_steps, maxlen=self.motor_delay_steps)

        # Tracking for moving averages and action history
        self.theta_history_size = self.config["observation"].get("theta_history_size", 5) if self.config else 5
        self.action_history_size = self.config["observation"].get("action_history_size", 4) if self.config else 4

        self.theta_history = collections.deque([0.0] * self.theta_history_size, maxlen=self.theta_history_size)
        self.theta_dot_history = collections.deque([0.0] * self.theta_history_size, maxlen=self.theta_history_size)
        self.action_history = collections.deque([0.0] * self.action_history_size, maxlen=self.action_history_size)

        # Initialize rendering
        self.fig = None
        self.ax = None

        # Load reward weights from config
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
                "angular_vel_far_from_center_penalty": reward_config["angular_vel_far_from_center_penalty"],
                "max_angle_for_angular_vel_far_from_center_penalty": reward_config[
                    "max_angle_for_angular_vel_far_from_center_penalty"
                ],
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
                "angular_vel_far_from_center_penalty": 0.5,
                "max_angle_for_angular_vel_far_from_center_penalty": 10.0,
            }

    def _get_enhanced_state(self) -> np.ndarray:
        """Create the enhanced state representation with moving averages and PWM history."""
        # Calculate moving averages
        theta_ma = np.mean(self.theta_history)
        theta_dot_ma = np.mean(self.theta_dot_history)

        # Basic state elements
        basic_state = self.state if hasattr(self, "state") and self.state is not None else np.zeros(3)

        # Combine everything into enhanced state
        enhanced_state = np.concatenate(
            [
                basic_state,  # [theta, theta_dot, prev_motor_command]
                [theta_ma, theta_dot_ma],  # Moving averages
                np.array(self.action_history),  # Action history snapshots
            ]
        )

        return enhanced_state

    def reset(
        self, seed: Optional[int] = None, should_zero_previous_action: bool = False, state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state with enhanced state representation.

        Args:
            seed: Random seed
            should_zero_previous_action: If True, set previous action to zero
            state: Optional initial state for the environment

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if state is not None and len(state) >= 3:
            # If a state is provided, use its basic elements
            self.state = state[:3]  # Take first 3 elements (theta, theta_dot, prev_motor_command)
        else:
            # Initialize with random angle and angular velocity
            theta_deg = self.np_random.uniform(-50, 50)  # theta in degrees
            theta_dot_dps = self.np_random.uniform(-250, 250)  # theta_dot in degrees per second
            random_motor_command = self.np_random.uniform(-1.0, 1.0)  # Random initial motor command
            prev_motor_command = 0.0 if should_zero_previous_action else random_motor_command

            self.state = np.array(
                [
                    np.deg2rad(theta_deg),  # Convert theta to radians
                    np.deg2rad(theta_dot_dps),  # Convert theta_dot to radians per second
                    prev_motor_command,  # Use random initial motor command
                ]
            )

        # Initialize histories and buffers
        self.theta_history = collections.deque(
            [self.state[0]] * self.theta_history_size, maxlen=self.theta_history_size
        )
        self.theta_dot_history = collections.deque(
            [self.state[1]] * self.theta_history_size, maxlen=self.theta_history_size
        )
        self.action_history = collections.deque(
            [self.state[2]] * self.action_history_size, maxlen=self.action_history_size
        )
        self.motor_command_buffer = collections.deque(
            [self.state[2]] * self.motor_delay_steps, maxlen=self.motor_delay_steps
        )

        self.prev_action = self.state[2]  # Store the initial motor command
        self.steps = 0

        if self.render_mode == "human":
            self._render_frame()

        # Return the enhanced state
        return self._get_enhanced_state(), {}

    def step(
        self, action: np.ndarray, action_as_actual_output: bool = False
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step with motor delay simulation.

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

        # Add new command to motor delay buffer
        self.motor_command_buffer.append(new_motor_command[0])

        # Get delayed command (the one that takes effect now due to motor delay)
        delayed_command = self.motor_command_buffer[0]

        # Update action history
        self.action_history.appendleft(new_motor_command[0])

        # Scale the delayed motor command to actual torque
        torque = delayed_command * self.max_torque

        # Extract theta and theta_dot for physics update
        theta, theta_dot, _ = self.state

        # Add to history for moving averages
        self.theta_history.appendleft(theta)
        self.theta_dot_history.appendleft(theta_dot)

        # Calculate angular acceleration with physics engine using the DELAYED command
        theta_ddot = self.physics.get_acceleration(self.state[:2], torque)

        # Update state, with SimNet if available
        if self.simnet is None:
            # Create physics state with just the angle components
            physics_state = np.array([theta, theta_dot])
            new_state = self.physics.integrate_state(physics_state, theta_ddot)
            # Combine with new motor command
            self.state = np.array(
                [
                    new_state[0],  # Updated theta
                    new_state[1],  # Updated theta_dot
                    new_motor_command[0],  # Store the new motor command
                ]
            )
        else:
            # For SimNet, we need to provide the augmented state
            augmented_state = self._get_enhanced_state()
            s_tensor = torch.tensor(augmented_state, dtype=torch.float32, device=self.device).unsqueeze(0)

            # For action, provide the actual command, not the delta
            a_tensor = torch.tensor([[new_motor_command[0]]], dtype=torch.float32, device=self.device)

            # Get prediction and extract the basic state elements
            next_augmented_state = self.simnet(s_tensor, a_tensor).cpu().detach().numpy()[0]
            self.state = np.array(
                [
                    next_augmented_state[0],  # theta
                    next_augmented_state[1],  # theta_dot
                    new_motor_command[0],  # new motor command
                ]
            )

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
                "delayed_command": delayed_command,  # Include the delayed command that took effect
            },
            "reached_stable": reached_stable,
        }

        if self.render_mode == "human":
            self._render_frame()

        # Return the enhanced state
        return self._get_enhanced_state(), reward, terminated, truncated, info

    def _compute_reward(self, reached_stable: bool) -> float:
        """Compute reward based on current state."""
        # Reward calculation remains the same
        w = self.reward_weights
        theta = self.state[0] + np.deg2rad(7.5)  # Offset to center the reward around zero
        theta_dot = self.state[1]

        # Directional component
        direction_component = -np.sign(theta) * theta_dot

        # Stillness reward
        stillness_reward = 0
        angle_threshold = np.deg2rad(10)
        if abs(theta) < angle_threshold:
            angle_factor = 1.0 - (abs(theta) / angle_threshold)
            stillness_reward = w["stillness"] * angle_factor

            if abs(theta_dot) < np.deg2rad(30):
                velocity_factor = max(0, 1.0 - (abs(theta_dot) / np.deg2rad(40)))
                stillness_reward += velocity_factor**2 * w["stillness"]

        termination_penalty = -20 if self._check_termination() else 0

        far_from_center_penalty = -abs(theta) * w["far_from_center_penalty"]

        angular_vel_far_from_center_penalty = (
            -abs(theta_dot) * w["angular_vel_far_from_center_penalty"]
            if abs(theta) < np.deg2rad(w["max_angle_for_angular_vel_far_from_center_penalty"])
            else 0
        )

        reward = (
            w["direction"] * direction_component
            + stillness_reward
            + termination_penalty
            + far_from_center_penalty
            + angular_vel_far_from_center_penalty
        )

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
        # Add motor delay information
        self.ax.set_title(
            f"θ: {theta * 180/np.pi:.1f}°\n"
            f"Motor cmd: {prev_action:.2f}\n"
            f"Delayed cmd: {self.motor_command_buffer[0]:.2f}\n"
            f"Steps: {self.steps}"
        )

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
