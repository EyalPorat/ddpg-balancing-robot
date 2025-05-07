import collections
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional

from .physics import PhysicsEngine, PhysicsParams
from ..models.simnet import SimNet


class BalancerEnv(gym.Env):
    """Gymnasium environment for a two-wheeled balancing robot with motor delay and time series state."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        physics_params: Optional[PhysicsParams] = None,
        config_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        simnet: Optional[SimNet] = None,
    ):
        super().__init__()

        # Load config
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

        # Time series and motor delay configuration
        if self.config and "physics" in self.config:
            self.time_steps = self.config["physics"].get("time_series_length", 10)
            self.motor_delay_steps = self.config["physics"].get("motor_response_delay", 3)
        else:
            self.time_steps = 10
            self.motor_delay_steps = 3

        self.single_state_dim = 3  # [theta, theta_dot, motor_cmd]

        # Motor command buffer for delay modeling
        self.motor_command_buffer = collections.deque([0.0] * self.time_steps, maxlen=self.time_steps)

        # State history to maintain time series
        self.state_history = collections.deque(maxlen=self.time_steps)

        # Define action space (normalized to [-1, 1])
        if self.config:
            self.max_torque = self.config["physics"]["max_torque"]
            self.max_delta = self.config["physics"].get("max_delta", 0.1)
        else:
            self.max_torque = 0.23
            self.max_delta = 0.1

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Define observation space - time series format
        if self.config:
            obs_config = self.config["observation"]
            single_obs_high = np.array(
                [
                    obs_config["angle_limit"],
                    obs_config["angular_velocity_limit"],
                    1.0,  # Motor command normalized to [-1, 1]
                ]
            )
        else:
            single_obs_high = np.array(
                [
                    np.pi / 2,  # theta
                    8.0,  # theta_dot
                    1.0,  # prev_motor_command
                ]
            )

        # Time series observation space
        obs_high = np.tile(single_obs_high, self.time_steps)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Initialize state and render setup
        self.steps = 0
        self.prev_action = 0.0
        self.fig = None
        self.ax = None

        # Initialize reward weights
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
            # Default reward weights
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

    def get_flat_time_series(self) -> np.ndarray:
        """Get flattened time series state from history."""
        # If not enough history, pad with copies of first state
        if len(self.state_history) < self.time_steps:
            # Get first state or use zeros if no history
            first_state = self.state_history[0] if self.state_history else np.zeros(self.single_state_dim)
            # Create padding with repeated first state
            padding = [first_state] * (self.time_steps - len(self.state_history))
            states = padding + list(self.state_history)
        else:
            states = list(self.state_history)

        # Flatten the list of states
        return np.concatenate(states)

    def reset(
        self, seed: Optional[int] = None, should_zero_previous_action: bool = False, state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.

        Returns time series state.
        """
        super().reset(seed=seed)

        # Clear state history and motor command buffer
        self.state_history = collections.deque(maxlen=self.time_steps)
        self.motor_command_buffer = collections.deque([0.0] * self.time_steps, maxlen=self.time_steps)

        if state is not None:
            # If time series state provided, extract and use the most recent state
            if len(state) == self.single_state_dim * self.time_steps:
                # Extract individual state samples from time series
                samples = [state[i : i + self.single_state_dim] for i in range(0, len(state), self.single_state_dim)]

                # Initialize state history with provided samples
                for sample in samples:
                    self.state_history.append(sample.copy())

                # Current state is the most recent one
                initial_state = samples[-1].copy()
            else:
                # Single state provided
                initial_state = state.copy()
                self.state_history.append(initial_state.copy())
        else:
            # Initialize with random angle and angular velocity
            theta_deg = self.np_random.uniform(-50, 50)
            theta_dot_dps = self.np_random.uniform(-150, 150)
            motor_cmd = 0.0 if should_zero_previous_action else self.np_random.uniform(-1.0, 1.0)

            initial_state = np.array([np.deg2rad(theta_deg), np.deg2rad(theta_dot_dps), motor_cmd])

            self.state_history.append(initial_state.copy())

        # If state history is not full, fill it with copies of initial state
        while len(self.state_history) < self.time_steps:
            self.state_history.appendleft(initial_state.copy())

        self.steps = 0
        self.prev_action = self.state_history[-1][2]  # Use motor command from most recent state

        if self.render_mode == "human":
            self._render_frame()

        # Return flattened time series state
        time_series_state = np.concatenate(list(self.state_history))
        return time_series_state, {}

    def step(
        self, action: np.ndarray, action_as_actual_output: bool = False
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step with motor delay.

        Returns time series state.
        """
        # Get current state (most recent in history)
        current_state = self.state_history[-1].copy()

        # Get previous motor command
        prev_motor_command = current_state[2]

        # Calculate delta (scaled by max_delta)
        delta = np.clip(action, -1.0, 1.0) * self.max_delta

        # Apply delta to previous command and clip to valid range
        new_motor_command = np.clip(prev_motor_command + delta, -1.0, 1.0)

        # If action is treated as actual output, use it directly
        if action_as_actual_output:
            new_motor_command = np.clip(action, -1.0, 1.0)

        # Add to motor command buffer for delay modeling
        self.motor_command_buffer.append(new_motor_command.item())

        # Get delayed motor command
        if len(self.motor_command_buffer) > self.motor_delay_steps:
            delayed_command = self.motor_command_buffer[-self.motor_delay_steps]
        else:
            delayed_command = 0.0

        # Scale the delayed motor command to actual torque
        torque = delayed_command * self.max_torque

        # Extract theta and theta_dot for physics update
        theta, theta_dot = current_state[:2]

        # Update state using physics or SimNet
        if self.simnet is None:
            # Calculate angular acceleration with physics engine
            theta_ddot = self.physics.get_acceleration(np.array([theta, theta_dot]), torque)

            # Integrate state using physics model
            physics_state = np.array([theta, theta_dot])
            new_physics_state = self.physics.integrate_state(physics_state, theta_ddot)

            # Create new state with updated physics and new command
            next_state = np.array(
                [
                    new_physics_state[0],  # Updated theta
                    new_physics_state[1],  # Updated theta_dot
                    new_motor_command.item(),  # Store the new motor command
                ]
            )
        else:
            # Use SimNet for next state prediction using time series
            # Get current time series state
            time_series_state = np.concatenate(list(self.state_history))

            # Convert to tensors for SimNet
            s_tensor = torch.tensor(time_series_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            a_tensor = torch.tensor([[new_motor_command.item()]], dtype=torch.float32, device=self.device)

            # Get next time series from SimNet
            next_time_series = self.simnet(s_tensor, a_tensor).cpu().detach().numpy()[0]

            # Extract the newest state (last 3 elements)
            next_state = next_time_series[-self.single_state_dim :].copy()
            next_state[2] = new_motor_command.item()  # Ensure motor command is correctly stored

        # Add noise to state if configured
        if self.config:
            noise_std = self.config["observation"]["noise_std"]
            # Add noise only to theta and theta_dot
            noise = self.np_random.normal(0, noise_std, size=2)
            next_state[:2] += noise

        # Add new state to history
        self.state_history.append(next_state.copy())

        self.steps += 1
        self.prev_action = new_motor_command.item()

        # Check if reached stable state
        min_angle_for_stable = np.deg2rad(6)
        min_angular_velocity_for_stable = np.deg2rad(6)
        reached_stable = (
            abs(next_state[0]) < min_angle_for_stable and abs(next_state[1]) < min_angular_velocity_for_stable
        )

        # Calculate rewards based on the most recent state
        reward = self._compute_reward(reached_stable)

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.steps >= self.max_steps

        # Additional info
        info = {
            "state_of_interest": {
                "angle": next_state[0],
                "energy": self.physics.get_energy(next_state[:2]),
                "prev_motor_command": next_state[2],
                "delta": delta.item(),
                "motor_delay": self.motor_delay_steps,
                "delayed_command": delayed_command,
            },
            "reached_stable": reached_stable,
        }

        if self.render_mode == "human":
            self._render_frame()

        # Return time series state
        time_series_state = np.concatenate(list(self.state_history))
        return time_series_state, reward, terminated, truncated, info

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
        """Compute reward based on most recent state."""
        w = self.reward_weights

        # Get most recent state
        current_state = self.state_history[-1] + np.deg2rad(6.8)  # Offset to center the reward around zero
        theta = current_state[0]
        theta_dot = current_state[1]

        # The reward calculation can remain the same, just using the most recent state
        direction_component = -np.sign(theta) * theta_dot

        # Stillness reward near balanced position
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
        """Check if episode should terminate based on most recent angle."""
        # Get most recent state
        current_state = self.state_history[-1]
        theta = current_state[0]

        # Terminate if angle too large
        if self.config:
            term_config = self.config["termination"]
            return abs(theta) > term_config["max_angle"]
        else:
            return abs(theta) > np.pi / 3

    def _render_frame(self):
        """Render one frame of the environment using the most recent state."""
        import matplotlib.pyplot as plt

        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111)

        self.ax.clear()

        # Extract most recent state for visualization
        current_state = self.state_history[-1]
        theta = current_state[0]
        prev_action = current_state[2]

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
        self.ax.set_title(
            f"θ: {theta * 180/np.pi:.1f}°\n"
            f"Motor: {prev_action:.2f}\n"
            f"Steps: {self.steps}\n"
            f"Motor Delay: {self.motor_delay_steps} steps"
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
