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
        config_dict: Optional[Dict] = None,
    ):
        """Initialize environment.

        Args:
            physics_params: Custom physics parameters
            config_path: Path to YAML configuration file
            render_mode: Rendering mode ('human' or 'rgb_array')
            simnet: Optional SimNet model for dynamics prediction
            config_dict: Configuration dictionary (alternative to config_path)
        """
        super().__init__()

        # Load config if provided
        if config_dict:
            self.config = config_dict
        elif config_path:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = None

        self.physics = PhysicsEngine(physics_params or PhysicsParams(config_path))
        self.simnet = simnet
        
        # Get rendering parameters
        if self.config:
            self.max_steps = self.config["termination"]["max_steps"]
            if render_mode is None and "render" in self.config:
                render_mode = self.config["render"]["mode"]
        else:
            self.max_steps = 500
            
        self.render_mode = render_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define action space (motor torque)
        if self.config:
            self.max_torque = self.config["physics"]["max_torque"]
        else:
            self.max_torque = 0.23
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Define observation space
        # [theta, theta_dot, x_axis, z_axis]
        # For accelerometer readings, use configured limits or defaults
        g = self.physics.params.g  # Gravitational acceleration
        
        if self.config:
            obs_config = self.config["observation"]
            
            # Get accelerometer range if specified
            if "accelerometer" in obs_config and "range" in obs_config["accelerometer"]:
                accel_limit = obs_config["accelerometer"]["range"]
            else:
                accel_limit = 2.0 * g
            
            obs_high = np.array(
                [
                    obs_config["angle_limit"],            # theta (angle)
                    obs_config["angular_velocity_limit"], # theta_dot (angular rate)
                    accel_limit,                          # x_axis (body frame acceleration)
                    accel_limit,                          # z_axis (body frame acceleration)
                ]
            )
        else:
            obs_high = np.array(
                [np.pi / 2, 8.0, 2.0 * g, 2.0 * g]  # theta  # theta_dot  # x_axis  # z_axis
            )

        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Initialize state and render setup
        self.state = None          # Internal full state
        self.observation = None    # Observation returned to agent
        self.last_accelerations = np.zeros(3)  # Store last accelerations for sensor calculations
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

    def _get_observation(self) -> np.ndarray:
        """Convert internal state to observation.
        
        Returns:
            Observation vector [theta, theta_dot, x_axis, z_axis]
            where x_axis and z_axis are accelerations in the body frame
        """
        # Extract state variables
        theta = self.state[0]
        theta_dot = self.state[1]
        
        # Use the actual acceleration from physics engine if available
        if hasattr(self, 'last_accelerations') and np.any(self.last_accelerations):
            theta_ddot = self.last_accelerations[0]  # Angular acceleration
            x_ddot = self.last_accelerations[1]      # Linear acceleration in world frame
        else:
            # Fallback if accelerations aren't stored
            # This is approximate and less accurate
            x_ddot = self.state[3]  # Using velocity as proxy (not ideal)
            theta_ddot = 0  # Cannot estimate accurately without physics
        
        # Get gravity constant
        g = self.physics.params.g
        
        # Calculate accelerations in body frame
        # For a balancing robot, we need to project both gravity and linear acceleration
        # onto the body frame axes
        
        # Rotation matrix from world to body frame
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Project gravity vector onto body frame
        # In world frame, gravity is (0, -g)
        gravity_x_body = g * sin_theta     # Gravity component along body x-axis
        gravity_z_body = -g * cos_theta    # Gravity component along body z-axis
        
        # Project linear acceleration onto body frame
        # x_ddot is horizontal acceleration in world frame
        accel_x_body = x_ddot * cos_theta  # World x acceleration component along body x-axis
        accel_z_body = x_ddot * sin_theta  # World x acceleration component along body z-axis
        
        # Add centripetal acceleration (caused by rotation of the body)
        # a_centripetal = ω² * r, where r is distance from center of rotation
        # This affects primarily the z-axis in body frame
        body_length = self.physics.params.l
        centripetal_accel = theta_dot**2 * body_length  # ω² * r
        centripetal_z = centripetal_accel * cos_theta  # Projection onto body z-axis
        
        # Combine all accelerations
        x_axis = accel_x_body + gravity_x_body                # Total acceleration along body x-axis
        z_axis = accel_z_body + gravity_z_body + centripetal_z  # Total acceleration along body z-axis
        
        # Add noise to simulate real sensor readings if configured
        if self.config and "observation" in self.config and "accelerometer" in self.config["observation"]:
            accel_config = self.config["observation"]["accelerometer"]
            if accel_config.get("add_noise", False):
                noise_std = accel_config.get("noise_std", 0.01 * g)
                x_axis += self.np_random.normal(0, noise_std)
                z_axis += self.np_random.normal(0, noise_std)
        
        return np.array([theta, theta_dot, x_axis, z_axis], dtype=np.float32)

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
        if self.config and "reset" in self.config:
            reset_config = self.config["reset"]
            angle_range = reset_config.get("angle_range", 0.3)
            position_range = reset_config.get("position_range", 0.15)
        else:
            angle_range = 0.3
            position_range = 0.15
            
        self.state = np.array(
            [
                self.np_random.uniform(-angle_range, angle_range),  # theta
                0.0,  # theta_dot
                self.np_random.uniform(-position_range, position_range),  # x
                0.0,  # x_dot
                0.0,  # phi
                0.0,  # phi_dot
            ]
        )

        self.steps = 0
        self.last_accelerations = np.zeros(3)  # Reset accelerations
        self.observation = self._get_observation()

        if self.render_mode == "human":
            self._render_frame()

        info = {
            "state_of_interest": {
                "angle": self.state[0],
                "position": self.state[2],
                "body_frame_accelerations": {
                    "x_axis": self.observation[2],
                    "z_axis": self.observation[3]
                }
            }
        }

        return self.observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step.

        Args:
            action: Action to take (scaled motor torque)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Scale action to actual torque
        torque = np.clip(action, -1.0, 1.0) * self.max_torque

        # Calculate accelerations with physics engine
        accelerations = self.physics.get_acceleration(self.state, torque)
        
        # Store accelerations for sensor readings
        self.last_accelerations = accelerations.copy()

        # Update state, with SimNet if available
        if self.simnet is None:
            self.state = self.physics.integrate_state(self.state, accelerations).flatten()
        else:
            s_tensor = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            a_tensor = torch.tensor(torque, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.state = self.simnet(s_tensor, a_tensor).cpu().detach().numpy()[0]
            
        self.steps += 1
        
        # Get observation from state
        self.observation = self._get_observation()

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
                "full_state": self.state.copy(),
                "body_frame_accelerations": {
                    "x_axis": self.observation[2],
                    "z_axis": self.observation[3]
                }
            }
        }

        if self.render_mode == "human":
            self._render_frame()

        return self.observation, reward, terminated, truncated, info

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
        from matplotlib.patches import Rectangle, Arrow

        if self.fig is None:
            plt.ion()
            if self.config and "render" in self.config:
                width = self.config["render"].get("width", 800) / 100
                height = self.config["render"].get("height", 600) / 100
                self.fig = plt.figure(figsize=(width, height))
            else:
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
        
        # Check if we should show accelerometer vectors
        show_accelerometer = True
        arrow_scale = 0.05
        
        if self.config and "render" in self.config:
            show_accelerometer = self.config["render"].get("show_accelerometer", True)
            arrow_scale = self.config["render"].get("accelerometer_scale", 0.05)
        
        if show_accelerometer:
            # Draw accelerometer arrows (body frame)
            x_accel = self.observation[2]
            z_accel = self.observation[3]
            
            # Convert to g units for display scaling (typical accelerometer reads in g)
            g = self.physics.params.g
            x_accel_g = x_accel / g
            z_accel_g = z_accel / g
            
            # Origin of the arrow at center of the body
            arrow_origin_x = (x + body_x) / 2
            arrow_origin_y = (wheel_radius + body_y) / 2
            
            # Direction vectors in world frame
            x_dir_x = arrow_scale * x_accel * np.cos(theta)
            x_dir_y = arrow_scale * x_accel * np.sin(theta)
            
            z_dir_x = -arrow_scale * z_accel * np.sin(theta)
            z_dir_y = arrow_scale * z_accel * np.cos(theta)
            
            # Draw acceleration vectors
            self.ax.arrow(arrow_origin_x, arrow_origin_y, x_dir_x, x_dir_y, 
                         head_width=0.01, head_length=0.015, fc='r', ec='r', label=f'X-Axis ({x_accel_g:.2f}g)')
            self.ax.arrow(arrow_origin_x, arrow_origin_y, z_dir_x, z_dir_y, 
                         head_width=0.01, head_length=0.015, fc='g', ec='g', label=f'Z-Axis ({z_accel_g:.2f}g)')
            
            # Add legend
            self.ax.legend(loc='upper right')

        # Configure view
        self.ax.set_xlim(x - 0.5, x + 0.5)
        self.ax.set_ylim(-0.1, 0.3)
        self.ax.set_aspect("equal")

        # Add title with state information
        g = self.physics.params.g
        title_text = f"θ: {theta * 180/np.pi:.1f}°, x: {x:.2f}m\n"
        title_text += f"x_axis: {self.observation[2]/g:.2f}g, z_axis: {self.observation[3]/g:.2f}g\n"
        title_text += f"Steps: {self.steps}"
        
        self.ax.set_title(title_text)

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
