import yaml
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PhysicsParams:
    """Physical parameters of the two-wheeled balancing robot."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize parameters from config file or defaults."""
        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)["physics"]

            # Load from config
            self.g = config["gravity"]
            self.M = config["body_mass"]
            self.m = config["wheel_mass"]
            self.l = config["body_length"]
            self.r = config["wheel_radius"]
            self.I = config["body_inertia"]
            self.i = config["wheel_inertia"]
            self.dt = config["timestep"]
            self.motor_deadzone = config["motor_deadzone"]
            self.static_friction_coeff = config["static_friction"]
        else:
            # Default values as before
            self.g = 9.81
            self.M = 0.06
            self.m = 0.04
            self.l = 0.025
            self.r = 0.033
            self.I = 0.001
            self.i = 2e-5
            self.dt = 0.01
            self.motor_deadzone = 0.04
            self.static_friction_coeff = 0.7


class PhysicsEngine:
    """Physics engine for the balancing robot."""

    def __init__(self, params: Optional[PhysicsParams] = None):
        """Initialize physics engine with parameters.

        Args:
            params: Physics parameters, uses defaults if None
        """
        self.params = params or PhysicsParams()

    def calculate_normal_force(self) -> float:
        """Calculate normal force on wheels."""
        return (self.params.M + 2 * self.params.m) * self.params.g

    def calculate_static_friction_threshold(self) -> float:
        """Calculate maximum static friction force."""
        return self.params.static_friction_coeff * self.calculate_normal_force()

    def get_acceleration(self, state: np.ndarray, torque: np.ndarray) -> float:
        """Calculate system accelerations based on current state and applied torque.
        Args:
            state: System state [theta, theta_dot]
            torque: Applied motor torque
        Returns:
            Angular acceleration (theta_ddot)
        """
        # Extract state components
        theta = state[0]
        theta_dot = state[1]
        p = self.params

        # Calculate static friction effects
        max_static_friction = self.calculate_static_friction_threshold()

        # Define critical angular velocity where torque effectiveness is zero
        critical_velocity = 4.0  # rad/s

        # Calculate torque effectiveness based on angular velocity
        # Effectiveness drops to zero at critical_velocity, then rises again beyond it
        velocity_ratio = abs(theta_dot) / critical_velocity

        # This formula creates a parabola that equals 1 at velocity_ratio=0,
        # drops to 0 at velocity_ratio=1 (critical_velocity),
        # and increases again beyond that point
        torque_effectiveness = 1.0 - (2.0 * velocity_ratio) + (velocity_ratio**2)

        # Apply effectiveness to torque
        effective_torque = torque.item() * torque_effectiveness

        # If no effective torque, check static friction
        if abs(effective_torque) < p.motor_deadzone:
            required_friction = abs(p.M * p.g * p.l * np.sin(theta))
            if required_friction <= max_static_friction:
                # Static friction keeps system rigid
                I_total = p.M * p.l**2 + p.I + 2 * (p.m * p.r**2 + p.i)
                theta_ddot = (p.M * p.g * p.l * np.sin(theta)) / I_total
            else:
                # Static friction exceeded
                theta_ddot = (p.M * p.g * p.l * np.sin(theta)) / (p.M * p.l**2 + p.I)
        else:
            # Normal dynamics with applied torque, assuming fixed base (no x movement)
            theta_ddot = (p.M * p.g * p.l * np.sin(theta) - effective_torque) / (p.M * p.l**2 + p.I)

        return theta_ddot

    def integrate_state(self, state: np.ndarray, theta_ddot: float) -> np.ndarray:
        """Integrate state using semi-implicit Euler method."""
        # Extract components
        theta, theta_dot = state[0], state[1]
        dt = self.params.dt

        # Semi-implicit Euler integration
        theta_dot_new = theta_dot + theta_ddot * dt

        # Add constraint on angular velocity (missing in current code)
        theta_dot_new = np.clip(theta_dot_new, -8.0, 8.0)  # Limit to reasonable values

        # Update position
        theta_new = theta + theta_dot_new * dt

        return np.array([theta_new, theta_dot_new])

    def get_energy(self, state: np.ndarray) -> float:
        """Calculate total energy of the system.

        Args:
            state: System state [theta, theta_dot]

        Returns:
            Total energy in Joules
        """
        p = self.params
        theta, theta_dot = state[0], state[1]

        # Potential energy
        PE = p.M * p.g * p.l * (1 - np.cos(theta))

        # Kinetic energy - just body rotation now
        KE_body_rot = 0.5 * p.I * theta_dot**2

        return PE + KE_body_rot
