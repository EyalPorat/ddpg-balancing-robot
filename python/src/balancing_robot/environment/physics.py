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

    def get_acceleration(self, state: np.ndarray, torque: np.ndarray, simnet=None) -> np.ndarray:
        """Calculate system accelerations based on current state and applied torque.

        Args:
            state: System state [theta, theta_dot, x, x_dot, phi, phi_dot]
            torque: Applied motor torque
            simnet: Optional SimNet model for dynamics prediction

        Returns:
            Array of [theta_ddot, x_ddot, phi_ddot]
        """
        # If SimNet is provided and active, use it for predictions
        if simnet is not None:
            return simnet.get_accelerations(state, torque)

        # Otherwise use physics-based calculations
        theta = state[0]
        p = self.params

        # Calculate static friction effects
        max_static_friction = self.calculate_static_friction_threshold()
        # print(f"Max static friction: {max_static_friction}")

        # If no torque, check static friction
        if abs(torque.item()) < p.motor_deadzone:
            required_friction = abs(p.M * p.g * p.l * np.sin(theta))
            # print(f"Required static friction: {required_friction}")

            if required_friction <= max_static_friction:
                # Static friction keeps system rigid
                I_total = p.M * p.l**2 + p.I + 2 * (p.m * p.r**2 + p.i)
                theta_ddot = (p.M * p.g * p.l * np.sin(theta)) / I_total
                x_ddot = p.r * theta_ddot
                phi_ddot = -theta_ddot
            else:
                # Static friction exceeded
                theta_ddot = (p.M * p.g * p.l * np.sin(theta)) / (p.M * p.l**2 + p.I)
                x_ddot = (-p.M * p.l * theta_ddot * np.cos(theta)) / (p.M + 2 * p.m)
                phi_ddot = 0
        else:
            # Normal dynamics with applied torque
            effective_force = torque.item() / p.r
            theta_ddot = (p.M * p.g * p.l * np.sin(theta) - torque.item()) / (p.M * p.l**2 + p.I)
            x_ddot = (effective_force - p.M * p.l * theta_ddot * np.cos(theta)) / (p.M + 2 * p.m)
            phi_ddot = torque.item() / p.i

        # print(f"theta_ddot: {theta_ddot}, x_ddot: {x_ddot}, phi_ddot: {phi_ddot}")
        return np.array([theta_ddot, x_ddot, phi_ddot])

    def integrate_state(self, state: np.ndarray, accelerations: np.ndarray) -> np.ndarray:
        """Integrate state using semi-implicit Euler method.

        Args:
            state: Current state [theta, theta_dot, x, x_dot, phi, phi_dot]
            accelerations: Calculated accelerations [theta_ddot, x_ddot, phi_ddot]

        Returns:
            Updated state array
        """
        # Extract components
        theta, theta_dot = state[0], state[1]
        x, x_dot = state[2], state[3]
        phi, phi_dot = state[4], state[5]

        theta_ddot, x_ddot, phi_ddot = accelerations

        # Semi-implicit Euler integration
        dt = self.params.dt

        # Update velocities
        theta_dot_new = theta_dot + theta_ddot * dt
        x_dot_new = x_dot + x_ddot * dt
        phi_dot_new = phi_dot + phi_ddot * dt

        # Update positions
        theta_new = theta + theta_dot_new * dt
        x_new = x + x_dot_new * dt
        phi_new = phi + phi_dot_new * dt

        return np.array([theta_new, theta_dot_new, x_new, x_dot_new, phi_new, phi_dot_new])

    def get_energy(self, state: np.ndarray) -> float:
        """Calculate total energy of the system.

        Args:
            state: System state [theta, theta_dot, x, x_dot, phi, phi_dot]

        Returns:
            Total energy in Joules
        """
        p = self.params
        theta, theta_dot = state[0], state[1]
        x_dot = state[3]
        phi_dot = state[5]

        # Potential energy
        PE = p.M * p.g * p.l * (1 - np.cos(theta))

        # Kinetic energy
        # Body rotation
        KE_body_rot = 0.5 * p.I * theta_dot**2
        # Body translation
        KE_body_trans = 0.5 * p.M * x_dot**2
        # Wheels
        KE_wheels = p.m * x_dot**2 + 0.5 * p.i * phi_dot**2

        return PE + KE_body_rot + KE_body_trans + KE_wheels
