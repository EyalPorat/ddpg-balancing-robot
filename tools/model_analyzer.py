import argparse
import math
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import collections
from pathlib import Path
from tqdm import tqdm
import matplotlib as mpl
from matplotlib.lines import Line2D

# Add the project module to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from python.src.balancing_robot.models import Actor, SimNet
from python.src.balancing_robot.environment import BalancerEnv


class ModelAnalyzer:
    """Analyze trained balancing robot models with enhanced state support."""

    def __init__(self, model_path, env_config_path, ddpg_config_path=None, output_dir="analysis_results", device="cpu"):
        """Initialize model analyzer.

        Args:
            model_path: Path to saved model checkpoint
            env_config_path: Path to environment config file
            ddpg_config_path: Path to DDPG model config file (optional)
            output_dir: Directory to save analysis outputs
            device: Device to run model on
        """
        self.model_path = Path(model_path)
        self.env_config_path = Path(env_config_path)
        self.ddpg_config_path = Path(ddpg_config_path) if ddpg_config_path else None
        self.output_dir = Path(output_dir)
        self.device = device

        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load environment config
        with open(env_config_path, "r") as f:
            self.env_config = yaml.safe_load(f)

        # Load DDPG config if provided
        if self.ddpg_config_path and self.ddpg_config_path.exists():
            with open(self.ddpg_config_path, "r") as f:
                self.ddpg_config = yaml.safe_load(f)
        else:
            self.ddpg_config = None

        # Combined config for convenience (env config takes precedence)
        self.config = {}
        if self.ddpg_config:
            self.config.update(self.ddpg_config)
        self.config.update(self.env_config)

        # Get enhanced state parameters
        self.theta_history_size = self.env_config["observation"].get("theta_history_size", 3)
        self.theta_dot_history_size = self.env_config["observation"].get("theta_dot_history_size", 3)
        self.action_history_size = self.env_config["observation"].get("action_history_size", 4)
        self.motor_delay_steps = self.env_config["physics"].get("motor_delay_steps", 2)

        # Calculate enhanced state dimension
        # Basic state (2) + action history + theta history + theta_dot history
        self.enhanced_state_dim = 2 + self.action_history_size + self.theta_history_size + self.theta_dot_history_size
        print(f"Using enhanced state representation with dimension: {self.enhanced_state_dim}")

        # Initialize model with enhanced state size
        self.actor = self._load_model()

        # Initialize and load SimNet with enhanced state
        self.simnet = SimNet(state_dim=self.enhanced_state_dim, action_dim=1, hidden_dims=(32, 32, 32))
        self.simnet.load_state_dict(torch.load("python/notebooks/logs/simnet_training/simnet_final.pt")["state_dict"])
        # self.simnet.load_state_dict(torch.load("python/notebooks/logs/simnet_training/physics/best_simnet.pt")["state_dict"])
        # self.simnet.load_state_dict(torch.load("python/notebooks/logs/simnet_training/real/best_simnet.pt")["state_dict"])
        self.simnet.to(device)
        self.simnet.eval()  # Set to evaluation mode

        # Load environment for reference values
        self.env = BalancerEnv(config_path=env_config_path, simnet=self.simnet)

        # Verify state dimensions match
        env_state_dim = self.env.observation_space.shape[0]
        if env_state_dim != self.enhanced_state_dim:
            print(
                f"WARNING: Environment state dimension ({env_state_dim}) doesn't match calculated dimension ({self.enhanced_state_dim})"
            )
            print("This may cause issues with state processing. Please check your configuration.")
        else:
            print(f"State dimensions verified: {self.enhanced_state_dim}")

        # Get state ranges from environment config if available
        if "observation" in self.env_config:
            obs_config = self.env_config["observation"]
            termination_config = self.env_config["termination"]
            angle_limit = np.deg2rad(60)
            velocity_limit = 4

            self.theta_range = np.linspace(-angle_limit, angle_limit, 100)
            self.theta_dot_range = np.linspace(-velocity_limit, velocity_limit, 100)

            # For detailed phase plot (reduced range)
            self.phase_theta_range = np.linspace(-angle_limit * 0.66, angle_limit * 0.66, 20)
            self.phase_theta_dot_range = np.linspace(-velocity_limit * 0.5, velocity_limit * 0.5, 20)
        else:
            # Use reasonable defaults if not specified
            self.theta_range = np.linspace(-np.pi / 2, np.pi / 2, 100)  # -90 to 90 degrees
            self.theta_dot_range = np.linspace(-8, 8, 100)  # -8 to 8 rad/s

            # For detailed phase plot
            self.phase_theta_range = np.linspace(-np.pi / 3, np.pi / 3, 20)  # -60 to 60 degrees
            self.phase_theta_dot_range = np.linspace(-4, 4, 20)  # -4 to 4 rad/s

    def _load_model(self):
        """Load the model from checkpoint with enhanced state support."""
        # Get model architecture from DDPG config if available
        hidden_dims = (10, 10)  # Default architecture

        if self.ddpg_config and "model" in self.ddpg_config and "actor" in self.ddpg_config["model"]:
            hidden_dims = self.ddpg_config["model"]["actor"].get("hidden_dims", hidden_dims)

        # We always use max_action=1.0 for the actor - actions are normalized to [-1, 1]
        # Scaling to physical torque happens in the environment
        max_action = 1.0

        # Create actor model with enhanced state dimension
        actor = Actor(
            state_dim=self.enhanced_state_dim, action_dim=1, max_action=max_action, hidden_dims=hidden_dims
        ).to(self.device)

        print(
            f"Loading model with architecture: state_dim={self.enhanced_state_dim}, action_dim=1, hidden_dims={hidden_dims}"
        )
        print(
            f"Enhanced state structure: theta, theta_dot, action_history[{self.action_history_size}], theta_history[{self.theta_history_size}], theta_dot_history[{self.theta_dot_history_size}]"
        )
        print(f"Using normalized action space with max_action=1.0")

        # Load weights
        checkpoint = torch.load(self.model_path, map_location=torch.device(self.device))

        # Print the model state_dict keys to help diagnose any issues
        try:
            actor.load_state_dict(checkpoint["state_dict"])
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Model state dictionary keys:")
            for key in checkpoint["state_dict"].keys():
                print(f"  {key}")
            print("Expected keys for current architecture:")
            for key in actor.state_dict().keys():
                print(f"  {key}")
            raise

        actor.eval()
        return actor

    def _create_enhanced_state(self, theta, theta_dot, action_history=None, theta_history=None, theta_dot_history=None):
        """Create an enhanced state vector with proper history elements."""
        # Initialize histories if not provided
        if action_history is None:
            action_history = np.zeros(self.action_history_size)
        if theta_history is None:
            theta_history = np.ones(self.theta_history_size) * theta
        if theta_dot_history is None:
            theta_dot_history = np.ones(self.theta_dot_history_size) * theta_dot

        # Combine into enhanced state
        enhanced_state = np.concatenate(
            [
                np.array([theta, theta_dot]),  # Basic state
                np.array(action_history).flatten(),  # Action history
                np.array(theta_history).flatten(),  # Theta history
                np.array(theta_dot_history).flatten(),  # Theta_dot history
            ]
        )

        # Verify state dimension
        expected_dim = 2 + self.action_history_size + self.theta_history_size + self.theta_dot_history_size
        if len(enhanced_state) != expected_dim:
            raise ValueError(f"Enhanced state dimension mismatch. Got {len(enhanced_state)}, expected {expected_dim}")

        return enhanced_state

    def predict_actions(self, states):
        """Predict actions for given states, handling enhanced state representation."""
        with torch.no_grad():
            # Convert to enhanced states if necessary
            enhanced_states = []
            for state in states:
                if len(state) < self.enhanced_state_dim:
                    # If basic state provided, expand to enhanced state
                    if len(state) == 2:
                        theta, theta_dot = state
                        enhanced_state = self._create_enhanced_state(theta, theta_dot)
                    else:
                        raise ValueError(f"Unexpected state dimension: {len(state)}")
                    enhanced_states.append(enhanced_state)
                else:
                    # Already enhanced
                    enhanced_states.append(state)

            enhanced_states = np.array(enhanced_states)
            enhanced_states_tensor = torch.FloatTensor(enhanced_states).to(self.device)
            actions = self.actor(enhanced_states_tensor).cpu().numpy()

        return actions

    def create_action_heatmap(self, action_history=None):
        """Create heatmap of actions across the state space for a fixed action history."""
        if action_history is None:
            action_history = np.zeros(self.action_history_size)

        print(f"Creating action heatmap...")

        # Create meshgrid of states
        theta_mesh, theta_dot_mesh = np.meshgrid(self.theta_range, self.theta_dot_range)

        # Create enhanced states for each grid point
        states = []
        for theta, theta_dot in zip(theta_mesh.flatten(), theta_dot_mesh.flatten()):
            enhanced_state = self._create_enhanced_state(theta, theta_dot, action_history)
            states.append(enhanced_state)

        states = np.array(states)

        # Predict actions
        actions = self.predict_actions(states)
        action_mesh = actions.reshape(theta_mesh.shape)

        # Plot heatmap
        plt.figure(figsize=(12, 10))

        # Convert theta to degrees for readability
        theta_degrees = self.theta_range * 180 / np.pi

        heatmap = plt.pcolormesh(theta_degrees, self.theta_dot_range, action_mesh, cmap="coolwarm", shading="auto")
        plt.colorbar(heatmap, label="Action (normalized [-1, 1])")
        plt.xlabel("Angle θ (degrees)")
        plt.ylabel("Angular Velocity θ̇ (rad/s)")
        plt.title(f"Controller Action Map")

        # Add contour lines
        contour = plt.contour(
            theta_degrees, self.theta_dot_range, action_mesh, levels=10, colors="black", alpha=0.5, linewidths=0.5
        )
        plt.clabel(contour, inline=True, fontsize=8)

        # Mark zero action contour distinctly
        zero_contour = plt.contour(
            theta_degrees, self.theta_dot_range, action_mesh, levels=[0], colors="green", linewidths=2
        )
        plt.clabel(zero_contour, inline=True, fontsize=10, fmt="0")

        plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"action_heatmap.png", dpi=300)

        print("Action heatmap created and saved.")

    def create_action_heatmaps_for_multiple_action_histories(self):
        """Create a 3×3 grid of action heatmaps for different action histories."""
        sns.set_style("white")
        action_values = np.linspace(-1.0, 1.0, 9)
        rows = cols = 3
        vmin, vmax = -1.0, 1.0

        fig, axes = plt.subplots(rows, cols, figsize=(12, 12), sharex=True, sharey=True, constrained_layout=True)

        theta_deg = self.theta_range * 180.0 / np.pi

        # Plot each panel
        for idx, action_value in enumerate(action_values):
            ax = axes.flat[idx]
            Θ, 𝑤 = np.meshgrid(self.theta_range, self.theta_dot_range)

            # Create action history filled with the current action value
            action_history = np.ones(self.action_history_size) * action_value

            # Create enhanced states
            states = []
            for theta, theta_dot in zip(Θ.flatten(), 𝑤.flatten()):
                enhanced_state = self._create_enhanced_state(theta, theta_dot, action_history)
                states.append(enhanced_state)

            states = np.array(states)

            # Get predictions
            data = self.predict_actions(states).reshape(Θ.shape)

            im = ax.pcolormesh(
                theta_deg, self.theta_dot_range, data, cmap="coolwarm", shading="auto", vmin=vmin, vmax=vmax
            )

            # smaller individual titles, nudged down
            ax.set_title(f"action_history = {action_value:.2f}", fontsize=8, pad=4)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.7)

        # Global labels
        fig.supxlabel("θ (deg)", fontsize=11, y=0.02)
        fig.supylabel("θ̇ (rad/s)", fontsize=11, x=0.02, rotation=90)

        # Single colorbar on the right
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location="right", fraction=0.02, pad=0.02)
        cbar.set_label("Action (normalized [-1,1])", fontsize=10)

        # Save & close
        fig.savefig(self.output_dir / "action_heatmaps_multiple_action_histories.png", dpi=300)
        plt.close(fig)

    def create_phase_space_plot(self):
        """Create phase space plot with velocity fields using SimNet with enhanced state."""
        print("Creating phase space plot...")

        # Create meshgrid for phase space
        theta_mesh, theta_dot_mesh = np.meshgrid(self.phase_theta_range, self.phase_theta_dot_range)

        # Create enhanced states for each grid point
        states = []
        for theta, theta_dot in zip(theta_mesh.flatten(), theta_dot_mesh.flatten()):
            enhanced_state = self._create_enhanced_state(theta, theta_dot)
            states.append(enhanced_state)

        states = np.array(states)

        # Predict actions
        actions = self.predict_actions(states)

        # Calculate accelerations using SimNet
        accelerations = []
        for state, action in zip(states, actions):
            # Convert to tensors for SimNet
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)

            # Get delta predictions from SimNet
            with torch.no_grad():
                delta_state = self.simnet.predict_delta(state_tensor, action_tensor)
                delta_theta_dot = delta_state[0, 1].item()  # Grab the angular velocity delta

            # Calculate acceleration as delta_theta_dot divided by dt
            dt = self.env.physics.params.dt
            acceleration = delta_theta_dot / dt

            accelerations.append(acceleration)

        accelerations = np.array(accelerations).reshape(theta_mesh.shape)

        # Calculate velocity components for phase space
        # Here, u represents the change in theta (which is just theta_dot)
        # and v represents the change in theta_dot (which is theta_ddot or acceleration)
        u = theta_dot_mesh
        v = accelerations

        # Calculate velocity magnitude for coloring
        speed = np.sqrt(u**2 + v**2)

        # Normalize vectors for visualization
        norm = np.sqrt(u**2 + v**2)
        u_norm = u / np.where(norm > 1e-8, norm, 1e-8)  # Avoid division by zero
        v_norm = v / np.where(norm > 1e-8, norm, 1e-8)

        # Plot phase space
        plt.figure(figsize=(12, 10))

        # Convert theta to degrees for readability
        theta_degrees = self.phase_theta_range * 180 / np.pi

        # Plot vector field
        plt.quiver(
            theta_degrees, self.phase_theta_dot_range, u_norm, v_norm, speed, cmap="viridis", scale=25, width=0.003
        )

        plt.colorbar(label="Velocity Magnitude")
        plt.xlabel("Angle θ (degrees)")
        plt.ylabel("Angular Velocity θ̇ (rad/s)")
        plt.title("Phase Space Dynamics (Enhanced State SimNet)")
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "phase_space.png", dpi=300)

        print("Phase space plot created and saved.")

    def create_delta_visualization(self):
        """Create visualization of the predicted deltas across the state space."""
        print("Creating delta visualization...")

        # Create meshgrid for state space
        theta_mesh, theta_dot_mesh = np.meshgrid(self.theta_range, self.theta_dot_range)

        # Prepare states for prediction
        states = np.column_stack(
            (
                theta_mesh.flatten(),
                theta_dot_mesh.flatten(),
                np.zeros(theta_mesh.size),  # Initialize prev_action to zero
            )
        )

        # Predict actions using the actor network
        actions = self.predict_actions(states)

        # Predict deltas using SimNet
        delta_thetas = []
        delta_theta_dots = []

        for state, action in zip(states, actions):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)

            with torch.no_grad():
                delta_state = self.simnet.predict_delta(state_tensor, action_tensor)
                delta_thetas.append(delta_state[0, 0].item())  # Delta in theta
                delta_theta_dots.append(delta_state[0, 1].item())  # Delta in theta_dot

        # Reshape for visualization
        delta_theta_mesh = np.array(delta_thetas).reshape(theta_mesh.shape)
        delta_theta_dot_mesh = np.array(delta_theta_dots).reshape(theta_mesh.shape)

        # Create visualizations
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Convert theta to degrees for readability
        theta_degrees = self.theta_range * 180 / np.pi

        # Delta theta heatmap
        im1 = axes[0].pcolormesh(theta_degrees, self.theta_dot_range, delta_theta_mesh, cmap="RdBu", shading="auto")
        plt.colorbar(im1, ax=axes[0], label="Δθ (change in angle)")
        axes[0].set_xlabel("Angle θ (degrees)")
        axes[0].set_ylabel("Angular Velocity θ̇ (rad/s)")
        axes[0].set_title("Predicted Change in Angle (Δθ)")
        axes[0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
        axes[0].axvline(x=0, color="k", linestyle="--", alpha=0.3)

        # Delta theta_dot heatmap
        im2 = axes[1].pcolormesh(theta_degrees, self.theta_dot_range, delta_theta_dot_mesh, cmap="RdBu", shading="auto")
        plt.colorbar(im2, ax=axes[1], label="Δθ̇ (change in angular velocity)")
        axes[1].set_xlabel("Angle θ (degrees)")
        axes[1].set_ylabel("Angular Velocity θ̇ (rad/s)")
        axes[1].set_title("Predicted Change in Angular Velocity (Δθ̇)")
        axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.3)
        axes[1].axvline(x=0, color="k", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "delta_visualization.png", dpi=300)

        print("Delta visualization created and saved.")

    def create_3d_action_surface(self):
        """Create 3D surface plot of actions."""
        print("Creating 3D action surface...")

        # Create meshgrid of states
        theta_mesh, theta_dot_mesh = np.meshgrid(self.theta_range, self.theta_dot_range)
        states = np.column_stack(
            (
                theta_mesh.flatten(),
                theta_dot_mesh.flatten(),
                # np.zeros(theta_mesh.size),  # Initialize prev_action to zero
            )
        )

        # Predict actions
        actions = self.predict_actions(states)
        action_mesh = actions.reshape(theta_mesh.shape)

        # Create 3D plot
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection="3d")

        # Convert theta to degrees for readability
        theta_degrees = self.theta_range * 180 / np.pi

        # Create surface plot
        surf = ax.plot_surface(
            np.meshgrid(theta_degrees, self.theta_dot_range)[0],
            np.meshgrid(theta_degrees, self.theta_dot_range)[1],
            action_mesh,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=True,
            alpha=0.8,
        )

        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Action (normalized [-1, 1])")

        # Add zero plane
        ax.contour(
            np.meshgrid(theta_degrees, self.theta_dot_range)[0],
            np.meshgrid(theta_degrees, self.theta_dot_range)[1],
            action_mesh,
            levels=[0],
            colors="green",
            linewidths=2,
        )

        # Set labels and title
        ax.set_xlabel("Angle θ (degrees)")
        ax.set_ylabel("Angular Velocity θ̇ (rad/s)")
        ax.set_zlabel("Action (normalized [-1, 1])")
        ax.set_title("3D Controller Action Surface")

        # Add reference planes
        xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        ax.plot([xlim[0], xlim[1]], [0, 0], [0, 0], "k--", alpha=0.3)
        ax.plot([0, 0], [ylim[0], ylim[1]], [0, 0], "k--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "action_surface_3d.png", dpi=300)

        print("3D action surface created and saved.")

    def analyze_stability_regions(self):
        """Analyze stability regions by simulating trajectories using SimNet."""
        print("Analyzing stability regions...")

        # Define a grid of initial conditions
        thetas = np.linspace(np.deg2rad(-50), np.deg2rad(50), 20)
        theta_dots = np.linspace(-4, 4, 20)

        # Initialize grid for stability tracking (True if state converges to balance)
        stability_grid = np.zeros((len(thetas), len(theta_dots)))

        # Track how many steps it takes to stabilize
        steps_to_stabilize = np.zeros((len(thetas), len(theta_dots)))
        steps_to_stabilize.fill(np.nan)  # Initialize with NaN for non-convergent trajectories

        # Simulation parameters
        max_steps = 500
        stable_threshold_theta = np.deg2rad(6)
        stable_threshold_theta_dot = np.deg2rad(20)

        # Simulate trajectories from each initial condition
        for i, theta in enumerate(tqdm(thetas, desc="Simulating trajectories")):
            for j, theta_dot in enumerate(theta_dots):
                # Initialize state with a full 3-element vector for SimNet
                state, _ = self.env.reset(
                    state=np.array([theta, theta_dot, 0.0])
                )  # Reset environment to the current state

                # Keep track of last 10 theta_dot values
                theta_dot_history = [theta_dot] * 2  # Initialize with initial value

                # Simulate trajectory using SimNet
                for step in range(max_steps):
                    # Get action from actor network
                    action = self.predict_actions([state])[0]

                    # Predict next state using SimNet
                    next_state, _, _, _, _ = self.env.step(action)

                    # Update theta_dot history
                    theta_dot_history.pop(0)  # Remove oldest
                    theta_dot_history.append(next_state[1])  # Add newest

                    # Check if balanced using average of last 10 readings for angular velocity
                    if (
                        abs(next_state[0] + np.deg2rad(7.5)) < stable_threshold_theta
                        and abs(np.mean(theta_dot_history)) < stable_threshold_theta_dot
                    ):
                        stability_grid[i, j] = 1
                        steps_to_stabilize[i, j] = step
                        break

                    state = next_state

                    # Check if state is diverging
                    if abs(state[0]) > np.pi:
                        break

        # Create stability region plot
        plt.figure(figsize=(12, 10))

        # Convert theta to degrees for readability
        theta_degrees = thetas * 180 / np.pi

        plt.pcolormesh(theta_degrees, theta_dots, stability_grid.T, cmap="RdYlGn", alpha=0.7, shading="auto")

        plt.colorbar(label="Convergence (1=stable)")
        plt.xlabel("Initial Angle θ (degrees)")
        plt.ylabel("Initial Angular Velocity θ̇ (rad/s)")
        plt.title("Controller Stability Regions (SimNet)")
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "stability_regions.png", dpi=300)

        # Create steps-to-stabilize plot
        plt.figure(figsize=(12, 10))

        # Use a masked array to handle the NaN values
        masked_steps = np.ma.masked_invalid(steps_to_stabilize)

        plt.pcolormesh(theta_degrees, theta_dots, masked_steps.T, cmap="viridis", shading="auto")

        plt.colorbar(label="Steps to Stabilize")
        plt.xlabel("Initial Angle θ (degrees)")
        plt.ylabel("Initial Angular Velocity θ̇ (rad/s)")
        plt.title("Steps to Stabilize (SimNet)")
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "steps_to_stabilize.png", dpi=300)

        print("Stability analysis completed and saved.")

    def analyze_simulated_trajectories(self):
        """Analyze system behavior for simulated trajectories from various initial conditions."""
        print("Analyzing simulated trajectories...")

        # Define different initial conditions for simulation
        initial_conditions = [
            # Starting positions with zero velocity
            (np.pi / 12, 0.0, "Small angle +15°"),
            (np.pi / 6, 0.0, "Medium angle +30°"),
            (np.pi / 4, 0.0, "Large angle +45°"),
            (-np.pi / 12, 0.0, "Small angle -15°"),
            (-np.pi / 6, 0.0, "Medium angle -30°"),
            (-np.pi / 4, 0.0, "Large angle -45°"),
            # Pure angular velocities
            (0.0, 1.0, "Small velocity +1 rad/s"),
            (0.0, 2.0, "Medium velocity +2 rad/s"),
            (0.0, 4.0, "Large velocity +4 rad/s"),
            (0.0, -1.0, "Small velocity -1 rad/s"),
            (0.0, -2.0, "Medium velocity -2 rad/s"),
            (0.0, -4.0, "Large velocity -4 rad/s"),
            # Combined states - same signs
            (np.pi / 12, 1.0, "Small +angle +velocity"),
            (np.pi / 6, 2.0, "Medium +angle +velocity"),
            (-np.pi / 12, -1.0, "Small -angle -velocity"),
            (-np.pi / 6, -2.0, "Medium -angle -velocity"),
            # Combined states - opposite signs
            (np.pi / 12, -1.0, "Small +angle -velocity"),
            (np.pi / 6, -2.0, "Medium +angle -velocity"),
            (-np.pi / 12, 1.0, "Small -angle +velocity"),
            (-np.pi / 6, 2.0, "Medium -angle +velocity"),
            # Near balanced states
            (np.pi / 36, 0.1, "Near balanced +angle"),
            (-np.pi / 36, -0.1, "Near balanced -angle"),
            (np.pi / 72, 0.05, "Very near balanced"),
        ]

        # Simulation parameters
        max_steps = 200
        trajectory_data = []

        # Simulate trajectory for each initial condition
        for theta, theta_dot, label in initial_conditions:
            state, _ = self.env.reset(state=np.array([theta, theta_dot]))  # Reset environment

            # Initialize trajectory
            trajectory = {"label": label, "states": [state.copy()], "actions": [], "time": [0.0]}

            # Simulate
            current_time = 0.0
            for _ in range(max_steps):
                # Get action
                action = self.predict_actions([state])[0]

                # Store action
                trajectory["actions"].append(action[0])

                next_state, _, _, _, _ = self.env.step(action)

                # Update time
                current_time += self.env.physics.params.dt

                # Store state
                trajectory["states"].append(next_state.copy())
                trajectory["time"].append(current_time)

                # Update for next iteration
                state = next_state

                # Break if robot has fallen
                if abs(state[0]) > np.pi / 2:
                    break

            # Convert lists to numpy arrays
            trajectory["states"] = np.array(trajectory["states"])
            trajectory["actions"] = np.array(trajectory["actions"])

            # Store trajectory
            trajectory_data.append(trajectory)

        # Plot trajectories
        plt.figure(figsize=(14, 8))

        for trajectory in trajectory_data:
            plt.plot(trajectory["time"], trajectory["states"][:, 0] * 180 / np.pi, label=trajectory["label"])

        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Angle θ (degrees)")
        plt.title("Simulated Angle Trajectories")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "angle_trajectories.png")

        # Plot angular velocities
        plt.figure(figsize=(14, 8))

        for trajectory in trajectory_data:
            plt.plot(trajectory["time"], trajectory["states"][:, 1], label=trajectory["label"])

        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity θ̇ (rad/s)")
        plt.title("Simulated Angular Velocity Trajectories")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "angular_velocity_trajectories.png")

        # Plot control actions
        plt.figure(figsize=(14, 8))

        for trajectory in trajectory_data:
            plt.plot(
                trajectory["time"][:-1],  # One fewer action than states
                trajectory["actions"],
                label=trajectory["label"],
            )

        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Action (normalized [-1, 1])")
        plt.title("Controller Actions")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "action_trajectories.png")

        # Create phase portrait
        plt.figure(figsize=(12, 12))

        for trajectory in trajectory_data:
            plt.plot(
                trajectory["states"][:, 0] * 180 / np.pi,  # Convert to degrees
                trajectory["states"][:, 1],
                label=trajectory["label"],
            )

            # Mark start point
            plt.scatter(
                trajectory["states"][0, 0] * 180 / np.pi,
                trajectory["states"][0, 1],
                marker="o",
                s=100,
                facecolors="none",
                edgecolors="black",
            )

        plt.grid(True)
        plt.xlabel("Angle θ (degrees)")
        plt.ylabel("Angular Velocity θ̇ (rad/s)")
        plt.title("Phase Portrait")
        plt.legend(loc="upper right")
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "phase_portrait.png")

        fig, axs = plt.subplots(3, 1, figsize=(14, 18))

        # 1) Angle θ trajectories
        for traj in trajectory_data:
            axs[0].plot(traj["time"], traj["states"][:, 0] * 180 / np.pi, label=traj["label"])
        axs[0].set_title("Angle θ Trajectories")
        axs[0].set_ylabel("θ (deg)")
        axs[0].grid(True)

        # 2) Angular velocity θ̇ trajectories
        for traj in trajectory_data:
            axs[1].plot(traj["time"], traj["states"][:, 1], label=traj["label"])
        axs[1].set_title("Angular Velocity θ̇ Trajectories")
        axs[1].set_ylabel("θ̇ (rad/s)")
        axs[1].grid(True)

        # 3) Control action trajectories
        for traj in trajectory_data:
            axs[2].plot(traj["time"][:-1], traj["actions"], label=traj["label"])
        axs[2].set_title("Controller Action Trajectories")
        axs[2].set_ylabel("Action (normalized)")
        axs[2].set_xlabel("Time (s)")
        axs[2].grid(True)

        # unified legend and layout
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        plt.tight_layout()
        fig.savefig(self.output_dir / "combined_trajectory_types.png", dpi=300)
        plt.close(fig)
        print("Combined trajectory‐type plot saved.")

    def generate_comparative_pd_controller(self):
        """Generate a classical PD controller for comparison."""
        print("Generating comparative PD controller...")

        # Choose reasonable PD gains that achieve similar performance to the DDPG
        # These would typically be tuned for the specific robot
        kp = 20.0  # Proportional gain
        kd = 1  # Derivative gain

        # Create lambda function for PD controller
        # Output normalized to [-1, 1] range for fair comparison with DDPG
        pd_controller = lambda theta, theta_dot: np.clip((kp * theta + kd * theta_dot), -1.0, 1.0)

        # Create state grid for comparison
        theta_mesh, theta_dot_mesh = np.meshgrid(self.theta_range, self.theta_dot_range)
        states = np.column_stack((theta_mesh.flatten(), theta_dot_mesh.flatten()))

        # Predict DDPG controller actions
        ddpg_actions = self.predict_actions(states)
        ddpg_action_mesh = ddpg_actions.reshape(theta_mesh.shape)

        # Calculate PD controller actions
        pd_actions = np.array([pd_controller(s[0], s[1]) for s in states])
        pd_action_mesh = pd_actions.reshape(theta_mesh.shape)

        # Calculate difference between controllers
        difference = ddpg_action_mesh - pd_action_mesh

        # Plot DDPG vs PD controller comparison
        fig, axs = plt.subplots(2, 2, figsize=(16, 14))

        # Convert theta to degrees for readability
        theta_degrees = self.theta_range * 180 / np.pi

        # DDPG controller heatmap
        im1 = axs[0, 0].pcolormesh(
            theta_degrees,
            self.theta_dot_range,
            ddpg_action_mesh,
            cmap="coolwarm",
            shading="auto",
            vmin=-np.max(np.abs(ddpg_action_mesh)),
            vmax=np.max(np.abs(ddpg_action_mesh)),
        )
        plt.colorbar(im1, ax=axs[0, 0], label="Action (normalized [-1, 1])")
        axs[0, 0].set_title("DDPG Controller")
        axs[0, 0].set_xlabel("Angle θ (degrees)")
        axs[0, 0].set_ylabel("Angular Velocity θ̇ (rad/s)")
        axs[0, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
        axs[0, 0].axvline(x=0, color="k", linestyle="--", alpha=0.3)

        # PD controller heatmap
        im2 = axs[0, 1].pcolormesh(
            theta_degrees,
            self.theta_dot_range,
            pd_action_mesh,
            cmap="coolwarm",
            shading="auto",
            vmin=-np.max(np.abs(pd_action_mesh)),
            vmax=np.max(np.abs(pd_action_mesh)),
        )
        plt.colorbar(im2, ax=axs[0, 1], label="Torque")
        # Mark zero action contour distinctly for PD controller
        zero_contour_pd = axs[0, 1].contour(
            theta_degrees, self.theta_dot_range, pd_action_mesh, levels=[0], colors="green", linewidths=2
        )
        axs[0, 1].clabel(zero_contour_pd, inline=True, fontsize=10, fmt="0")
        axs[0, 1].set_title(f"PD Controller (Kp={kp}, Kd={kd})")
        axs[0, 1].set_xlabel("Angle θ (degrees)")
        axs[0, 1].set_ylabel("Angular Velocity θ̇ (rad/s)")
        axs[0, 1].axhline(y=0, color="k", linestyle="--", alpha=0.3)
        axs[0, 1].axvline(x=0, color="k", linestyle="--", alpha=0.3)

        # Difference heatmap
        im3 = axs[1, 0].pcolormesh(
            theta_degrees,
            self.theta_dot_range,
            difference,
            cmap="RdBu",
            shading="auto",
            vmin=-np.max(np.abs(difference)),
            vmax=np.max(np.abs(difference)),
        )
        plt.colorbar(im3, ax=axs[1, 0], label="Torque Difference")
        axs[1, 0].set_title("DDPG - PD Difference")
        axs[1, 0].set_xlabel("Angle θ (degrees)")
        axs[1, 0].set_ylabel("Angular Velocity θ̇ (rad/s)")
        axs[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
        axs[1, 0].axvline(x=0, color="k", linestyle="--", alpha=0.3)

        # Line comparison plot for theta = 0
        theta_zero_idx = len(theta_degrees) // 2
        axs[1, 1].plot(self.theta_dot_range, ddpg_action_mesh[:, theta_zero_idx], label="DDPG")
        axs[1, 1].plot(self.theta_dot_range, pd_action_mesh[:, theta_zero_idx], label="PD", linestyle="--")
        axs[1, 1].set_title("Controller Comparison (θ = 0°)")
        axs[1, 1].set_xlabel("Angular Velocity θ̇ (rad/s)")
        axs[1, 1].set_ylabel("Action (torque)")
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "ddpg_vs_pd.png", dpi=300)

        print("Controller comparison completed and saved.")

    def create_trajectory_heatmap_overlay(self, max_steps=200, grid_size=10):
        """Plot action‐heatmap with colored trajectories, arrows, and start/end markers."""

        # — 1) Heatmap background
        theta_mesh, theta_dot_mesh = np.meshgrid(self.theta_range, self.theta_dot_range)

        # Create enhanced states for heatmap
        states = []
        for theta, theta_dot in zip(theta_mesh.flatten(), theta_dot_mesh.flatten()):
            enhanced_state = self._create_enhanced_state(theta, theta_dot)
            states.append(enhanced_state)

        states = np.array(states)

        # Get action predictions
        action_mesh = self.predict_actions(states).reshape(theta_mesh.shape)

        fig, ax = plt.subplots(figsize=(12, 10))
        theta_deg = self.theta_range * 180.0 / np.pi

        hm = ax.pcolormesh(theta_deg, self.theta_dot_range, action_mesh, cmap="coolwarm", shading="gouraud", alpha=0.7)
        cbar = fig.colorbar(hm, ax=ax, pad=0.02)
        cbar.set_label("Action (norm. –1…1)", rotation=270, labelpad=15)

        # Zero-action contour
        ax.contour(
            theta_deg,
            self.theta_dot_range,
            action_mesh,
            levels=[0],
            colors="white",
            linestyles="--",
            linewidths=1.5,
            alpha=0.7,
        )

        # — 2) Trajectory starts on a grid
        θ_min, θ_max = theta_deg.min(), theta_deg.max()
        ω_min, ω_max = self.theta_dot_range.min(), self.theta_dot_range.max()

        thetas0 = np.linspace(np.deg2rad(-60), np.deg2rad(60), grid_size)
        ωs0 = np.linspace(-4, 4, grid_size)
        starts = [(θ0, ω0) for θ0 in thetas0 for ω0 in ωs0]

        max_delta = getattr(self.env, 'max_delta', 0.25)  # Default to 0.25 if not specified
        
        cmap = mpl.cm.get_cmap("tab10")
        for idx, (θ0, ω0) in enumerate(starts):
            color = cmap(idx % cmap.N)
            traj = []

            # Initial enhanced state
            enhanced_state = self._create_enhanced_state(θ0, ω0)

            # Initialize history buffers
            theta_history = collections.deque([θ0] * self.theta_history_size, maxlen=self.theta_history_size)
            theta_dot_history = collections.deque([ω0] * self.theta_history_size, maxlen=self.theta_history_size)
            
            # Track current motor command through simulation
            current_motor_command = self.env.action_history[-1] if self.env.action_history else 0.0

            for _ in range(max_steps):
                # Get current basic state values
                theta = enhanced_state[0]
                theta_dot = enhanced_state[1]

                # Convert theta to degrees for plotting
                ang_d = theta * 180.0 / np.pi
                vel = theta_dot

                # Check if still in range
                if not (θ_min <= ang_d <= θ_max and ω_min <= vel <= ω_max):
                    break

                traj.append((ang_d, vel))

                with torch.no_grad():
                    # Get action (delta) from actor network
                    state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0).to(self.device)
                    delta_action = self.actor(state_tensor).cpu().numpy()[0][0]
                    
                    # Scale delta by max_delta
                    delta_action *= max_delta
                    
                    # Add to current command and clip to valid range
                    new_motor_command = np.clip(current_motor_command + delta_action, -1.0, 1.0)
                    
                    # Save for next iteration
                    current_motor_command = new_motor_command

                # Update history buffers
                theta_history.append(theta)
                theta_dot_history.append(theta_dot)

                # Use the corrected action (new_motor_command) for SimNet, not the delta
                action_tensor = torch.FloatTensor([[current_motor_command]]).to(self.device)

                with torch.no_grad():
                    next_state = self.simnet(state_tensor, action_tensor).cpu().numpy()[0]

                # Update state for next iteration
                enhanced_state = next_state
                
                # Break if robot has fallen (like in first function)
                if abs(enhanced_state[0]) > np.pi / 2:
                    break

            if len(traj) < 2:
                continue

            xs, ys = zip(*traj)
            ax.plot(xs, ys, color=color, lw=1.5, alpha=0.8)

            # arrows at regular intervals
            N = max(1, len(xs) // 8)
            ax.quiver(
                xs[:-1:N],
                ys[:-1:N],
                np.diff(xs)[::N],
                np.diff(ys)[::N],
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.003,
                color=color,
                alpha=0.8,
            )

            # start/end markers
            ax.scatter(xs[0], ys[0], marker="o", color=color, edgecolor="k", s=50, zorder=5)
            ax.scatter(xs[-1], ys[-1], marker=">", color=color, s=40, zorder=5)

        # — 3) Final styling
        ax.set_axisbelow(True)
        ax.grid(color="white", linestyle="--", linewidth=0.5, alpha=0.5)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        ax.set_xlabel("Angle θ (deg)")
        ax.set_ylabel("Angular Velocity θ̇ (rad/s)")
        ax.set_title(f"Action Heatmap with Trajectories")

        plt.tight_layout()
        plt.savefig(self.output_dir / f"traj_over_heatmap.png", dpi=300)
        plt.close(fig)
        print("Trajectory-overlay heatmap saved.")

    def create_natural_trajectory_overlay(self, grid_size=10, max_steps=100):
        """Plot trajectory heatmap without any control actions (natural dynamics only)."""
        print("Creating natural dynamics trajectory overlay...")

        fig, ax = plt.subplots(figsize=(12, 10))

        # Define state ranges for visualization
        θ_min, θ_max = self.theta_range.min() * 180 / np.pi, self.theta_range.max() * 180 / np.pi
        ω_min, ω_max = self.theta_dot_range.min(), self.theta_dot_range.max()

        # Set up the background with grid
        ax.set_facecolor("#f8f8f8")  # Light gray background
        ax.grid(color="white", linestyle="--", linewidth=0.8, alpha=0.7)

        # Add zero-axes for reference
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=1.0, alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=1.0, alpha=0.5)

        # Create grid of starting points
        thetas0 = np.linspace(self.theta_range.min(), self.theta_range.max(), grid_size)
        ωs0 = np.linspace(self.theta_dot_range.min(), self.theta_dot_range.max(), grid_size)
        starts = [(θ0, ω0) for θ0 in thetas0 for ω0 in ωs0]

        # Setup colormap for trajectories
        cmap = mpl.cm.get_cmap("viridis")

        # Track endpoints to visualize basin of attraction
        endpoints = []

        # Track trajectories
        for idx, (θ0, ω0) in enumerate(starts):
            color = cmap(idx % cmap.N)
            traj = []

            # Initialize enhanced state with zero action
            enhanced_state = self._create_enhanced_state(θ0, ω0)

            # Initialize history buffers
            theta_history = collections.deque([θ0] * self.theta_history_size, maxlen=self.theta_history_size)
            theta_dot_history = collections.deque([ω0] * self.theta_history_size, maxlen=self.theta_history_size)
            action_history = collections.deque([0.0] * self.action_history_size, maxlen=self.action_history_size)

            for step in range(max_steps):
                # Get current state values
                theta = enhanced_state[0]
                theta_dot = enhanced_state[1]

                # Convert to degrees for plotting
                ang_d = theta * 180.0 / np.pi
                vel = theta_dot

                # Check if still in drawable range
                if not (θ_min <= ang_d <= θ_max and ω_min <= vel <= ω_max):
                    break

                traj.append((ang_d, vel))

                # Apply ZERO action (natural dynamics)
                action = 0.0

                # Update history buffers
                theta_history.append(theta)
                theta_dot_history.append(theta_dot)
                action_history.appendleft(action)  # Add newest action at front

                # Predict next state using SimNet with zero action
                state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0).to(self.device)
                action_tensor = torch.FloatTensor([[action]]).to(self.device)

                with torch.no_grad():
                    next_state = self.simnet(state_tensor, action_tensor).cpu().numpy()[0]

                enhanced_state = next_state

                # Check if state is diverging
                if abs(enhanced_state[0]) > np.pi:
                    break

            # Only plot if we have enough points
            if len(traj) < 2:
                continue

            # Store endpoint
            if len(traj) > 0:
                endpoints.append(traj[-1])

            # Plot the trajectory
            xs, ys = zip(*traj)
            ax.plot(xs, ys, color=color, lw=1.5, alpha=0.8)

            # Add arrows to show direction
            if len(xs) > 10:
                # Add more arrows for longer trajectories
                N = max(1, len(xs) // 8)
                ax.quiver(
                    xs[:-1:N],
                    ys[:-1:N],
                    np.diff(xs)[::N],
                    np.diff(ys)[::N],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.003,
                    color=color,
                    alpha=0.8,
                )

            # Mark start/end points
            ax.scatter(xs[0], ys[0], marker="o", color="white", edgecolor=color, s=50, zorder=5, linewidth=1.5)
            ax.scatter(xs[-1], ys[-1], marker="x", color=color, s=40, zorder=5, linewidth=2)

        # Draw contour around endpoints to visualize basin of attraction if we have any endpoints
        if endpoints:
            try:
                endpoints = np.array(endpoints)
                # Only attempt if we have enough points
                if len(endpoints) > 5:
                    from scipy.spatial import ConvexHull

                    hull = ConvexHull(endpoints)
                    hull_points = endpoints[hull.vertices]
                    ax.plot(hull_points[:, 0], hull_points[:, 1], "r--", lw=2, alpha=0.6)
            except Exception as e:
                # Silently continue if we can't create the hull
                pass

        # Final styling
        ax.set_xlabel("Angle θ (deg)")
        ax.set_ylabel("Angular Velocity θ̇ (rad/s)")
        ax.set_title("Natural Dynamics Trajectories (No Control) - Enhanced State")

        # Add legend explaining markers
        legend_elements = [
            Line2D([0], [0], color="k", lw=1.5, alpha=0.8, label="Trajectory"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="w",
                markeredgecolor="k",
                markersize=8,
                label="Starting Point",
            ),
            Line2D([0], [0], marker="x", color="k", markersize=8, label="Ending Point"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        plt.savefig(self.output_dir / "natural_dynamics_trajectories.png", dpi=300)
        plt.close(fig)

        print("Natural dynamics trajectory overlay saved.")

    def run_complete_analysis(self):
        """Run all analysis methods."""
        self.create_action_heatmap()
        self.create_action_heatmaps_for_multiple_action_histories()
        self.create_phase_space_plot()
        self.create_3d_action_surface()
        self.analyze_stability_regions()
        self.analyze_simulated_trajectories()
        self.create_trajectory_heatmap_overlay()
        self.create_natural_trajectory_overlay()

        print("\nComplete analysis finished. Results saved to:", self.output_dir)

        # Create summary HTML page
        self.create_summary_page()

    def create_summary_page(self):
        """Create an HTML summary page with all generated plots."""
        print("Creating summary page...")

        # Get all image files
        image_files = sorted(list(self.output_dir.glob("*.png")))

        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced State Balancing Robot Model Analysis</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                h2 {
                    color: #555;
                    margin-top: 30px;
                }
                .image-container {
                    margin: 20px 0;
                    text-align: center;
                }
                img {
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 5px;
                    margin-bottom: 10px;
                }
                .caption {
                    font-style: italic;
                    color: #666;
                }
                .section {
                    margin-bottom: 40px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 20px;
                }
                .enhanced-state-info {
                    background-color: #f0f7ff;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
            </style>
        </head>
        <body>
            <h1>Enhanced State Balancing Robot Model Analysis</h1>
            <div class="enhanced-state-info">
                <p>Analysis of the trained DDPG controller using enhanced state representation. Enhanced state includes:</p>
                <ul>
                    <li>Basic state: theta, theta_dot</li>
                    <li>Action history: Last {0} motor commands</li>
                    <li>Theta history: Last {1} theta values</li>
                    <li>Theta_dot history: Last {2} theta_dot values</li>
                </ul>
                <p>All actions are normalized to [-1, 1] range, with motor delay compensation built into the control policy.</p>
            </div>
        """

        # Group images by category based on filename
        categories = {
            "response": "Controller Response Curves",
            "action": "Action Analysis",
            "phase": "Phase Space Analysis",
            "stability": "Stability Analysis",
            "sensitivity": "Sensitivity Analysis",
            "nonlinearity": "Nonlinearity Analysis",
            "trajectory": "Trajectory Analysis",
            "pd": "Comparison with PD Controller",
            "robustness": "Robustness Analysis",
        }

        for category, title in categories.items():
            related_images = [img for img in image_files if category in img.name.lower()]

            if related_images:
                html_content += f"""
                <div class="section">
                    <h2>{title}</h2>
                """

                for img in related_images:
                    # Create a readable caption from filename
                    caption = img.stem.replace("_", " ").title()

                    html_content += f"""
                    <div class="image-container">
                        <img src="{img.name}" alt="{caption}">
                        <div class="caption">{caption}</div>
                    </div>
                    """

                html_content += "</div>"

        html_content += """
        </body>
        </html>
        """

        # Write HTML file
        with open(self.output_dir / "analysis_summary.html", "w") as f:
            f.write(html_content)

        print("Summary page created at:", self.output_dir / "analysis_summary.html")


def main():
    parser = argparse.ArgumentParser(description="Analyze trained balancing robot models")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--output", type=str, default="analysis_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run model on (cpu or cuda)")
    parser.add_argument(
        "--analysis",
        type=str,
        default="all",
        choices=["all", "heatmap", "phase", "surface", "stability", "trajectory", "pd", "trajectory_heatmap_overlay"],
        help="Specific analysis to run",
    )

    args = parser.parse_args()

    analyzer = ModelAnalyzer(model_path=args.model, config_path=args.config, output_dir=args.output, device=args.device)

    if args.analysis == "all":
        analyzer.run_complete_analysis()
    elif args.analysis == "heatmap":
        analyzer.create_action_heatmap()
    elif args.analysis == "phase":
        analyzer.create_phase_space_plot()
    elif args.analysis == "surface":
        analyzer.create_3d_action_surface()
    elif args.analysis == "stability":
        analyzer.analyze_stability_regions()
    elif args.analysis == "trajectory":
        analyzer.analyze_simulated_trajectories()
    elif args.analysis == "pd":
        analyzer.generate_comparative_pd_controller()
    elif args.analysis == "trajectory_heatmap_overlay":
        analyzer.create_trajectory_heatmap_overlay()
        analyzer.create_natural_trajectory_overlay()


if __name__ == "__main__":
    main()
