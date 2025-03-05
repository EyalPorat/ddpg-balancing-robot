import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def create_episode_animation(states: np.ndarray, actions: np.ndarray, save_path: str = None, fps: int = 30):
    """
    Create an animation of the trained agent balancing.

    Args:
        states: Array of states (theta, theta_dot, x, x_dot, phi, phi_dot)
        actions: Array of actions (torque)
        save_path: Path to save the animation
        fps: Frames per second

    Returns:
        Animation object
    """
    # Determine fixed view limits
    x_positions = [s[2] for s in states]
    x_min, x_max = min(x_positions), max(x_positions)
    view_width = max(x_max - x_min + 0.4, 1.0)  # Add padding
    x_center = (x_min + x_max) / 2

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    def animate(i):
        ax.clear()
        state = states[i]
        action = actions[i]

        # Draw robot
        wheel_x = state[2]
        wheel_y = 0.033
        body_x = wheel_x + 0.025 * np.sin(state[0])
        body_y = wheel_y + 0.025 * np.cos(state[0])

        # Draw ground line
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)

        circle = plt.Circle((wheel_x, wheel_y), 0.033, fill=False, color="black")
        ax.add_patch(circle)
        ax.plot([wheel_x, body_x], [wheel_y, body_y], "b-", linewidth=3)

        plt.title(f"θ: {state[0]*180/np.pi:.1f}°, x: {state[2]:.2f}m\Torque: {action.item():.2f}")
        ax.set_xlim(x_center - view_width / 2, x_center + view_width / 2)
        ax.set_ylim(-0.1, 0.2)
        ax.grid(True)
        ax.set_aspect("equal")

    anim = animation.FuncAnimation(fig, animate, frames=len(states), interval=20)

    if save_path:
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(save_path, writer=writer)

    plt.close()
    return anim


def plot_predictions_comparison(physics_preds: np.ndarray, simnet_preds: np.ndarray, save_path: str = None):
    """Compare physics and SimNet predictions."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    titles = ["theta", "theta_dot", "x", "x_dot", "phi", "phi_dot"]

    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.scatter(physics_preds[:, i], simnet_preds[:, i], alpha=0.5)
        ax.plot(
            [min(physics_preds[:, i]), max(physics_preds[:, i])],
            [min(physics_preds[:, i]), max(physics_preds[:, i])],
            "r--",
            alpha=0.8,
        )
        ax.set_xlabel("Physics")
        ax.set_ylabel("SimNet")
        ax.set_title(title)
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

    return fig
