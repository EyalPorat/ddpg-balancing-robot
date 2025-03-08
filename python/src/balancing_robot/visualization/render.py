import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def create_episode_animation(states: np.ndarray, actions: np.ndarray, save_path: str = None, fps: int = 30):
    """
    Create an animation of the trained agent balancing.

    Args:
        states: Array of states (theta, theta_dot)
        actions: Array of actions (torque)
        save_path: Path to save the animation
        fps: Frames per second

    Returns:
        Animation object
    """
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    def animate(i):
        ax.clear()
        state = states[i]
        action = actions[i]

        # Fixed wheel position for visualization
        wheel_x = 0
        wheel_y = 0.033  # wheel radius

        # Draw robot
        body_x = wheel_x + 0.025 * np.sin(state[0])
        body_y = wheel_y + 0.025 * np.cos(state[0])

        # Draw ground line
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)

        circle = plt.Circle((wheel_x, wheel_y), 0.033, fill=False, color="black")
        ax.add_patch(circle)
        ax.plot([wheel_x, body_x], [wheel_y, body_y], "b-", linewidth=3)

        # Add angle info and torque
        plt.title(f"θ: {state[0]*180/np.pi:.1f}°, θ̇: {state[1]:.2f} rad/s\nTorque: {action.item():.2f}")

        # Fixed view
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.1, 0.2)
        ax.grid(True)
        ax.set_aspect("equal")

    anim = animation.FuncAnimation(fig, animate, frames=len(states), interval=20)

    if save_path:
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(save_path, writer=writer)

    plt.close()
    return anim


def plot_predictions_comparison(
    physics_preds: np.ndarray, simnet_preds: np.ndarray, simnet_physics_preds: np.ndarray, save_path: str = None
):
    """Compare physics and SimNet predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    titles = ["theta", "theta_dot"]

    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.scatter(physics_preds[:, i], simnet_preds[:, i], alpha=0.5, label="Fine-Tuned SimNet", color="orange")
        ax.scatter(physics_preds[:, i], simnet_physics_preds[:, i], alpha=0.2, label="SimNet Physics", color="lightblue")
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
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

    return fig
