import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Tuple

def create_episode_animation(states: np.ndarray, save_path: str = None, fps: int = 30):
    """Create animation of robot balancing episode."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame):
        ax.clear()
        state = states[frame]
        
        # Robot parameters
        wheel_radius = 0.033
        body_length = 0.05
        
        # Extract position and angle
        x, theta = state[2], state[0]
        
        # Ground
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Wheel
        wheel = plt.Circle((x, wheel_radius), wheel_radius, fill=False, color='black')
        ax.add_patch(wheel)
        
        # Body
        body_x = x + body_length * np.sin(theta)
        body_y = wheel_radius + body_length * np.cos(theta)
        ax.plot([x, body_x], [wheel_radius, body_y], 'b-', linewidth=3)
        
        # Configure view
        ax.set_xlim(x - 0.5, x + 0.5)
        ax.set_ylim(-0.1, 0.3)
        ax.set_aspect('equal')
        ax.set_title(f'θ: {theta * 180/np.pi:.1f}°, x: {x:.2f}m')
        
    anim = animation.FuncAnimation(
        fig, update, frames=len(states), 
        interval=1000/fps, blit=False
    )
    
    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=fps)
        
    plt.close()
    return anim

def plot_predictions_comparison(physics_preds: np.ndarray, 
                              simnet_preds: np.ndarray, 
                              save_path: str = None):
    """Compare physics and SimNet predictions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['θ̈', 'ẍ', 'φ̈']
    
    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.scatter(physics_preds[:, i], simnet_preds[:, i], alpha=0.5)
        ax.plot([min(physics_preds[:, i]), max(physics_preds[:, i])],
                [min(physics_preds[:, i]), max(physics_preds[:, i])],
                'r--', alpha=0.8)
        ax.set_xlabel('Physics')
        ax.set_ylabel('SimNet')
        ax.set_title(title)
        ax.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        
    return fig
