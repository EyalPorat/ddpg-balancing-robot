import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import seaborn as sns
from pathlib import Path

def plot_training_metrics(metrics: List[Dict], save_path: Path = None):
    """Plot training progress metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    episodes = range(len(metrics))
    
    # Rewards
    ax = axes[0, 0]
    ax.plot(episodes, [m['episode_reward'] for m in metrics])
    ax.set_title('Episode Rewards')
    ax.set_xlabel('Episode')
    ax.grid(True)

    # Q-values
    ax = axes[0, 1]
    ax.plot(episodes, [m['q_value'] for m in metrics])
    ax.set_title('Average Q-Values')
    ax.set_xlabel('Episode')
    ax.grid(True)

    # Losses
    ax = axes[1, 0]
    ax.plot(episodes, [m['actor_loss'] for m in metrics], label='Actor')
    ax.plot(episodes, [m['critic_loss'] for m in metrics], label='Critic')
    ax.set_title('Losses')
    ax.set_xlabel('Episode')
    ax.legend()
    ax.grid(True)

    # Episode lengths
    ax = axes[1, 1]
    ax.plot(episodes, [m['episode_length'] for m in metrics])
    ax.set_title('Episode Length')
    ax.set_xlabel('Episode')
    ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_state_distributions(states: np.ndarray, save_path: Path = None):
    """Plot distributions of state variables."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    titles = ['Angle', 'Angular Velocity', 'Position', 'Velocity', 'Wheel Angle', 'Wheel Velocity']
    
    for i, (ax, title) in enumerate(zip(axes.flat, titles)):
        sns.histplot(states[:, i], ax=ax)
        ax.set_title(title)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    
    return fig
