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
    eval_rewards = []
    last_reward = 0
    for m in metrics:
        reward = m.get('eval_reward', last_reward)
        eval_rewards.append(reward)
        last_reward = reward
    ax.plot(episodes, eval_rewards)
    ax.set_title('Episode Rewards')
    ax.set_xlabel('Episode')
    ax.grid(True)

    # Q-values
    ax = axes[0, 1]
    q_values = []
    last_q_value = 0
    for m in metrics:
        q_value = m.get('q_value', last_q_value)
        q_values.append(q_value)
        last_q_value = q_value
    ax.plot(episodes, q_values)
    ax.set_title('Average Q-Values')
    ax.set_xlabel('Episode')
    ax.grid(True)

    # Losses
    ax = axes[1, 0]
    actor_losses = []
    critic_losses = []
    last_actor_value = 0
    last_critic_value = 0
    for m in metrics:
        actor_loss = m.get('actor_loss', last_actor_value)
        critic_loss = m.get('critic_loss', last_critic_value)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        last_actor_value = actor_loss
        last_critic_value = critic_loss
    ax.plot(episodes, actor_losses, label='Actor')
    ax.plot(episodes, critic_losses, label='Critic')
    ax.set_title('Losses')
    ax.set_xlabel('Episode')
    ax.legend()
    ax.grid(True)

    # Episode lengths
    ax = axes[1, 1]
    episodes_lengths = []
    last_episode_length = 0
    for m in metrics:
        episode_length = m.get('episode_length', last_episode_length)
        episodes_lengths.append(episode_length)
        last_episode_length = episode_length
    ax.plot(episodes, episodes_lengths)
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
