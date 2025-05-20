# Balancing Robot Python Framework

A comprehensive Python framework for training and deploying reinforcement learning models for self-balancing robots. Features DDPG implementation, neural network-based dynamics simulation, and real-world deployment tools.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start Examples

### Training SimNet - `train_simnet.ipynb`

### Training DDPG Model - `train_ddpg.ipynb`

### Deploying Model to Robot

```shell
python tools/model_deployer.py --model "C:\Users\eyalp\Downloads\actor_episode_1200 (3).pt"
```

## Package Structure

```
balancing_robot/
├── models/              # Neural network models
│   ├── actor.py        # DDPG actor network
│   ├── critic.py       # DDPG critic network
│   ├── simnet.py       # Dynamics simulation network
│   └── replay_buffer.py# Experience replay implementation
├── environment/         # Training environments
│   ├── balancer_env.py # Main training environment
│   ├── physics.py      # Physics simulation
├── training/           # Training algorithms
│   ├── ddpg_trainer.py # DDPG implementation
│   ├── simnet_trainer.py# SimNet training
│   └── utils.py        # Training utilities
└── visualization/      # Visualization tools
    ├── training_plots.py# Training metrics plots
    └── render.py       # Environment rendering
```

## Configuration

The framework uses YAML configuration files:

### DDPG Configuration
```yaml
model:
  actor:
    hidden_dims: [10, 10]
    learning_rate: 0.0001
  critic:
    hidden_dims: [256, 128, 64]
    learning_rate: 0.0003

training:
  total_episodes: 2000
  batch_size: 512
  gamma: 0.99
  tau: 0.005
```

### Environment Configuration
```yaml
physics:
  gravity: 9.81
  body_mass: 0.06
  wheel_mass: 0.04
  body_length: 0.025
  wheel_radius: 0.033

reward:
  angle_weight: 2.0
  position_weight: 5.0
```

## API Reference

### Models

#### Actor
```python
actor = Actor(
    state_dim=3,
    action_dim=1,
    max_action=1.0,
    hidden_dims=(10, 10)
)
```

#### SimNet
```python
simnet = SimNet(
    state_dim=6,
    action_dim=1,
    hidden_dims=(32, 32, 32)
)
```

### Environment

```python
env = BalancerEnv(
    config_path='configs/env_config.yaml',
    render_mode='human'
)
```

### Training

```python
trainer = DDPGTrainer(
    env=env,
    config_path='configs/ddpg_config.yaml'
)
```
