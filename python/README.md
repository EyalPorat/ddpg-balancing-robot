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

### Training DDPG Model

```python
from balancing_robot.environment import BalancerEnv
from balancing_robot.training import DDPGTrainer
import yaml

# Load configuration
with open('configs/ddpg_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create environment and trainer
env = BalancerEnv(config_path='configs/env_config.yaml')
trainer = DDPGTrainer(env, config_path='configs/ddpg_config.yaml')

# Train model
history = trainer.train(num_episodes=2000)
```

### Training SimNet

```python
from balancing_robot.models import SimNet
from balancing_robot.training import SimNetTrainer

# Initialize trainer
trainer = SimNetTrainer(config_path='configs/simnet_config.yaml')

# Train on physics data first
train_data, val_data = trainer.collect_physics_data()
trainer.train(train_data, val_data)

# Then fine-tune on real data
trainer.train(real_train_data, real_val_data, is_finetuning=True)
```

### Deploying Model to Robot

```python
from balancing_robot.tools import ModelDeployer

deployer = ModelDeployer()
deployer.deploy_model('checkpoints/best_model.pt', ip='192.168.1.255')
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
│   └── simnet_env.py   # SimNet-based environment
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
    hidden_dims: [8, 8]
    learning_rate: 0.0001
  critic:
    hidden_dims: [256, 256]
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
    state_dim=6,
    action_dim=1,
    max_action=1.0,
    hidden_dims=(8, 8)
)
```

#### SimNet
```python
simnet = SimNet(
    state_dim=6,
    action_dim=1,
    hidden_dims=(128, 128)
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
