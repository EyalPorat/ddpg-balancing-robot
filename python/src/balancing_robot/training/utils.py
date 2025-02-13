import torch
import numpy as np
from typing import Dict, Any, Union, Tuple
import yaml
from pathlib import Path
import json
from datetime import datetime

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def polyak_update(target: torch.nn.Module, 
                  source: torch.nn.Module, 
                  tau: float) -> None:
    """Soft update target network parameters.
    
    Args:
        target: Target network
        source: Source network
        tau: Interpolation parameter (0=no update, 1=hard copy)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def create_log_dir(base_path: str = "logs") -> Path:
    """Create logging directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(base_path) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

class TrainingLogger:
    """Utility for logging training metrics."""
    
    def __init__(self, log_dir: Union[str, Path]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = []
        
    def log(self, metrics: Dict[str, Any]) -> None:
        """Log metrics for current step."""
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics.append(metrics)
        
    def save(self) -> None:
        """Save metrics to file."""
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def get_latest(self, metric_name: str) -> float:
        """Get latest value for given metric."""
        if not self.metrics:
            return 0.0
        return self.metrics[-1].get(metric_name, 0.0)

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_model(model: torch.nn.Module, 
               path: Union[str, Path], 
               metadata: Dict[str, Any] = None) -> None:
    """Save model weights and metadata."""
    save_dict = {
        'state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    torch.save(save_dict, path)

def load_model(model: torch.nn.Module, 
               path: Union[str, Path]) -> Dict[str, Any]:
    """Load model weights and return metadata."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint.get('metadata', {})

def compute_gae(rewards: np.ndarray, 
                values: np.ndarray, 
                dones: np.ndarray, 
                gamma: float = 0.99, 
                lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
            
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        
    returns = advantages + values
    return advantages, returns
