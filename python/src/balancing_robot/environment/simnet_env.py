import numpy as np
from typing import Optional, Dict, Any, Tuple

from .balancer_env import BalancerEnv
from .physics import PhysicsParams
from ..models.simnet import SimNet

class SimNetBalancerEnv(BalancerEnv):
    """Balancing robot environment that uses SimNet for dynamics prediction."""
    
    def __init__(
        self,
        simnet: SimNet,
        physics_params: Optional[PhysicsParams] = None,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
        hybrid_ratio: float = 1.0,
        adaptation_steps: int = 0
    ):
        """Initialize SimNet environment.
        
        Args:
            simnet: Trained SimNet model for dynamics prediction
            physics_params: Custom physics parameters (for hybrid simulation)
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ('human' or 'rgb_array')
            hybrid_ratio: Mix ratio between SimNet and physics (0=pure physics, 1=pure SimNet)
            adaptation_steps: Number of steps to gradually increase hybrid_ratio from 0 to target
        """
        super().__init__(physics_params, None, max_steps, render_mode)
        
        self.simnet = simnet
        self.target_hybrid_ratio = hybrid_ratio
        self.current_hybrid_ratio = 0.0
        self.adaptation_steps = adaptation_steps
        self.total_steps = 0
        
        # Enable tracking of prediction errors
        self.tracking_stats = {
            'prediction_errors': [],
            'physics_predictions': [],
            'simnet_predictions': []
        }
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step with SimNet predictions.
        
        Args:
            action: Action to take (scaled motor torque)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Update hybrid ratio if in adaptation phase
        if self.adaptation_steps > 0:
            self.current_hybrid_ratio = min(
                self.target_hybrid_ratio,
                self.total_steps / self.adaptation_steps * self.target_hybrid_ratio
            )
        else:
            self.current_hybrid_ratio = self.target_hybrid_ratio
        
        # Scale action to actual torque
        torque = np.clip(action, -1.0, 1.0) * self.max_torque
        
        # Get predictions from both physics and SimNet
        physics_accels = self.physics.get_acceleration(self.state, torque)
        simnet_accels = self.simnet.get_accelerations(self.state, action)
        
        # Compute hybrid accelerations
        accelerations = (
            (1 - self.current_hybrid_ratio) * physics_accels +
            self.current_hybrid_ratio * simnet_accels
        )
        
        # Track prediction differences
        if self.current_hybrid_ratio > 0:
            self.tracking_stats['prediction_errors'].append(
                np.mean((physics_accels - simnet_accels) ** 2)
            )
            self.tracking_stats['physics_predictions'].append(physics_accels)
            self.tracking_stats['simnet_predictions'].append(simnet_accels)
        
        # Update state
        self.state = self.physics.integrate_state(self.state, accelerations)
        self.steps += 1
        self.total_steps += 1
        
        # Calculate rewards and check termination
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = self.steps >= self.max_steps
        
        # Additional info including prediction metrics
        info = {
            'state_of_interest': {
                'angle': self.state[0],
                'position': self.state[2],
                'energy': self.physics.get_energy(self.state)
            },
            'simulation': {
                'hybrid_ratio': self.current_hybrid_ratio,
                'prediction_error': self.tracking_stats['prediction_errors'][-1] if self.tracking_stats['prediction_errors'] else 0.0,
                'physics_pred': physics_accels,
                'simnet_pred': simnet_accels
            }
        }
        
        if self.render_mode == 'human':
            self._render_frame()
            
        return self.state, reward, terminated, truncated, info
    
    def reset(self, 
              seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and optionally reset adaptation phase.
        
        Args:
            seed: Random seed
            options: Can include 'reset_adaptation': bool to restart hybrid ratio adaptation
            
        Returns:
            Tuple of (observation, info)
        """
        observation, info = super().reset(seed=seed)
        
        # Check if we should reset adaptation
        if options and options.get('reset_adaptation', False):
            self.current_hybrid_ratio = 0.0
            self.total_steps = 0
            self.tracking_stats = {
                'prediction_errors': [],
                'physics_predictions': [],
                'simnet_predictions': []
            }
            
        return observation, info
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get statistics about prediction accuracy.
        
        Returns:
            Dictionary containing prediction error statistics
        """
        if not self.tracking_stats['prediction_errors']:
            return {}
            
        errors = np.array(self.tracking_stats['prediction_errors'])
        physics_preds = np.array(self.tracking_stats['physics_predictions'])
        simnet_preds = np.array(self.tracking_stats['simnet_predictions'])
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'correlation': np.corrcoef(
                physics_preds.flatten(),
                simnet_preds.flatten()
            )[0, 1],
            'total_steps': self.total_steps
        }
    
    def _render_frame(self):
        """Render one frame with additional SimNet information."""
        # Call parent rendering
        super()._render_frame()
        
        if self.fig is not None and hasattr(self, 'ax'):
            # Add SimNet specific information to title
            current_title = self.ax.get_title()
            self.ax.set_title(
                f'{current_title}\n'
                f'SimNet Ratio: {self.current_hybrid_ratio:.2f}'
            )
            
            if self.tracking_stats['prediction_errors']:
                error = self.tracking_stats['prediction_errors'][-1]
                self.ax.text(
                    0.02, 0.98,
                    f'Pred Error: {error:.2e}',
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
