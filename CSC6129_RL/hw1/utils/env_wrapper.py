"""
Environment wrapper for LunarLander-v2.
"""
import gymnasium as gym
import numpy as np

class LunarLanderWrapper:
    """
    Wrapper for LunarLander-v2 environment.
    """
    def __init__(self, discrete=True):
        self.env = gym.make('LunarLander-v2')
        self.discrete = discrete
        
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
    
    def reset(self):
        """Reset environment."""
        state, _ = self.env.reset()
        return torch.FloatTensor(state)
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            next_state, reward, done, info
        """
        if self.discrete:
            action = action.item() if isinstance(action, torch.Tensor) else action
        else:
            action = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
        
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        return (
            torch.FloatTensor(next_state),
            reward,
            done,
            info
        )
    
    def render(self):
        """Render environment."""
        self.env.render()
    
    def close(self):
        """Close environment."""
        self.env.close()