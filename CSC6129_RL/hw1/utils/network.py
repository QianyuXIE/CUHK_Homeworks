"""
Neural network architectures for policy gradient and DQN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """
    Stochastic policy network (Gaussian for continuous, Categorical for discrete).
    """
    def __init__(self, obs_dim, act_dim, discrete=True, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.discrete = discrete
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if discrete:
            self.head = nn.Linear(hidden_dim, act_dim)
        else:
            self.mean = nn.Linear(hidden_dim, act_dim)
            self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, state):
        """
        Forward pass to get action distribution.
        
        Args:
            state: Tensor of shape (batch_size, obs_dim)
            
        Returns:
            distribution: torch.distributions object
        """
        x = self.net(state)
        
        if self.discrete:
            logits = self.head(x)
            return torch.distributions.Categorical(logits=logits)
        else:
            mean = self.mean(x)
            std = torch.exp(self.log_std)
            return torch.distributions.Normal(mean, std)

class ValueNetwork(nn.Module):
    """
    State-value function approximator.
    """
    def __init__(self, obs_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        Forward pass to get value estimate.
        
        Args:
            state: Tensor of shape (batch_size, obs_dim)
            
        Returns:
            value: Tensor of shape (batch_size, 1)
        """
        return self.net(state).squeeze(-1)

class QNetwork(nn.Module):
    """
    Q-value function approximator for DQN.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
    
    def forward(self, state):
        """
        Forward pass to get Q-values for all actions.
        
        Args:
            state: Tensor of shape (batch_size, obs_dim)
            
        Returns:
            q_values: Tensor of shape (batch_size, act_dim)
        """
        return self.net(state)