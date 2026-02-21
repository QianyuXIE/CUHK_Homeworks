"""
Generalized Advantage Estimation (GAE) implementation.
"""
import torch
import torch.nn as nn
import numpy as np
from utils.network import PolicyNetwork, ValueNetwork
from utils.env_wrapper import LunarLanderWrapper

class GAE:
    """
    Policy gradient with Generalized Advantage Estimation.
    """
    def __init__(self, env, policy_net, value_net, lr_policy=3e-4, 
                 lr_value=1e-3, gamma=0.99, lam=0.95, 
                 use_advantage_norm=True, discrete=True):
        """
        Initialize GAE agent.
        
        Args:
            env: Environment wrapper
            policy_net: Policy network
            value_net: Value network
            lr_policy: Policy learning rate
            lr_value: Value function learning rate
            gamma: Discount factor
            lam: GAE parameter (0 <= lam <= 1)
            use_advantage_norm: If True, normalize advantages
            discrete: If True, use discrete action space
        """
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=lr_policy)
        self.optimizer_value = torch.optim.Adam(value_net.parameters(), lr=lr_value)
        self.gamma = gamma
        self.lam = lam
        self.use_advantage_norm = use_advantage_norm
        self.discrete = discrete
    
    def compute_gae(self, rewards, values, next_value):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            next_value: Value estimate for next state
            
        Returns:
            advantages: Tensor of GAE advantages
        """
        T = len(rewards)
        advantages = torch.zeros(T)
        
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                delta = rewards[t] + self.gamma * next_value - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            
            gae = delta + self.gamma * self.lam * gae
            advantages[t] = gae
        
        return advantages
    
    def collect_trajectories(self, num_trajectories=10, max_steps=1000):
        """
        Collect multiple trajectories with GAE.
        
        Returns:
            all_states, all_actions, all_advantages
        """
        all_states = []
        all_actions = []
        all_advantages = []
        
        for _ in range(num_trajectories):
            states, actions, rewards, values, next_values = self.collect_trajectory(max_steps)
            
            # Convert to tensors
            states = torch.stack(states)
            if self.discrete:
                actions = torch.stack(actions)
            else:
                actions = torch.stack(actions)
            
            rewards = torch.FloatTensor(rewards)
            values = torch.stack(values)
            
            # Compute GAE advantages
            advantages = self.compute_gae(rewards, values, next_values[-1])
            
            all_states.append(states)
            all_actions.append(actions)
            all_advantages.append(advantages)
        
        return all_states, all_actions, all_advantages
    
    def collect_trajectory(self, max_steps=1000):
        """
        Collect one trajectory for GAE.
        """
        states = []
        actions = []
        rewards = []
        values = []
        next_values = []
        
        state = self.env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Get action
            dist = self.policy_net(state.unsqueeze(0))
            action = dist.sample()
            
            # Take action
            next_state, reward, done, _ = self.env.step(action)
            
            # Store
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Compute value estimates
            with torch.no_grad():
                value = self.value_net(state.unsqueeze(0))
                next_value = self.value_net(next_state.unsqueeze(0))
            
            values.append(value.squeeze())
            next_values.append(next_value.squeeze())
            
            state = next_state
            steps += 1
        
        return states, actions, rewards, values, next_values
    
    def update(self, states, actions, advantages):
        """
        Perform one update step.
        
        Returns:
            policy_loss
        """
        # Concatenate
        states = torch.cat(states)
        if self.discrete:
            actions = torch.cat(actions)
        else:
            actions = torch.cat(actions)
        
        advantages = torch.cat(advantages)
        
        # Normalize advantages
        if self.use_advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy
        self.optimizer_policy.zero_grad()
        dist = self.policy_net(states)
        log_probs = dist.log_prob(actions)
        policy_loss = -(log_probs * advantages).mean()
        policy_loss.backward()
        self.optimizer_policy.step()
        
        # Update value function
        self.optimizer_value.zero_grad()
        predicted_values = self.value_net(states)
        value_loss = nn.MSELoss()(predicted_values.squeeze(), advantages + 
                                self.value_net(states).detach().squeeze())
        value_loss.backward()
        self.optimizer_value.step()
        
        return policy_loss.item(), value_loss.item()
    
    def train(self, num_updates=500, trajectories_per_update=10, 
              max_steps=1000, log_interval=50):
        """
        Train the agent.
        """
        rewards = []
        
        for update in range(num_updates):
            # Collect trajectories
            states, actions, advantages = self.collect_trajectories(
                trajectories_per_update, max_steps
            )
            
            # Update
            policy_loss, value_loss = self.update(states, actions, advantages)
            
            # Evaluate
            _, _, eval_rewards = self.collect_trajectory(max_steps)[:3]
            episode_reward = sum(eval_rewards)
            rewards.append(episode_reward)
            
            # Logging
            if (update + 1) % log_interval == 0:
                avg_reward = np.mean(rewards[-log_interval:])
                print(f"Update {update + 1}/{num_updates}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Policy Loss: {policy_loss:.4f}")
        
        return rewards