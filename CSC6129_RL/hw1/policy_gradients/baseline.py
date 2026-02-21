"""
Policy gradient with learned baseline and advantage normalization.
"""
import torch
import torch.nn as nn
import numpy as np
from utils.network import PolicyNetwork, ValueNetwork
from utils.env_wrapper import LunarLanderWrapper

class PolicyGradientBaseline:
    """
    Policy gradient with learned value function baseline.
    """
    def __init__(self, env, policy_net, value_net, lr_policy=3e-4, 
                 lr_value=1e-3, gamma=0.99, use_advantage_norm=True,
                 discrete=True):
        """
        Initialize policy gradient with baseline.
        
        Args:
            env: Environment wrapper
            policy_net: Policy network
            value_net: Value network
            lr_policy: Policy learning rate
            lr_value: Value function learning rate
            gamma: Discount factor
            use_advantage_norm: If True, normalize advantages
            discrete: If True, use discrete action space
        """
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=lr_policy)
        self.optimizer_value = torch.optim.Adam(value_net.parameters(), lr=lr_value)
        self.gamma = gamma
        self.use_advantage_norm = use_advantage_norm
        self.discrete = discrete
    
    def compute_returns(self, rewards):
        """
        Compute discounted returns.
        
        Args:
            rewards: List of rewards
            
        Returns:
            returns: Tensor of returns
        """
        T = len(rewards)
        returns = torch.zeros(T)
        
        G = 0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        
        return returns
    
    def collect_trajectories(self, num_trajectories=10, max_steps=1000):
        """
        Collect multiple trajectories.
        
        Returns:
            all_states, all_actions, all_returns, all_values
        """
        all_states = []
        all_actions = []
        all_returns = []
        all_values = []
        
        for _ in range(num_trajectories):
            states, actions, rewards = self.collect_trajectory(max_steps)
            returns = self.compute_returns(rewards)
            
            # Convert to tensors
            states = torch.stack(states)
            if self.discrete:
                actions = torch.stack(actions)
            else:
                actions = torch.stack(actions)
            
            # Compute value estimates
            with torch.no_grad():
                values = self.value_net(states).detach()
            
            all_states.append(states)
            all_actions.append(actions)
            all_returns.append(returns)
            all_values.append(values)
        
        return all_states, all_actions, all_returns, all_values
    
    def collect_trajectory(self, max_steps=1000):
        """
        Collect one trajectory.
        """
        states = []
        actions = []
        rewards = []
        
        state = self.env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            dist = self.policy_net(state.unsqueeze(0))
            action = dist.sample()
            
            next_state, reward, done, _ = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            steps += 1
        
        return states, actions, rewards
    
    def update(self, states, actions, returns, values):
        """
        Perform one update step.
        
        Args:
            states: List of state tensors
            actions: List of action tensors
            returns: List of return tensors
            values: List of value tensors
            
        Returns:
            policy_loss, value_loss
        """
        # Concatenate all trajectories
        states = torch.cat(states)
        if self.discrete:
            actions = torch.cat(actions)
        else:
            actions = torch.cat(actions)
        
        returns = torch.cat(returns)
        values = torch.cat(values)
        
        # Compute advantages
        advantages = returns - values
        
        # Normalize advantages if requested
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
        value_loss = nn.MSELoss()(predicted_values.squeeze(), returns)
        value_loss.backward()
        self.optimizer_value.step()
        
        return policy_loss.item(), value_loss.item()
    
    def train(self, num_updates=500, trajectories_per_update=10, 
              max_steps=1000, log_interval=50):
        """
        Train the agent.
        
        Returns:
            rewards: List of episode rewards
        """
        rewards = []
        
        for update in range(num_updates):
            # Collect trajectories
            states, actions, returns, values = self.collect_trajectories(
                trajectories_per_update, max_steps
            )
            
            # Update networks
            policy_loss, value_loss = self.update(states, actions, returns, values)
            
            # Evaluate on one trajectory
            _, _, eval_rewards = self.collect_trajectory(max_steps)
            episode_reward = sum(eval_rewards)
            rewards.append(episode_reward)
            
            # Logging
            if (update + 1) % log_interval == 0:
                avg_reward = np.mean(rewards[-log_interval:])
                print(f"Update {update + 1}/{num_updates}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Policy Loss: {policy_loss:.4f}, "
                      f"Value Loss: {value_loss:.4f}")
        
        return rewards