"""
REINFORCE algorithm implementation.
"""
import torch
import numpy as np
from utils.network import PolicyNetwork
from utils.env_wrapper import LunarLanderWrapper

class REINFORCE:
    """
    REINFORCE algorithm with two return estimators.
    """
    def __init__(self, env, policy_net, lr=3e-4, gamma=0.99, 
                 use_reward_to_go=True, discrete=True):
        """
        Initialize REINFORCE agent.
        
        Args:
            env: Environment wrapper
            policy_net: Policy network
            lr: Learning rate
            gamma: Discount factor
            use_reward_to_go: If True, use reward-to-go; else use trajectory return
            discrete: If True, use discrete action space
        """
        self.env = env
        self.policy_net = policy_net
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.discrete = discrete
    
    def compute_returns(self, rewards):
        """
        Compute returns for a trajectory.
        
        Args:
            rewards: List of rewards for one trajectory
            
        Returns:
            returns: Tensor of returns for each time step
        """
        T = len(rewards)
        returns = torch.zeros(T)
        
        if self.use_reward_to_go:
            # Reward-to-go: G_t = sum_{t'=t}^{T-1} gamma^{t'-t} r_t'
            G = 0
            for t in reversed(range(T)):
                G = rewards[t] + self.gamma * G
                returns[t] = G
        else:
            # Trajectory return: G = sum_{t'=0}^{T-1} gamma^{t'} r_t'
            G = sum(self.gamma**t * r for t, r in enumerate(rewards))
            returns[:] = G
        
        return returns
    
    def collect_trajectory(self, max_steps=1000):
        """
        Collect one trajectory.
        
        Returns:
            states, actions, rewards: Lists of collected data
        """
        states = []
        actions = []
        rewards = []
        
        state = self.env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Get action from policy
            dist = self.policy_net(state.unsqueeze(0))
            action = dist.sample()
            
            # Take action
            next_state, reward, done, _ = self.env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            steps += 1
        
        return states, actions, rewards
    
    def update(self, states, actions, returns):
        """
        Perform one policy gradient update.
        
        Args:
            states: List of states
            actions: List of actions
            returns: Tensor of returns for each time step
        """
        self.optimizer.zero_grad()
        
        # Convert to tensors
        states = torch.stack(states)
        if self.discrete:
            actions = torch.stack(actions)
        else:
            actions = torch.stack(actions)
        
        # Compute log probabilities
        dist = self.policy_net(states)
        log_probs = dist.log_prob(actions)
        
        # Compute loss
        loss = -(log_probs * returns).mean()
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes=1000, log_interval=100):
        """
        Train the agent.
        
        Args:
            num_episodes: Number of episodes to train
            log_interval: Interval for logging
            
        Returns:
            rewards: List of episode rewards
        """
        rewards = []
        
        for episode in range(num_episodes):
            # Collect trajectory
            states, actions, rewards_list = self.collect_trajectory()
            
            # Compute returns
            returns = self.compute_returns(rewards_list)
            returns = returns.to(self.policy_net.net[0].weight.device)
            
            # Update policy
            loss = self.update(states, actions, returns)
            
            # Store episode reward
            episode_reward = sum(rewards_list)
            rewards.append(episode_reward)
            
            # Logging
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(rewards[-log_interval:])
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")
        
        return rewards