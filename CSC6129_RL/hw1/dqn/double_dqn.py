"""
Double DQN implementation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from utils.network import QNetwork
from utils.env_wrapper import LunarLanderWrapper

class DoubleDQN:
    """
    Double DQN implementation.
    """
    def __init__(self, env, q_net, target_net, buffer, 
                 lr=1e-3, gamma=0.99, batch_size=64,
                 target_update_freq=1000, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=10000):
        """
        Initialize Double DQN.
        """
        self.env = env
        self.q_net = q_net
        self.target_net = target_net
        self.buffer = buffer
        self.optimizer = optim.Adam(q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0
        
        self.target_net.load_state_dict(q_net.state_dict())
    
    def epsilon(self):
        """Compute current epsilon."""
        if self.steps < self.epsilon_decay:
            return self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.steps / self.epsilon_decay
        else:
            return self.epsilon_end
    
    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon():
            return torch.randint(0, self.env.act_dim, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.q_net(state.unsqueeze(0))
                return q_values.argmax().item()
    
    def update(self):
        """Perform one Double DQN update."""
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Compute Q-values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN: select action with online network, evaluate with target network
        with torch.no_grad():
            # Select best action using online network
            online_actions = self.q_net(next_states).argmax(1)
            
            # Evaluate with target network
            next_q_values = self.target_net(next_states).gather(1, online_actions.unsqueeze(1)).squeeze()
            
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.HuberLoss()(q_values, targets)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.steps += 1
        
        return loss.item()
    
    def train(self, total_steps=50000, log_interval=100):
        """Train the agent."""
        rewards = []
        
        state = self.env.reset()
        episode_reward = 0
        
        for step in range(total_steps):
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            self.buffer.push(state, action, reward, next_state, done)
            
            loss = self.update()
            
            episode_reward += reward
            
            if done:
                rewards.append(episode_reward)
                episode_reward = 0
                state = self.env.reset()
            else:
                state = next_state
            
            if step % log_interval == 0 and step > 0:
                avg_reward = np.mean(rewards[-100:])
                print(f"Step {step}/{total_steps}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon():.3f}")
        
        return rewards