"""
Deep Q-Network (DQN) implementation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from utils.network import QNetwork
from utils.env_wrapper import LunarLanderWrapper

class ReplayBuffer:
    """
    Experience replay buffer.
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.FloatTensor(rewards),
            torch.stack(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQN:
    """
    Deep Q-Network implementation.
    """
    def __init__(self, env, q_net, target_net, buffer, 
                 lr=1e-3, gamma=0.99, batch_size=64,
                 target_update_freq=1000, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=10000, 
                 grad_clip=1.0):
        """
        Initialize DQN agent.
        
        Args:
            env: Environment wrapper
            q_net: Online Q-network
            target_net: Target Q-network
            buffer: Replay buffer
            lr: Learning rate
            gamma: Discount factor
            batch_size: Batch size
            target_update_freq: Target network update frequency
            epsilon_start: Initial epsilon
            epsilon_end: Final epsilon
            epsilon_decay: Epsilon decay steps
            grad_clip: Gradient clipping threshold
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
        self.grad_clip = grad_clip
        self.steps = 0
        
        # Copy weights to target network
        self.target_net.load_state_dict(q_net.state_dict())
    
    def epsilon(self):
        """Compute current epsilon."""
        if self.steps < self.epsilon_decay:
            return self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.steps / self.epsilon_decay
        else:
            return self.epsilon_end
    
    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
        """
        if random.random() < self.epsilon():
            return torch.randint(0, self.env.act_dim, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.q_net(state.unsqueeze(0))
                return q_values.argmax().item()
    
    def update(self):
        """
        Perform one DQN update.
        
        Returns:
            loss: TD loss
        """
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Compute Q-values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.HuberLoss()(q_values, targets)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        
        self.optimizer.step()
        
        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.steps += 1
        
        return loss.item()
    
    def train(self, total_steps=50000, eval_freq=1000, log_interval=100):
        """
        Train the DQN agent.
        
        Returns:
            rewards: List of episode rewards
        """
        rewards = []
        eval_rewards = []
        
        state = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(total_steps):
            # Select action
            action = self.select_action(state)
            
            # Take action
            next_state, reward, done, _ = self.env.step(action)
            
            # Store transition
            self.buffer.push(state, torch.tensor(action), reward, next_state, done)
            
            # Update
            loss = self.update()
            
            # Store reward
            episode_reward += reward
            episode_steps += 1
            
            # Reset if done
            if done:
                rewards.append(episode_reward)
                
                # Evaluation
                if step % eval_freq == 0:
                    eval_reward = self.evaluate(10)
                    eval_rewards.append(eval_reward)
                    print(f"Step {step}/{total_steps}, "
                          f"Episode Reward: {episode_reward:.2f}, "
                          f"Epsilon: {self.epsilon():.3f}, "
                          f"Eval Reward: {eval_reward:.2f}")
                
                # Reset environment
                state = self.env.reset()
                episode_reward = 0
                episode_steps = 0
            else:
                state = next_state
            
            # Logging
            if step % log_interval == 0 and step > 0:
                avg_reward = np.mean(rewards[-100:])
                print(f"Step {step}/{total_steps}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon():.3f}")
        
        return rewards, eval_rewards
    
    def evaluate(self, num_episodes=10, render=False):
        """
        Evaluate the agent.
        """
        returns = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    q_values = self.q_net(state.unsqueeze(0))
                    action = q_values.argmax().item()
                
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                if render:
                    self.env.render()
            
            returns.append(episode_reward)
        
        return np.mean(returns)