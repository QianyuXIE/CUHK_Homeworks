"""
Run DQN experiments.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from dqn.dqn import DQN, ReplayBuffer
from dqn.multi_step import MultiStepDQN, MultiStepReplayBuffer
from dqn.double_dqn import DoubleDQN
from utils.env_wrapper import LunarLanderWrapper

def run_dqn_experiments():
    """Run DQN experiments."""
    print("Running DQN experiments...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    env = LunarLanderWrapper(discrete=True)
    
    # Create networks
    q_net = QNetwork(env.obs_dim, env.act_dim)
    target_net = QNetwork(env.obs_dim, env.act_dim)
    buffer = ReplayBuffer(capacity=100000)
    
    # Train DQN
    print("\n1. Training DQN...")
    dqn = DQN(env, q_net, target_net, buffer)
    rewards_dqn, eval_rewards_dqn = dqn.train(total_steps=50000, eval_freq=2000, log_interval=1000)
    
    # Save results
    np.savez('dqn_results.npz',
             rewards=rewards_dqn,
             eval_rewards=eval_rewards_dqn)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(smooth(rewards_dqn), label='Training')
    plt.plot(np.arange(0, len(eval_rewards_dqn)*2000, 2000), eval_rewards_dqn, 
             'o-', label='Evaluation (every 2000 steps)')
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.title('DQN Training Curve')
    plt.legend()
    plt.grid(True)
    
    print("✓ DQN experiment completed!")

def run_multistep_experiments():
    """Run multi-step DQN experiments."""
    print("\nRunning multi-step DQN experiments...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    env = LunarLanderWrapper(discrete=True)
    
    n_steps_list = [1, 3, 5]
    all_results = {}
    
    for n_steps in n_steps_list:
        print(f"\nRunning multi-step DQN with N={n_steps}...")
        
        q_net = QNetwork(env.obs_dim, env.act_dim)
        target_net = QNetwork(env.obs_dim, env.act_dim)
        buffer = MultiStepReplayBuffer(capacity=100000, n_steps=n_steps)
        
        dqn = MultiStepDQN(env, q_net, target_net, buffer, n_steps=n_steps)
        rewards = dqn.train(total_steps=50000, log_interval=2000)
        
        all_results[f'n_{n_steps}'] = rewards
    
    # Plot results
    plt.figure(figsize=(12, 8))
    for n_steps in n_steps_list:
        plt.plot(smooth(all_results[f'n_{n_steps}']), label=f'N={n_steps}')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.title('Multi-step DQN: Effect of N')
    plt.legend()
    plt.grid(True)
    plt.savefig('multistep_comparison.png')
    plt.close()
    
    np.savez('multistep_results.npz', **all_results)
    
    print("✓ Multi-step experiments completed!")

def run_double_dqn_experiment():
    """Run Double DQN experiment."""
    print("\nRunning Double DQN experiment...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    env = LunarLanderWrapper(discrete=True)
    
    # Train DQN
    print("\n1. Training DQN...")
    q_net_dqn = QNetwork(env.obs_dim, env.act_dim)
    target_net_dqn = QNetwork(env.obs_dim, env.act_dim)
    buffer_dqn = ReplayBuffer(capacity=100000)
    dqn = DQN(env, q_net_dqn, target_net_dqn, buffer_dqn)
    rewards_dqn = dqn.train(total_steps=50000, log_interval=2000)
    
    # Train Double DQN
    print("\n2. Training Double DQN...")
    q_net_ddqn = QNetwork(env.obs_dim, env.act_dim)
    target_net_ddqn = QNetwork(env.obs_dim, env.act_dim)
    buffer_ddqn = ReplayBuffer(capacity=100000)
    ddqn = DoubleDQN(env, q_net_ddqn, target_net_ddqn, buffer_ddqn)
    rewards_ddqn = ddqn.train(total_steps=50000, log_interval=2000)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(smooth(rewards_dqn), label='DQN')
    plt.plot(smooth(rewards_ddqn), label='Double DQN')
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.title('DQN vs Double DQN')
    plt.legend()
    plt.grid(True)
    plt.savefig('ddqn_comparison.png')
    plt.close()
    
    np.savez('ddqn_results.npz',
             rewards_dqn=rewards_dqn,
             rewards_ddqn=rewards_ddqn)
    
    print("✓ Double DQN experiment completed!")

def smooth(x, window=20):
    """Smooth time series data."""
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode='valid')

if __name__ == '__main__':
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.chdir('results')
    
    # Run all experiments
    run_dqn_experiments()
    run_multistep_experiments()
    run_double_dqn_experiment()
    
    print("\n" + "="*50)
    print("All DQN experiments completed!")
    print("="*50)