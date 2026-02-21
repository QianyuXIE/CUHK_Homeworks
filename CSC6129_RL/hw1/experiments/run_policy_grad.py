"""
Run policy gradient experiments.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from policy_gradients.reinforce import REINFORCE
from policy_gradients.baseline import PolicyGradientBaseline
from policy_gradients.gae import GAE
from utils.env_wrapper import LunarLanderWrapper

def run_reinforce_experiments():
    """Run REINFORCE experiments (trajectory return vs reward-to-go)."""
    print("Running REINFORCE experiments...")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create environment (discrete for simplicity)
    env = LunarLanderWrapper(discrete=True)
    
    # Create networks
    policy_net = PolicyNetwork(env.obs_dim, env.act_dim, discrete=True)
    
    # Experiment 1: Trajectory return
    print("\n1. Running REINFORCE with trajectory return...")
    reinforce_traj = REINFORCE(env, policy_net, use_reward_to_go=False, discrete=True)
    rewards_traj = reinforce_traj.train(num_episodes=500, log_interval=50)
    
    # Reset network for fair comparison
    policy_net = PolicyNetwork(env.obs_dim, env.act_dim, discrete=True)
    
    # Experiment 2: Reward-to-go
    print("\n2. Running REINFORCE with reward-to-go...")
    reinforce_rtg = REINFORCE(env, policy_net, use_reward_to_go=True, discrete=True)
    rewards_rtg = reinforce_rtg.train(num_episodes=500, log_interval=50)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Smooth rewards
    def smooth(x, window=20):
        return np.convolve(x, np.ones(window)/window, mode='valid')
    
    plt.subplot(2, 2, 1)
    plt.plot(smooth(rewards_traj), label='Trajectory Return')
    plt.plot(smooth(rewards_rtg), label='Reward-to-Go')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('REINFORCE: Trajectory Return vs Reward-to-Go')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('reinforce_comparison.png')
    plt.close()
    
    # Save results
    np.savez('reinforce_results.npz',
             rewards_traj=rewards_traj,
             rewards_rtg=rewards_rtg)
    
    print("✓ REINFORCE experiments completed!")

def run_baseline_experiments():
    """Run baseline experiments."""
    print("\nRunning baseline experiments...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    env = LunarLanderWrapper(discrete=True)
    
    # Experiment 1: No baseline
    print("\n1. Running without baseline...")
    policy_net = PolicyNetwork(env.obs_dim, env.act_dim, discrete=True)
    value_net = ValueNetwork(env.obs_dim)
    pg_no_baseline = PolicyGradientBaseline(
        env, policy_net, value_net, use_advantage_norm=False
    )
    rewards_no_baseline = pg_no_baseline.train(num_updates=300, log_interval=30)
    
    # Experiment 2: With baseline, no normalization
    print("\n2. Running with baseline (no normalization)...")
    policy_net = PolicyNetwork(env.obs_dim, env.act_dim, discrete=True)
    value_net = ValueNetwork(env.obs_dim)
    pg_baseline_no_norm = PolicyGradientBaseline(
        env, policy_net, value_net, use_advantage_norm=False
    )
    rewards_baseline_no_norm = pg_baseline_no_norm.train(num_updates=300, log_interval=30)
    
    # Experiment 3: With baseline and normalization
    print("\n3. Running with baseline (with normalization)...")
    policy_net = PolicyNetwork(env.obs_dim, env.act_dim, discrete=True)
    value_net = ValueNetwork(env.obs_dim)
    pg_baseline_norm = PolicyGradientBaseline(
        env, policy_net, value_net, use_advantage_norm=True
    )
    rewards_baseline_norm = pg_baseline_norm.train(num_updates=300, log_interval=30)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(smooth(rewards_no_baseline), label='No Baseline')
    plt.plot(smooth(rewards_baseline_no_norm), label='With Baseline (No Norm)')
    plt.plot(smooth(rewards_baseline_norm), label='With Baseline (With Norm)')
    plt.xlabel('Update')
    plt.ylabel('Reward')
    plt.title('Baseline Effectiveness')
    plt.legend()
    plt.grid(True)
    plt.savefig('baseline_comparison.png')
    plt.close()
    
    np.savez('baseline_results.npz',
             rewards_no_baseline=rewards_no_baseline,
             rewards_baseline_no_norm=rewards_baseline_no_norm,
             rewards_baseline_norm=rewards_baseline_norm)
    
    print("✓ Baseline experiments completed!")

def run_gae_experiments():
    """Run GAE experiments with different lambda values."""
    print("\nRunning GAE experiments...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    env = LunarLanderWrapper(discrete=True)
    
    lambdas = [0.0, 0.95, 1.0]
    all_results = {}
    
    for lam in lambdas:
        print(f"\nRunning GAE with λ={lam}...")
        policy_net = PolicyNetwork(env.obs_dim, env.act_dim, discrete=True)
        value_net = ValueNetwork(env.obs_dim)
        
        gae = GAE(env, policy_net, value_net, lam=lam, discrete=True)
        rewards = gae.train(num_updates=300, log_interval=30)
        
        all_results[f'lambda_{lam}'] = rewards
    
    # Plot results
    plt.figure(figsize=(12, 8))
    for lam in lambdas:
        plt.plot(smooth(all_results[f'lambda_{lam}']), label=f'λ={lam}')
    
    plt.xlabel('Update')
    plt.ylabel('Reward')
    plt.title('GAE: Effect of λ parameter')
    plt.legend()
    plt.grid(True)
    plt.savefig('gae_comparison.png')
    plt.close()
    
    np.savez('gae_results.npz', **all_results)
    
    print("✓ GAE experiments completed!")

if __name__ == '__main__':
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.chdir('results')
    
    # Run all experiments
    run_reinforce_experiments()
    run_baseline_experiments()
    run_gae_experiments()
    
    print("\n" + "="*50)
    print("All policy gradient experiments completed!")
    print("="*50)