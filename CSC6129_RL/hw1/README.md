# Reinforcement Learning Homework

## Overview
This repository contains implementations of core reinforcement learning algorithms: Policy Gradients (REINFORCE, Baselines, GAE) and Deep Q-Networks (DQN, Multi-step, Double DQN) on the LunarLander-v2 environment.

## Installation

```bash
pip install gymnasium[box2d] torch matplotlib numpy
```

## Running Experiments

### Policy Gradient Experiments

```bash
python experiments/run_policy_grad.py
```

This will run:
1. REINFORCE with trajectory return vs reward-to-go
2. Policy gradient with/without baseline and advantage normalization
3. GAE with different λ values (0, 0.95, 1)

### DQN Experiments

```bash
python experiments/run_dqn.py
```

This will run:
1. DQN training and evaluation
2. Multi-step DQN with N=1, 3, 5
3. DQN vs Double DQN comparison

## File Structure
```
homework/
├── policy_gradients/
│   ├── reinforce.py          # REINFORCE implementation
│   ├── baseline.py           # Policy gradient with baseline
│   └── gae.py                # GAE implementation
├── dqn/
│   ├── dqn.py                # DQN implementation
│   ├── multi_step.py         # Multi-step DQN
│   └── double_dqn.py         # Double DQN
├── utils/
│   ├── network.py            # Neural network architectures
│   └── env_wrapper.py        # Environment wrapper
├── experiments/
│   ├── run_policy_grad.py    # Policy gradient experiments
│   └── run_dqn.py            # DQN experiments
└── README.md                 # This file
```