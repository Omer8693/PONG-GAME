# Pong Game – Reinforcement Learning Agents

This project showcases how to train AI agents to play a custom Pong game using various reinforcement learning (RL) algorithms. The environment is built with Pygame and designed to be compatible with Gym-style interfaces. Training metrics are logged using TensorBoard, and a built-in evaluation script enables side-by-side model comparison.

## Features

- Custom Pygame-based Pong environment
- Implemented RL algorithms:
  - REINFORCE (Policy Gradient)
  - A2C (Advantage Actor-Critic)
  - Expected SARSA (Value-based)
- Live training metrics via TensorBoard
- Evaluation & comparison of trained agents
- Simple and Complex environment modes

## Project Structure

```
Pong_Game/
├── Agent/
│   ├── agent_pg.py               # REINFORCE policy network
│   ├── agent_expected_sarsa.py  # Q-network for Expected SARSA
│   └── Paddle.py                # Paddle sprite logic
│
├── Env.py                       # Custom Gym-compatible Pong environment
├── Constants.py                 # Centralized settings & hyperparameters
│
├── train_pg.py                  # Train REINFORCE agent
├── train_a2c.py                 # Train A2C agent
├── train_expected_sarsa.py      # Train Expected SARSA agent
├── evaluate.py                  # Evaluate and compare trained agents
│
├── models/                      # Saved model files
└── Pong_Log/                    # TensorBoard logs
```

## Setup

Install required dependencies:

```bash
pip install pygame torch stable-baselines3 gymnasium matplotlib tensorboard
```

## Training

Each algorithm has a dedicated training script:

```bash
python train_pg.py               # REINFORCE
python train_a2c.py              # A2C
python train_expected_sarsa.py   # Expected SARSA
```

## Visualize with TensorBoard

Start TensorBoard:

```bash
tensorboard --logdir Pong_Log/
```

Then open your browser at: http://localhost:6006  
You'll find reward curves, loss values, and entropy graphs for each algorithm.

## Evaluating Trained Models

```bash
python evaluate.py 200k simple
```

Arguments:

- `timesteps`: Model size (e.g., `50k`, `200k`)
- `mode`: Either `simple` or `complex`

What it does:

- Loads available models
- Runs 100 episodes per agent
- Prints average rewards
- Plots reward vs. episode
- Saves results as CSV

## Hyperparameters

(from `Constants.py`)

| Parameter         | Description                             | Default   |
|------------------|-----------------------------------------|-----------|
| `learning_rate`   | Optimizer step size                     | `0.0004`  |
| `ent_coef`        | Entropy coefficient (exploration)       | `0.01`    |
| `gamma`           | Discount factor                         | `0.95`    |
| `gae_lambda`      | GAE smoothing parameter                 | `0.95`    |
| `max_grad_norm`   | Gradient clipping threshold             | `0.5`     |
| `total_timesteps` | Total training timesteps per algorithm  | `200000`  |

## Environment Modes

- Simple: Default paddle and ball speed.
- Complex: Ball speed increases by 10% per bounce, paddle shrinks every 100 timesteps.

### Reward System

| Event               | Reward  |
|---------------------|---------|
| Hit the ball        | +30     |
| Stay aligned        | +1      |
| Move closer to ball | +0.1    |
| Miss the ball       | -30     |

## Acknowledgments

Built using:

- PyTorch
- Stable-Baselines3
- Gymnasium
- Pygame
