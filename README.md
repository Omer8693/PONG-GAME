# 🏓 Pong Game – Reinforcement Learning Agents

This project demonstrates how to train AI agents to play a simplified version of Pong using various reinforcement learning algorithms. The environment is custom-built using Pygame and integrates easily with Gym-style interfaces. Training metrics are logged via TensorBoard, and models can be compared using a built-in evaluation script.


## 🚀 Features

- 🎮 Custom Pygame-based Pong environment
- 🧠 RL algorithms implemented:
  - REINFORCE (Policy Gradient)
  - A2C (Advantage Actor-Critic)
  - Expected SARSA
- 📈 Training logs and metrics via TensorBoard
- 📊 Evaluation and comparison across models
- 🧪 Simple and Complex environment modes

## 📂 Project Structure

Pong Game/
├── Agent/
│ ├── agent_pg.py # REINFORCE policy network
│ ├── agent_expected_sarsa.py # Q-network for Expected SARSA
│ └── Paddle.py # Paddle (player) sprite logic
│
├── Env.py # Custom Gym-compatible Pong environment
├── Constants.py # Centralized settings & hyperparameters
│
├── train_pg.py # Train REINFORCE agent
├── train_a2c.py # Train A2C agent
├── train_expected_sarsa.py # Train Expected SARSA agent
├── evaluate.py # Evaluate and compare trained agents
│
├── models/ # Directory for saved model files
└── Pong_Log/ # TensorBoard logs

## ⚙️ Setup

Install required packages:

```bash
pip install pygame torch stable-baselines3 gymnasium matplotlib tensorboard

🏋️‍♂️ Training
Each algorithm has a dedicated training script. Run them as follows:
python train_pg.py               # REINFORCE
python train_a2c.py              # A2C
python train_expected_sarsa.py   # Expected SARSA

📊 Visualizing with TensorBoard
tensorboard --logdir Pong_Log/
Then visit: http://localhost:6006
You’ll see reward curves, loss, and entropy graphs.

 Evaluating Trained Models
python evaluate.py 200k simple
Arguments:
timesteps: same as model name (e.g. 50k, 200k)
mode: simple or complex
The script:
Loads available models
Runs 100 episodes per agent
Prints reward statistics
Saves a reward-vs-episode line plot
Exports results to CSV

⚙️ Hyperparameters (from Constants.py)
| Parameter         | Description                             | Default  |
| ----------------- | --------------------------------------- | -------- |
| `learning_rate`   | Optimizer step size                     | `0.0004` |
| `ent_coef`        | Entropy coefficient (for exploration)   | `0.01`   |
| `gamma`           | Discount factor for future rewards      | `0.95`   |
| `gae_lambda`      | GAE smoothing parameter                 | `0.95`   |
| `max_grad_norm`   | Clipping value for gradients            | `0.5`    |
| `total_timesteps` | Total training time steps per algorithm | `200000` |


🎮 Environment Modes
Simple: Standard paddle & ball behavior.
Complex: Paddle shrinks over time, ball speeds up with each bounce.

Reward system:
Hit the ball: +30
Stay aligned with the ball: +1
Get closer to the ball: +0.1
Miss the ball: -30

🙏 Acknowledgments
Built with:
PyTorch
Stable-Baselines3
Gymnasium
Pygame
