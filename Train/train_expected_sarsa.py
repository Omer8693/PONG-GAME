import sys
import os

# Add parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

import Constants
from Env import PongEnv
from Agent.agent_expected_sarsa import QNetwork

def train_expected_sarsa(complex_mode=False):
    """
    Trains an Expected SARSA agent in the Pong environment.
    """
    env = PongEnv(complex_mode=complex_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Initialize Q-network and optimizer
    q_network = QNetwork(obs_dim, act_dim)
    optimizer = optim.Adam(q_network.parameters(), lr=Constants.learning_rate)

    # Epsilon-greedy exploration parameter
    epsilon = 0.1
    gamma = Constants.gamma

    # Setup logging
    mode_str = "complex" if complex_mode else "simple"
    log_dir = os.path.join(Constants.tensorboard_log, f"{Constants.ExpectedSARSA_sub_folder}_{mode_str}")
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs(Constants.Model_Save_Path, exist_ok=True)

    episode_rewards = []
    num_episodes = Constants.total_timesteps // 300  # Assumption: ~300 steps per episode

    print(f"Training Expected SARSA for {num_episodes} episodes ({mode_str} mode)...")

    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            step_count = 0
            max_steps = 1000  # Maximum allowed steps per episode

            while not done:
                state_tensor = torch.from_numpy(np.array(state)).float()

                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = q_network(state_tensor)
                        action = torch.argmax(q_values).item()

                # Execute the action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step_count += 1

                # Penalize if step limit is reached
                if step_count >= max_steps:
                    done = True
                    reward -= 30

                next_state_tensor = torch.from_numpy(np.array(next_state)).float()

                # Compute expected Q-value using policy
                with torch.no_grad():
                    next_q_values = q_network(next_state_tensor)
                    action_probs = torch.ones(act_dim) * (epsilon / act_dim)
                    best_action = torch.argmax(next_q_values).item()
                    action_probs[best_action] += (1.0 - epsilon)
                    expected_value = (next_q_values * action_probs).sum()

                # Compute TD target and loss
                q_values = q_network(state_tensor)
                target = reward + gamma * expected_value * (1 - int(done))
                loss = (target - q_values[action]) ** 2

                # Backpropagation step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Move to next state
                state = next_state

            # Log performance and loss
            episode_rewards.append(total_reward)
            writer.add_scalar("rollout/ep_rew_mean", total_reward, episode)
            writer.add_scalar("train/loss", loss.item(), episode)

            if episode % 10 == 0:
                print(f"Episode {episode}: reward = {total_reward:.2f}, loss = {loss.item():.4f}")

        # Save the trained model
        save_path = os.path.join(Constants.Model_Save_Path, f"ExpectedSARSA_{Constants.timesteps_str}_{mode_str}.pt")
        torch.save(q_network.state_dict(), save_path)
        print(f"✅ Model saved at: {save_path}")

    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        writer.close()
        env.close()

if __name__ == "__main__":
    # Train in both simple and complex modes
    train_expected_sarsa(complex_mode=False)
    train_expected_sarsa(complex_mode=True)
