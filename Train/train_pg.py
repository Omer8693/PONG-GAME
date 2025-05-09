import sys
import os

# Add parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from Agent.agent_pg import PolicyNetwork
import Constants
from Env import PongEnv

def train_reinforce(complex_mode=False):
    """
    Trains a policy using the REINFORCE algorithm with Generalized Advantage Estimation (GAE).
    """
    env = PongEnv(complex_mode=complex_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=Constants.learning_rate)

    # Set up logging directory and writer
    mode_str = "complex" if complex_mode else "simple"
    log_dir = os.path.join(Constants.tensorboard_log, f"{Constants.REINFORCE_sub_folder}_{mode_str}")
    writer = SummaryWriter(log_dir=log_dir)

    os.makedirs(Constants.Model_Save_Path, exist_ok=True)

    # Estimate number of episodes based on average steps
    avg_steps = 300
    num_episodes = Constants.total_timesteps // avg_steps
    episode_rewards = []

    print(f"Training REINFORCE for {num_episodes} episodes ({mode_str} mode)...")

    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            log_probs = []
            rewards = []
            values = []
            total_reward = 0
            done = False

            # Rollout the episode
            while not done:
                state_tensor = torch.from_numpy(np.array(state)).float()
                action, log_prob = policy.act(state_tensor)
                value = policy.value(state_tensor)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                total_reward += reward
                state = next_state

            # Compute discounted returns using GAE
            next_state_tensor = torch.from_numpy(np.array(next_state)).float()
            next_value = policy.value(next_state_tensor) if not done else 0
            returns = compute_gae(rewards, values, next_value, Constants.gamma, Constants.gae_lambda)

            # Normalize returns for stability
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Compute policy entropy for exploration encouragement
            entropy = torch.stack([
                -policy.dist(torch.from_numpy(np.array(s)).float()).entropy() for s in [state]
            ]).mean()

            # Compute REINFORCE loss
            loss = torch.stack([
                -lp * ret for lp, ret in zip(log_probs, returns)
            ]).sum() - Constants.ent_coef * entropy

            # Backpropagation step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging and statistics
            episode_rewards.append(total_reward)
            writer.add_scalar("rollout/ep_rew_mean", total_reward, episode)
            writer.add_scalar("train/loss", loss.item(), episode)
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward:.2f}, Loss: {loss.item():.4f}, Entropy: {entropy.item():.4f}")
                writer.add_scalar("train/entropy", entropy.item(), episode)

        # Save the trained policy
        save_path = os.path.join(Constants.Model_Save_Path, f"REINFORCE_{Constants.timesteps_str}_{mode_str}.pt")
        torch.save(policy.state_dict(), save_path)
        print(f"✅ Model saved at: {save_path}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        writer.close()
        env.close()

def compute_gae(rewards, values, next_value, gamma, gae_lambda):
    """
    Computes Generalized Advantage Estimation (GAE) returns.
    """
    returns = []
    gae = 0
    for r, v, nv in zip(reversed(rewards), reversed(values), reversed([next_value] + values[:-1])):
        delta = r + gamma * nv - v
        gae = delta + gamma * gae_lambda * gae
        returns.insert(0, gae + v)
    return torch.tensor(returns, dtype=torch.float32)

if __name__ == "__main__":
    # Train policy in both simple and complex environment settings
    train_reinforce(complex_mode=False) # Simple mode
    train_reinforce(complex_mode=True) # Complex mode
