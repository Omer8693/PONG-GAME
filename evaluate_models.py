import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from Env import PongEnv
from Agent.agent_pg import PolicyNetwork
from Agent.agent_expected_sarsa import QNetwork
from stable_baselines3 import A2C

# Komut satırından argümanlar: timesteps, mode
timesteps_str = sys.argv[1] if len(sys.argv) > 1 else "50k"
mode_str = sys.argv[2] if len(sys.argv) > 2 else "simple"
is_complex = mode_str.lower() == "complex"

model_dir = "./models"
ALGORITHMS = {
    "REINFORCE": "pg",
    "ExpectedSARSA": "sarsa",
    "A2C": "a2c"
}

NUM_EPISODES = 100

def resolve_model_path_and_type(algo_name):
    for ext in [".pt", ".zip"]:
        filename = f"{algo_name}_{timesteps_str}_{mode_str}{ext}"
        full_path = os.path.join(model_dir, filename)
        if os.path.exists(full_path):
            return full_path, ALGORITHMS[algo_name]
    return None, None

def evaluate_policy(model_name, model_type, model_path):
    env = PongEnv(complex_mode=is_complex)
    total_rewards = []

    if model_type == "pg":
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        model = PolicyNetwork(obs_dim, act_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    elif model_type == "sarsa":
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        model = QNetwork(obs_dim, act_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    elif model_type == "a2c":
        model = A2C.load(model_path)
    else:
        raise ValueError("Unknown model type!")

    for _ in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            if model_type == "pg":
                obs_tensor = torch.from_numpy(np.array(obs)).float()
                action, _ = model.get_action(obs_tensor)
            elif model_type == "sarsa":
                obs_tensor = torch.from_numpy(np.array(obs)).float()
                q_values = model(obs_tensor)
                action = torch.argmax(q_values).item()
            elif model_type == "a2c":
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        total_rewards.append(total_reward)

    env.close()
    return (
        np.mean(total_rewards),
        np.std(total_rewards),
        np.min(total_rewards),
        np.max(total_rewards),
        total_rewards
    )

def plot_bar_and_box(results, results_raw):
    algos = list(results.keys())
    means = [results[a]["mean"] for a in algos]
    stds = [results[a]["std"] for a in algos]
    data = [results_raw[a] for a in algos]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    axs[0].bar(algos, means, yerr=stds, capsize=10, color="skyblue")
    axs[0].set_title("Average Reward")
    axs[0].set_ylabel("Reward")
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Boxplot
    axs[1].boxplot(data, labels=algos, patch_artist=True,
                   boxprops=dict(facecolor="lightgreen"),
                   medianprops=dict(color="darkgreen"))
    axs[1].set_title("Reward Distribution")
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle(f"Model Evaluation – {timesteps_str}, {mode_str.capitalize()} Mode", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"eval_plot_{timesteps_str}_{mode_str}.png")
    plt.show()

def plot_lineplot(results_raw):
    plt.figure(figsize=(10, 5))
    for algo, rewards in results_raw.items():
        plt.plot(rewards, label=algo, linewidth=2)

    plt.title(f"Episode Reward per Model – {timesteps_str}, {mode_str.capitalize()} Mode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"lineplot_{timesteps_str}_{mode_str}.png")
    plt.show()

def run_evaluation():
    print(f"📊 Evaluating models for: {timesteps_str} - {mode_str} mode")
    print(f"{'Model':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 52)

    results = {}
    results_raw = {}
    table_data = []

    for algo in ALGORITHMS.keys():
        model_path, model_type = resolve_model_path_and_type(algo)
        if model_path:
            mean, std, min_r, max_r, raw = evaluate_policy(algo, model_type, model_path)
            results[algo] = {"mean": mean, "std": std}
            results_raw[algo] = raw
            print(f"{algo:<20} {mean:>8.2f} {std:>8.2f} {min_r:>8.2f} {max_r:>8.2f}")
            table_data.append({
                "Model": algo,
                "Mean": round(mean, 2),
                "Std": round(std, 2),
                "Min": round(min_r, 2),
                "Max": round(max_r, 2)
            })
        else:
            print(f"{algo:<20} {'MISSING':>8} {'-':>8} {'-':>8} {'-':>8}")

    if results:
        plot_bar_and_box(results, results_raw)
        plot_lineplot(results_raw)

        df = pd.DataFrame(table_data)
        print("\n📊 Performance Summary Table:")
        print(df.to_string(index=False))

        csv_name = f"results_{timesteps_str}_{mode_str}.csv"
        df.to_csv(csv_name, index=False)
        print(f"\n✅ CSV saved as: {csv_name}")

if __name__ == "__main__":
    run_evaluation()
