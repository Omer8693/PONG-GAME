# === Window Configuration ===
# Screen dimensions for the Pong game
WIDTH, HEIGHT = 900, 650

# === Agent (Paddle) Configuration ===
MOVE_COEF = 4                     # Movement speed
PADDLE_HEIGHT = 60               # Initial paddle height
PADDLE_WIDTH = 10                # Paddle width

# === Ball Configuration ===
BALL_Xspeed = 2                  # Initial horizontal speed
BALL_Yspeed = 2                  # Initial vertical speed

# === Training Hyperparameters ===
# These values control learning behavior across all agents
learning_rate = 0.0004
ent_coef = 0.01
gamma = 0.95
gae_lambda = 0.95
max_grad_norm = 0.5
total_timesteps = 200000        # Total training duration, 50k, 100k, 150k, 200k,

# === Logging and Saving Configuration ===
Model_Save_Path = "./models/"           # Directory to save trained models
tensorboard_log = "./Pong_Log/"         # TensorBoard logging directory

# === Subfolder Names Based on Timesteps ===
# Converts the total timesteps to a readable string (e.g., "200k")
timesteps_str = f"{int(total_timesteps / 1000)}k"
A2C_sub_folder = f"A2C_{timesteps_str}"
ExpectedSARSA_sub_folder = f"ExpectedSARSA_{timesteps_str}"
REINFORCE_sub_folder = f"REINFORCE_{timesteps_str}"
