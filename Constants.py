# Window Configure
WIDTH, HEIGHT = 900, 650

# Agent Configure
MOVE_COEF = 4
PADDLE_HEIGHT = 60
PADDLE_WIDTH = 10

# Ball Configure
BALL_Xspeed = 2
BALL_Yspeed = 2

# Hyperparameters
learning_rate = 0.0004
ent_coef = 0.01
gamma = 0.95
gae_lambda = 0.95
max_grad_norm = 0.5
total_timesteps = 200000  # 

# Logging and Saving
Model_Save_Path = "./models/"
tensorboard_log = "./Pong_Log/"

# Timesteps'e göre dinamik alt klasör isimleri
timesteps_str = f"{int(total_timesteps / 1000)}k"  # Örn: 200000 → "200k"
A2C_sub_folder = f"A2C_{timesteps_str}"
ExpectedSARSA_sub_folder = f"ExpectedSARSA_{timesteps_str}"
REINFORCE_sub_folder = f"REINFORCE_{timesteps_str}"