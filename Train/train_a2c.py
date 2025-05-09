import sys
import os
import logging

# Add parent directory to sys.path so imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import A2C
import Constants
from Env import PongEnv

def train_a2c(complex_mode=False):
    """
    Trains an A2C agent on the Pong environment in either simple or complex mode.
    """
    env = PongEnv(complex_mode=complex_mode)
    
    try:
        # Mode string for folder/model naming
        mode_str = "complex" if complex_mode else "simple"
        print(f"Training A2C for {Constants.total_timesteps} timesteps ({mode_str} mode)...")
        
        # Initialize A2C model with hyperparameters from Constants
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=Constants.learning_rate,
            ent_coef=Constants.ent_coef,
            gamma=Constants.gamma,
            gae_lambda=Constants.gae_lambda,
            max_grad_norm=Constants.max_grad_norm,
            verbose=0,  # Suppress console output
            tensorboard_log=Constants.tensorboard_log
        )

        # Train the model
        model.learn(
            total_timesteps=Constants.total_timesteps,
            log_interval=1,
            tb_log_name=f"{Constants.A2C_sub_folder}_{mode_str}",
            progress_bar=False
        )

        # Create directory if it doesn't exist, then save the model
        save_path = os.path.join(Constants.Model_Save_Path, f"A2C_{Constants.timesteps_str}_{mode_str}.zip")
        os.makedirs(Constants.Model_Save_Path, exist_ok=True)
        model.save(save_path)

        # Logging the save operation
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"A2C model saved to {save_path}")

    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        env.close()

if __name__ == "__main__":
    # Train in both simple and complex mode
    train_a2c(complex_mode=False)  # Simple mode
    train_a2c(complex_mode=True)   # Complex mode
