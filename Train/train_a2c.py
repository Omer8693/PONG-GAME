
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging
from stable_baselines3 import A2C
import Constants
from Env import PongEnv

def train_a2c(complex_mode=False):
    env = PongEnv(complex_mode=complex_mode)
    try:
        mode_str = "complex" if complex_mode else "simple"
        print(f"Training A2C for {Constants.total_timesteps} timesteps ({mode_str} mode)...")
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=Constants.learning_rate,
            ent_coef=Constants.ent_coef,
            gamma=Constants.gamma,
            gae_lambda=Constants.gae_lambda,
            max_grad_norm=Constants.max_grad_norm,
            verbose=0,  # Logları ekrana yazdırma
            tensorboard_log=Constants.tensorboard_log
        )
        model.learn(
            total_timesteps=Constants.total_timesteps,
            log_interval=1,
            tb_log_name=f"{Constants.A2C_sub_folder}_{mode_str}",
            progress_bar=False
        )

        
        save_path = os.path.join(Constants.Model_Save_Path, f"A2C_{Constants.timesteps_str}_{mode_str}.zip")
        os.makedirs(Constants.Model_Save_Path, exist_ok=True)  # Klasörü oluştur
        model.save(save_path)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"A2C model saved to {save_path}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        env.close()

if __name__ == "__main__":


    train_a2c(complex_mode=False)  # Basit mod
    train_a2c(complex_mode=True)   # Karmaşık mod



