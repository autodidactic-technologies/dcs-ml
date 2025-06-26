import os
import minigrid
import gymnasium as gym

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

# Create and wrap the MiniGrid environment
def make_env(env_name = "MiniGrid-DoorKey-6x6-v0", max_steps=100):
    env = gym.make(env_name, render_mode="rgb_array",max_steps=max_steps)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    return env

if __name__ == "__main__":
    env = make_env()

    # Define model save path
    os.makedirs("models", exist_ok=True)
    save_path = "models/ppo_minigrid_doorkey_6x6.zip"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.backends.cuda.is_built() and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Define PPO model
    model = PPO(
        "CnnPolicy",  # Because observations are images
        env,
        verbose=1,
        tensorboard_log="./ppo_logs/",
        device=device


    )

    # Optional: checkpoint every N steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./models/",
        name_prefix="ppo_minigrid_doorkey_6x6"
    )

    # Train the model
    model.learn(total_timesteps=150_000, callback=checkpoint_callback)

    # Save the final model
    model.save(save_path)
    print(f"\n✅ PPO model saved to: {save_path}")
