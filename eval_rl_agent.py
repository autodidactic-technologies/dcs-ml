import gymnasium as gym
import torch
from stable_baselines3 import PPO
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import numpy as np

def make_env(env_name="MiniGrid-DoorKey-6x6-v0", max_steps=100):
    env = gym.make(env_name, render_mode="human", max_steps=max_steps)  # Show window
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    return env

if __name__ == "__main__":
    # Set device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.backends.cuda.is_built() and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Load env and model
    env = make_env()
    model_path = "models/ppo_minigrid_doorkey_6x6.zip"
    model = PPO.load(model_path, device=device)

    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

    env.close()
    print(f"✅ Evaluation done in {step} steps, total reward: {total_reward}")
