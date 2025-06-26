import gymnasium as gym
import torch
from stable_baselines3 import PPO
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import numpy as np

def make_env(env_name="MiniGrid-DoorKey-6x6-v0", max_steps=100):
    env = gym.make(env_name, render_mode="rgb_array", max_steps=max_steps)  # show window
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    return env

if __name__ == "__main__":
    # Set device (with MPS/CUDA fallback)
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

    num_episodes = 10
    rewards = []
    steps = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            env.render()

        rewards.append(episode_reward)
        steps.append(step_count)
        print(f"[Episode {episode+1}/{num_episodes}] Reward: {episode_reward:.2f}, Steps: {step_count}")

    env.close()

    print("\n✅ Evaluation Summary")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Average Steps: {np.mean(steps):.2f}")
    print(f"Success Rate (Reward > 0): {np.mean([r > 0 for r in rewards]) * 100:.1f}%")
