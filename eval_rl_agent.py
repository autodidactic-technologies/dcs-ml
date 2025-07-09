import gymnasium as gym
import torch
from stable_baselines3 import PPO
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import numpy as np
import wandb

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
    model_path = "models/ppo_minigrid_doorkey_6x6_250000_steps_yedek"
    model = PPO.load(model_path, device=device)

    num_episodes = 100
    rewards = []
    steps = []

    wandb.init(
        project="PPO_MiniGrid_Training",  # use same project as training for consistency
        entity="BILGEM_DCS_RL",
        name=f"ppo_minigrid_doorkey_eval_ppo_{num_episodes}_episodes",
        config={
            "env_name": "MiniGrid-DoorKey-6x6-v0",
            "model_path": model_path,
            "num_episodes": num_episodes
        }
    )
    successes = []
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
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
        is_success = episode_reward > 0
        successes.append(is_success)
        cumulative_avg_success = np.mean(successes) * 100
        print(f"[Episode {episode+1}/{num_episodes}] Reward: {episode_reward:.2f}, Steps: {step_count}")
        wandb.log({
            "episode": episode + 1,
            "episode_reward": episode_reward,
            "episode_steps": step_count,
            "cumulative_avg_success_rate": cumulative_avg_success
        })

    env.close()
    wandb.log({
        "average_reward": np.mean(rewards),
        "average_steps": np.mean(steps),
        "success_rate": np.mean([r > 0 for r in rewards])*100
    })

    print("\n✅ Evaluation Summary")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Average Steps: {np.mean(steps):.2f}")
    print(f"Success Rate (Reward > 0): {np.mean([r > 0 for r in rewards]) * 100:.1f}%")
