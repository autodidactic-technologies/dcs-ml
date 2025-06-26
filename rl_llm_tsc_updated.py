import os
import torch
import gym
import minigrid
import numpy as np
from loguru import logger

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from TSCAssistant.tsc_assistant_updated import TSCAgent
from utils.make_tsc_env import make_env
from stable_baselines3 import PPO

# ====== Utility ======
def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.backends.cuda.is_built() and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def print_summary(rewards, steps):
    print("\n✅ Evaluation Summary")
    print(f"Avg Reward: {np.mean(rewards):.2f}")
    print(f"Avg Steps: {np.mean(steps):.1f}")
    print(f"Success Rate: {np.mean([r > 0 for r in rewards]) * 100:.1f}%")

# ====== Main Evaluation Loop ======
if __name__ == '__main__':
    # Setup
    device = get_device()
    print("Using device:", device)

    llm_model_name = "llama3"
    chat = ChatOllama(model=llm_model_name, temperature=0.0)
    llm = RunnableLambda(lambda x: chat.invoke(x))

    env_name = "MiniGrid-DoorKey-6x6-v0"
    env = make_env(env_name=env_name, max_steps=100)()

    model_path = "models/ppo_minigrid_doorkey_6x6.zip"
    model = PPO.load(model_path, device=device)
    tsc_agent = TSCAgent(llm=llm, verbose=True)

    # Evaluation settings
    num_episodes = 10
    llm_frequency = 3  # every n steps LLM refines action

    all_rewards = []
    all_steps = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        sim_step = 0
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            if sim_step % llm_frequency == 0:
                action, _ = tsc_agent.agent_run(
                    sim_step=sim_step,
                    obs=obs,
                    action=action,
                    infos={"env": env_name}
                )

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            sim_step += 1
            env.render()

        all_rewards.append(total_reward)
        all_steps.append(sim_step)
        print(f"[Episode {episode+1}] Reward: {total_reward:.2f}, Steps: {sim_step}")

    env.close()
    print_summary(all_rewards, all_steps)
