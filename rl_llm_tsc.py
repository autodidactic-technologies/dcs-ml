import os, torch, gym, minigrid
from loguru import logger
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from TSCAssistant.tsc_assistant import TSCAgent
from utils.make_tsc_env import make_env
from stable_baselines3 import PPO

# Optional utility
import json
def dict_to_str(d): return json.dumps(d, indent=2)
# Set device
if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.backends.cuda.is_built() and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)
if __name__ == '__main__':
    # ====== LLM Setup ======
    llm_model_name = "llama3"
    llm_temperature = 0.0
    chat = ChatOllama(model=llm_model_name, temperature=llm_temperature)
    llm = RunnableLambda(lambda x: chat.invoke(x))

    # ====== Env and PPO Model ======
    env = make_env()()
    device=device
    model_path = "models/ppo_minigrid_doorkey_6x6.zip"
    model = PPO.load(model_path, device=device)

    # ====== LLM Agent + Inference Loop ======
    tsc_agent = TSCAgent(llm=llm, verbose=True)
    obs, info = env.reset()
    done = False
    sim_step = 0
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        if sim_step % 4 == 0:
            action = tsc_agent.agent_run(
                sim_step=sim_step,
                action=action,
                obs=obs,
                infos={"env": "MiniGrid-DoorKey-6x6-v0"}
            )

        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        total_reward += reward
        sim_step += 1
        env.render()

    print(f"\n✅ Done in {sim_step} steps, Total Reward: {total_reward}")
    env.close()
