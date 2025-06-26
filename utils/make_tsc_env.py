import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

def make_env(env_name="MiniGrid-DoorKey-6x6-v0", max_steps=100):
    """
    Creates a MiniGrid environment with visual observation wrappers and returns
    a factory function suitable for vectorized environments like SubprocVecEnv.
    """
    def _init():
        env = gym.make(env_name, render_mode="human", max_steps=max_steps)
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)
        env = Monitor(env)  # optional logging
        return env

    return _init