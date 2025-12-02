import gym
import numpy as np
from gym.spaces import Box, Discrete


class GoalObsActionWrapper(gym.Wrapper):
    """
    Wrapper for GoalEnv to:
    - Flatten observation: use only the continuous state vector (14 dims), drop the discrete step counter.
    - Provide helper methods to build parameter placeholders for the action tuple.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Observation: take only the continuous state Box
        assert isinstance(env.observation_space, gym.spaces.Tuple)
        box: Box = env.observation_space.spaces[0]
        self._orig_obs_space: Box = box
        self.observation_space = Box(low=box.low.astype(np.float32),
                                     high=box.high.astype(np.float32),
                                     dtype=np.float32)
        # Action space: keep original for stepping, but expose a simplified description
        self.action_space = env.action_space

        # Cache per-branch parameter shapes (lengths)
        assert isinstance(self.action_space, gym.spaces.Tuple)
        assert isinstance(self.action_space.spaces[0], Discrete)
        self.n_actions = int(self.action_space.spaces[0].n)
        self.param_spaces = []
        for i in range(self.n_actions):
            ps: Box = self.action_space.spaces[1].spaces[i]
            self.param_spaces.append(ps)
        self.max_param_dim = max(int(np.prod(ps.shape)) for ps in self.param_spaces)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._flatten_obs(obs)

    def step(self, action):
        """
        Expects action as a tuple: (act_index: int, param_vector: np.ndarray for the chosen branch)
        Converts it into the env's required tuple format: (index, (params_for_each_branch, ...)).
        """
        act_index, chosen_params = action
        # Build full parameter tuple
        param_tuple = []
        for i, ps in enumerate(self.param_spaces):
            if i == act_index:
                # Ensure correct shape and clipping to bounds
                p = np.asarray(chosen_params, dtype=np.float32)
                if ps.shape:
                    p = p.reshape(ps.shape)
                else:
                    p = np.array([p], dtype=np.float32)
                p = np.clip(p, ps.low, ps.high)
                param_tuple.append(p)
            else:
                # Placeholder zeros of correct shape within bounds
                zeros = np.zeros(ps.shape, dtype=np.float32)
                param_tuple.append(np.clip(zeros, ps.low, ps.high))
        env_action = (int(act_index), tuple(param_tuple))
        obs, reward, done, info = self.env.step(env_action)
        return self._flatten_obs(obs), reward, done, info

    def _flatten_obs(self, obs):
        # obs is (state_vec, steps_int)
        state_vec = obs[0]
        return np.asarray(state_vec, dtype=np.float32)
