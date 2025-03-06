import gym
import numpy as np
import socket
import json
import time

class DCSGymEnv(gym.Env):
    def __init__(self):
        super(DCSGymEnv, self).__init__()

        # UDP Socket for communication with DCS
        self.server_address = ('127.0.0.1', 5005)  # Update with actual DCS export socket address
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)

        # Observation Space (Flattened vector)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        # Action Space
        self.action_space = gym.spaces.Discrete(4)  # 0: Nothing, 1: Fire, 2: Release Countermeasures, 3: Follow Enemy

    def reset(self):
        """ Reset environment and get initial state from DCS """
        state = self._get_dcs_state()
        return state

    def step(self, action):
        """ Take an action and receive new state, reward, and done flag """
        self._send_action_to_dcs(action)
        time.sleep(0.1)  # Small delay for sync

        next_state = self._get_dcs_state()
        reward = self._calculate_reward(next_state, action)
        done = False  # Modify if needed

        return next_state, reward, done, {}

    def _get_dcs_state(self):
        """ Receives data from DCS and returns structured observations """
        try:
            self.sock.sendto(b"GET_STATE", self.server_address)
            data, _ = self.sock.recvfrom(4096)  # Expecting JSON data
            state = json.loads(data.decode('utf-8'))

            ego = state["ego"]
            enemies = state["enemies"]
            allies = state["allies"]

            obs = [
                ego["speed"], ego["altitude_sea_level"], ego["altitude_ground"], ego["heading"],
                ego["munition_count"],  # Total count of missiles
            ]

            # Add first 3 enemy aircraft data
            for enemy in enemies[:3]:
                obs += [enemy["distance"], enemy["speed"], enemy["heading"], enemy["coalition"]]

            # Add first 3 ally aircraft data
            for ally in allies[:3]:
                obs += [ally["distance"], ally["speed"], ally["heading"], ally["coalition"]]

            return np.array(obs, dtype=np.float32)
        except Exception as e:
            print(f"Error receiving state from DCS: {e}")
            return np.zeros(21, dtype=np.float32)  # Return a default state

    def _send_action_to_dcs(self, action):
        """ Sends action to DCS """
        action_data = json.dumps({"action": action})
        self.sock.sendto(action_data.encode('utf-8'), self.server_address)

    def _calculate_reward(self, state, action):
        """ Defines a simple reward mechanism """
        ego_speed, ego_alt, _, ego_heading, munition_count = state[:5]

        reward = 0
        if action == 1:  # Fire
            if munition_count > 0:
                reward = 10  # Reward for attempting a fire
            else:
                reward = -5  # Penalty for trying to fire without munitions

        elif action == 2:  # Release Countermeasure
            reward = 5  # Reward for defensive action

        elif action == 3:  # Follow Enemy
            reward = 1  # Small positive reward for moving towards enemy

        return reward

    def close(self):
        """ Close socket on environment shutdown """
        self.sock.close()
