# HarfangEnv_GYM.py
"""
Harfang Gym-style environment with vectorized missile tails.

Major points
------------
- Preserves existing reward/termination/tactics. No numerical changes.
- Returns gym-style step tuple: (obs, reward, done, info).
  * info includes: opponent_obs, success (step flag), episode_success, now_missile_state, missile1_state, n_missile1_state
- Observations are **purely numeric** (stable_baselines-friendly).
  * Missile data is vectorized into fixed-length packs: [present, mx, my, mz, heading_deg] * MAX_TRACKED_MISSILES
  * Absolute missile positions are in meters (unchanged assumption used by evasive logic).
- Adds observation_space with dynamic size computed at reset.
- Centralizes primitive actions in Action_Helper; SimpleEnemy extends base env for a minimalist adversary.

State layout (indices)
----------------------
0..2     : Pos_Diff = [dx, dy, dz] from Ally POV (normalized by 1e4)
3..5     : Ally Euler angles (normalized by NormStates["Plane_Euler_angles"])
6        : target_angle (degrees, as provided by sim)
7        : ally target_locked  (1 if True else -1)
8        : ally missile1_state (1 if slot present else -1)
9..11    : Opponent Euler angles (normalized)
12       : Opponent health level (0..1)
13..15   : Ally position (normalized by 1e4)
16..18   : Opponent position (normalized by 1e4)
19       : Ally heading (degrees)
20       : Ally health level (0..1)
21       : Ally pitch attitude (normalized)
22..     : Missile vector packs (MAX_TRACKED_MISSILES * 5 floats):
           For each i in [0..MAX-1]:
             [present, mx, my, mz, heading_deg]
           present ∈ {0,1}, positions in meters (absolute), heading in degrees.
"""
import numpy as np
import gym
import os
import inspect
import random
import math
import re
import time

# harfang SDK
from . import dogfight_client as df
from .constants import *  # NormStates etc.

MAX_TRACKED_MISSILES = 4
MISSILE_PACK_LEN = 5  # [present, mx, my, mz, heading]


class HarfangEnv:
    """
    Harfang air-combat environment.

    Notes
    -----
    * For algorithmic compatibility with stable_baselines, observations contain only numeric
      values, with missile info vectorized into fixed-length packs at the tail.
    * step() returns (obs, reward, done, info), where `info["opponent_obs"]` provides the
      opponent's observation vector for symmetric/scripted policies outside RL use.
    """

    def __init__(self):
        # --- runtime flags/state ---
        self.done = False
        self.loc_diff = 0.0
        self.success = 0
        self.episode_success = False
        self.fire_success = False
        self.now_missile_state = False

        # IDs
        self.Plane_ID_oppo = "ennemy_2"
        self.Plane_ID_ally = "ally_1"

        # Health/bookkeeping
        self.oppo_health = 0.2
        self.ally_health = 1.0

        # Locks/missile slot snapshot
        self.Ally_target_locked = False
        self.n_Ally_target_locked = False
        self.missile1_state = True
        self.n_missile1_state = True

        self.target_angle = 0.0
        self.Plane_Irtifa = 0.0

        # Missile tracker
        self.missile_handler = MissileHandler()

        # Spaces (action fixed; observation computed after first reset)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # observation_space will be finalized at first reset (depends on lengths)
        self.observation_space = None

    # ------------------------------- Public API -------------------------------- #
    def reset(self):
        """
        Reset simulation and return a pair (ally_obs, oppo_obs) for scripts that
        control both sides (rule-based). This mirrors the previous API.

        For gym/RL usage, prefer `reset_gym()` that returns (obs, info).
        """
        self._reset_episode_common()
        state_ally = self._get_observation()          # ally POV
        state_oppo = self._get_enemy_observation()    # opponent POV

        # finalize observation_space on first call
        if self.observation_space is None:
            obs_len = len(state_ally)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32
            )

        return state_ally, state_oppo

    def reset_gym(self):
        """
        Gym-style reset returning (obs, info).
        """
        ally_obs, oppo_obs = self.reset()
        return ally_obs, {"opponent_obs": oppo_obs}

    def step(self, action_ally, action_enemy):
        """
        Apply ally/opponent actions. Returns gym-style 4-tuple:

        Returns
        -------
        obs : np.ndarray
            Ally observation after step.
        reward : float
        done : bool
        info : dict
            - "opponent_obs": opponent observation after step
            - "success": per-step success flag used by previous code
            - "episode_success": episode-level success
            - "now_missile_state", "missile1_state", "n_missile1_state"
        """
        self._apply_action(action_ally, action_enemy)
        self.missile_handler.refresh_missiles()

        n_state = self._get_observation()
        n_state_oppo = self._get_enemy_observation()

        self._get_reward(self.state, action_ally, n_state)
        self.state = n_state
        self.oppo_state = n_state_oppo
        self._get_termination()

        info = {
            "opponent_obs": n_state_oppo,
            "success": self.success,
            "episode_success": self.episode_success,
            "now_missile_state": self.now_missile_state,
            "missile1_state": self.missile1_state,
            "n_missile1_state": self.n_missile1_state,
        }
        return n_state, float(self.reward), bool(self.done), info

    # Legacy helper (kept intact)
    def step_test(self, action):
        self._apply_action(action, [0.0, 0.0, 0.0, 0.0])
        n_state = self._get_observation()
        self._get_reward(self.state, action, n_state)
        self.state = n_state
        self._get_termination()
        return n_state, self.reward, self.done, {}, self.now_missile_state, self.missile1_state, self.n_missile1_state, self.Ally_target_locked, self.success

    # ------------------------------- Internals --------------------------------- #
    def _reset_episode_common(self):
        self.Ally_target_locked = False
        self.n_Oppo_target_locked = False
        self.n_Ally_target_locked = False
        self.missile1_state = True
        self.n_missile1_state = True
        self.success = 0
        self.done = False
        self.episode_success = False
        self.fire_success = False
        self.now_missile_state = False

        # Makine durumlarını sıfırla
        self._reset_machine()
        self._reset_missile()
        self.missile_handler.refresh_missiles()

        # 🔧 ESKİ DAVRANIŞIN GERİ GETİRİLMESİ — hedef ataması (KİLİT İÇİN ŞART)
        df.set_target_id(self.Plane_ID_ally, self.Plane_ID_oppo)  # ally hedefi: enemy
        df.set_target_id(self.Plane_ID_oppo, self.Plane_ID_ally)  # enemy hedefi: ally

        # State önbellekleri
        self.state = None
        self.oppo_state = None

    def _apply_action(self, action_ally, action_enemy):
        # Ally controls
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        # Opponent controls
        df.set_plane_pitch(self.Plane_ID_oppo, float(action_enemy[0]))
        df.set_plane_roll(self.Plane_ID_oppo, float(action_enemy[1]))
        df.set_plane_yaw(self.Plane_ID_oppo, float(action_enemy[2]))

        # Missile fire handling (finalized to use actual available slots per side)
        self.now_missile_state = False
        if float(action_ally[3]) > 0.0:
            ally_unfired_slots = self._unfired_slots(self.Plane_ID_ally)
            if ally_unfired_slots:
                df.fire_missile(self.Plane_ID_ally, min(ally_unfired_slots))
                self.now_missile_state = True

        if float(action_enemy[3]) > 0.0:
            oppo_unfired_slots = self._unfired_slots(self.Plane_ID_oppo)
            if oppo_unfired_slots:
                df.fire_missile(self.Plane_ID_oppo, min(oppo_unfired_slots))

        df.update_scene()

    @staticmethod
    def _unfired_slots(plane_id):
        """
        Return a list of integer slot indices that currently hold an unfired missile
        for the given plane_id.
        """
        # Slots e.g. ["ally_1AIM_SL0", "ally_1Meteor1", ...]
        slots = df.get_machine_missiles_list(plane_id)
        slot_state = df.get_missiles_device_slots_state(plane_id).get("missiles_slots", [])
        unfired = []
        for i, present in enumerate(slot_state):
            if i < len(slots) and bool(present):
                # Unfired missiles are those with position [0,0,0]
                missile_id_guess = MissileHandler.slotid_to_missileid(slots[i])
                try:
                    st = df.get_missile_state(missile_id_guess)
                    if list(st.get("position", [0.0, 0.0, 0.0])) == [0.0, 0.0, 0.0]:
                        unfired.append(i)
                except Exception:
                    # if state cannot be read, still allow fire by slot index
                    unfired.append(i)
        return unfired

    def _get_reward(self, state, action, n_state):
        # Preserved reward function
        self.reward = 0
        self.success = 0
        self._get_loc_diff()

        self.reward -= (0.0001 * self.loc_diff)
        self.reward -= self.target_angle * 10

        if self.Plane_Irtifa < 2000:
            self.reward -= 4
        if self.Plane_Irtifa > 7000:
            self.reward -= 4

        if self.now_missile_state is True:
            self.reward -= 8
            if self.missile1_state and (self.Ally_target_locked is False):
                self.success = -1
                print('failed to fire')
            elif self.missile1_state and (self.Ally_target_locked is True):
                print('successful to fire')
                self.success = 1
                self.fire_success = True
            else:
                self.reward -= 0

        if self.oppo_health['health_level'] <= 0.1 and self.fire_success:
            self.reward += 600
            print('enemy have fallen')

    def _get_termination(self):
        if self.Plane_Irtifa < 500 or self.Plane_Irtifa > 10000:
            self.done = True
        if self.oppo_health['health_level'] <= 0:
            self.done = True
            self.episode_success = True
        if self.ally_health['health_level'] < 1.0:
            self.done = True

    def _reset_machine(self):
        df.reset_machine("ally_1")
        df.reset_machine("ennemy_2")
        df.set_health("ennemy_2", 0.2)
        df.set_health("ally_1", 1.0)
        self.oppo_health = 0.2
        self.ally_health = 1.0

        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)

        df.set_plane_thrust(self.Plane_ID_ally, 1.0)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def _reset_missile(self):
        self.now_missile_state = False
        df.rearm_machine(self.Plane_ID_ally)
        df.rearm_machine(self.Plane_ID_oppo)

    def _get_loc_diff(self):
        self.loc_diff = (
            ((self.Aircraft_Loc[0] - self.Oppo_Loc[0]) ** 2) +
            ((self.Aircraft_Loc[1] - self.Oppo_Loc[1]) ** 2) +
            ((self.Aircraft_Loc[2] - self.Oppo_Loc[2]) ** 2)
        ) ** 0.5

    # ------------------------------- Observations ------------------------------- #
    def _get_observation(self):
        """
        Ally POV observation vector. See module docstring for indices.
        """
        plane = df.get_plane_state(self.Plane_ID_ally)
        oppo = df.get_plane_state(self.Plane_ID_oppo)

        Plane_Pos = [plane["position"][0] / NormStates["Plane_position"],
                     plane["position"][1] / NormStates["Plane_position"],
                     plane["position"][2] / NormStates["Plane_position"]]
        Plane_Euler = [plane["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                       plane["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                       plane["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]
        Plane_Heading = plane["heading"]  # degrees
        Plane_Pitch_Att = plane["pitch_attitude"] / NormStates["Plane_pitch_attitude"]
        Plane_Roll_Att = plane["roll_attitude"] / NormStates["Plane_roll_attitude"]  # noqa: F841

        Oppo_Pos = [oppo["position"][0] / NormStates["Plane_position"],
                    oppo["position"][1] / NormStates["Plane_position"],
                    oppo["position"][2] / NormStates["Plane_position"]]
        Oppo_Euler = [oppo["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                      oppo["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                      oppo["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]

        self.Plane_Irtifa = plane["position"][1]
        self.Aircraft_Loc = plane["position"]
        self.Oppo_Loc = oppo["position"]

        # Locks/flags
        self.Ally_target_locked = self.n_Ally_target_locked
        self.n_Ally_target_locked = plane["target_locked"]
        locked = 1 if self.n_Ally_target_locked else -1

        self.Oppo_target_locked = self.n_Oppo_target_locked
        self.n_Oppo_target_locked = oppo["target_locked"]
        oppo_locked = 1 if self.n_Oppo_target_locked else -1  # noqa: F841 (kept for symmetry)

        target_angle = plane['target_angle']
        self.target_angle = target_angle

        Pos_Diff = [Oppo_Pos[0] - Plane_Pos[0],
                    Oppo_Pos[1] - Plane_Pos[1],
                    Oppo_Pos[2] - Plane_Pos[2]]

        self.oppo_health = df.get_health(self.Plane_ID_oppo)
        self.ally_health = df.get_health(self.Plane_ID_ally)
        ally_hea = self.ally_health['health_level']
        oppo_hea = self.oppo_health['health_level']

        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_ally)
        self.missile1_state = self.n_missile1_state
        self.n_missile1_state = Missile_state["missiles_slots"][0] if Missile_state["missiles_slots"] else False
        missile1_state_val = 1 if self.n_missile1_state else -1

        # Vectorize incoming enemy missiles (absolute meters)
        missile_vec = self._vectorize_missiles(self.get_enemy_missile_vector())

        States = np.concatenate((
            Pos_Diff,                      # 0..2
            Plane_Euler,                   # 3..5
            [target_angle],                # 6
            [locked],                      # 7
            [missile1_state_val],          # 8
            Oppo_Euler,                    # 9..11
            [oppo_hea],                    # 12
            Plane_Pos,                     # 13..15
            Oppo_Pos,                      # 16..18
            [Plane_Heading],               # 19
            [ally_hea],                    # 20
            [Plane_Pitch_Att],             # 21
            missile_vec                    # 22..
        ), axis=None).astype(np.float32)

        self.state = States
        return States

    def _get_enemy_observation(self):
        """
        Opponent POV observation vector (same layout semantics as ally POV).
        """
        plane = df.get_plane_state(self.Plane_ID_oppo)  # enemy self
        oppo = df.get_plane_state(self.Plane_ID_ally)   # ally as opponent from enemy POV

        Plane_Pos = [plane["position"][0] / NormStates["Plane_position"],
                     plane["position"][1] / NormStates["Plane_position"],
                     plane["position"][2] / NormStates["Plane_position"]]
        Oppo_Pos = [oppo["position"][0] / NormStates["Plane_position"],
                    oppo["position"][1] / NormStates["Plane_position"],
                    oppo["position"][2] / NormStates["Plane_position"]]

        Plane_Euler = [plane["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                       plane["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                       plane["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]
        Oppo_Euler = [oppo["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                      oppo["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                      oppo["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]

        Plane_Heading = plane["heading"]
        Plane_Pitch_Att = plane["pitch_attitude"] / NormStates["Plane_pitch_attitude"]  # noqa: F841
        Plane_Roll_Att = plane["roll_attitude"] / NormStates["Plane_roll_attitude"]      # noqa: F841

        n_Oppo_target_locked = plane["target_locked"]
        locked = 1 if n_Oppo_target_locked else -1

        n_Ally_target_locked = oppo["target_locked"]  # noqa: F841 (kept for parity)
        target_angle = plane["target_angle"]

        Pos_Diff = [Oppo_Pos[0] - Plane_Pos[0],
                    Oppo_Pos[1] - Plane_Pos[1],
                    Oppo_Pos[2] - Plane_Pos[2]]

        missile_vec = self._vectorize_missiles(self.get_ally_missile_vector())

        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_oppo)
        missile1_state_val = 1 if (Missile_state["missiles_slots"] and Missile_state["missiles_slots"][0]) else -1

        oppo_health = df.get_health(self.Plane_ID_oppo)
        oppo_hea = oppo_health['health_level']

        States = np.concatenate((
            Pos_Diff,            # 0..2
            Plane_Euler,         # 3..5
            [target_angle],      # 6
            [locked],            # 7
            [missile1_state_val],# 8
            Oppo_Euler,          # 9..11 (ally euler)
            [oppo_hea],          # 12 (enemy self health)
            Plane_Pos,           # 13..15 (enemy self pos)
            Oppo_Pos,            # 16..18 (ally pos)
            [Plane_Heading],     # 19 (enemy heading)
            missile_vec          # 20..
        ), axis=None).astype(np.float32)

        self.oppo_state = States
        return States

    # --------------------------- Missile vector helpers -------------------------- #
    def _vectorize_missiles(self, missiles):
        """
        Convert a list of missile dicts into a fixed-length float vector:
        [present, mx, my, mz, heading_deg] * MAX_TRACKED_MISSILES
        Using absolute position values in meters to preserve original evasive logic.
        """
        vec = np.zeros((MAX_TRACKED_MISSILES * MISSILE_PACK_LEN,), dtype=np.float32)
        if not missiles:
            return vec
        count = min(len(missiles), MAX_TRACKED_MISSILES)
        for i in range(count):
            m = missiles[i]
            base = i * MISSILE_PACK_LEN
            pos = m.get("position", [0.0, 0.0, 0.0])
            vec[base + 0] = 1.0
            vec[base + 1] = float(pos[0])
            vec[base + 2] = float(pos[1])
            vec[base + 3] = float(pos[2])
            vec[base + 4] = float(m.get("heading", 0.0))
        return vec

    def _parse_missiles_from_state(self, state):
        """
        Inverse of _vectorize_missiles() for use by action helpers that expect a list
        of dict missiles from the state tail. This preserves previous behavior while
        keeping the observation fully numeric for RL.
        """
        missiles = []
        start = 22
        if len(state) <= start:
            return missiles
        for i in range(MAX_TRACKED_MISSILES):
            base = start + i * MISSILE_PACK_LEN
            if base + 4 >= len(state):
                break
            present = state[base + 0] > 0.5
            if not present:
                continue
            mx = float(state[base + 1]); my = float(state[base + 2]); mz = float(state[base + 3])
            hdg = float(state[base + 4])
            if (mx, my, mz) == (0.0, 0.0, 0.0):
                continue
            missiles.append({
                "missile_id": f"M{i}",
                "position": [mx, my, mz],
                "heading": hdg,
            })
        return missiles

    # Public missile queries (unchanged semantics)
    def get_enemy_missile_vector(self):
        """List of dicts for all current enemy missiles."""
        self.missile_handler.refresh_missiles()
        missile_info_list = []
        for mid in sorted(self.missile_handler.enemy_missiles):
            try:
                state = df.get_missile_state(mid)
                if not state.get("wreck", False):
                    missile_info_list.append({
                        "missile_id": mid,
                        "position": list(state["position"][:3]),
                        "heading": state.get("heading", 0),
                    })
            except Exception:
                pass
        return missile_info_list

    def get_ally_missile_vector(self):
        """List of dicts for all current ally missiles."""
        self.missile_handler.refresh_missiles()
        missile_info_list = []
        for mid in sorted(self.missile_handler.ally_missiles):
            try:
                state = df.get_missile_state(mid)
                if not state.get("wreck", False):
                    missile_info_list.append({
                        "missile_id": mid,
                        "position": list(state["position"][:3]),
                        "heading": state.get("heading", 0),
                    })
            except Exception:
                pass
        return missile_info_list

    # ----------------------------- Utility / Logging ---------------------------- #
    def get_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_ally)
        return np.array(plane_state["position"][:3])

    def get_oppo_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_oppo)
        return np.array(plane_state["position"][:3])

    def save_parameters_to_txt(self, log_dir):
        source_code1 = inspect.getsource(self._get_reward)
        source_code2 = inspect.getsource(self._reset_machine)
        source_code3 = inspect.getsource(self._get_termination)
        filename = os.path.join(log_dir, "log2.txt")
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(source_code1 + '\n')
            file.write(source_code2 + '\n')
            file.write(source_code3 + '\n')

    # --------------------------- Expert-data helpers ---------------------------- #
    def get_loc_diff(self, state):
        loc_diff = (((state[0] * 10000) ** 2) + ((state[1] * 10000) ** 2) + ((state[2] * 10000) ** 2)) ** 0.5
        return loc_diff

    def get_reward(self, state, action, n_state):
        # preserved copy of the compact reward for data extraction
        reward = 0
        step_success = 0
        loc_diff = self.get_loc_diff(n_state)
        reward -= (0.0001 * loc_diff)
        reward -= (n_state[6]) * 10
        if action[-1] > 0:
            reward -= 8
            if state[8] > 0 and state[7] < 0:
                step_success = -1
            elif state[8] > 0 and state[7] > 0:
                step_success = 1
            else:
                reward -= 0
        if n_state[-1] < 0.1:
            reward += 600
        return reward, step_success

    def get_termination(self, state):
        return bool(state[-1] <= 0.1)


# ----------------------------- Missile management ------------------------------ #
class MissileHandler:
    MISSILE_TYPES = ["AIM_SL", "Meteor", "Karaoke", "Mica", "S400", "Sidewinder"]

    def __init__(self):
        self.ally_id = "ally_1"
        self.enemy_id = "ennemy_2"
        self.ally_slots = []
        self.enemy_slots = []
        self.ally_missiles = set()
        self.enemy_missiles = set()
        self.refresh_missiles()

    def refresh_missiles(self):
        self.ally_slots = df.get_machine_missiles_list(self.ally_id)
        self.ally_slot_states = df.get_missiles_device_slots_state(self.ally_id).get("missiles_slots", [])
        self.ally_missiles = self._get_current_missiles("ally")

        self.enemy_slots = df.get_machine_missiles_list(self.enemy_id)
        self.enemy_slot_states = df.get_missiles_device_slots_state(self.enemy_id).get("missiles_slots", [])
        self.enemy_missiles = self._get_current_missiles("ennemy")

        self.ally_missiles = set([mid for mid in self.ally_missiles if not self.is_missile_wreck(mid)])
        self.enemy_missiles = set([mid for mid in self.enemy_missiles if not self.is_missile_wreck(mid)])

    def _get_current_missiles(self, side_str):
        missiles = df.get_missiles_list()
        filtered = []
        for mid in missiles:
            if any(t in mid for t in self.MISSILE_TYPES) and mid.startswith(side_str):
                filtered.append(mid)
        return set(filtered)

    @staticmethod
    def slotid_to_missileid(slot_id):
        m = re.match(r"(\w+?)_?(AIM_SL|Meteor|Karaoke|Mica|S400|Sidewinder)(\d+)", slot_id)
        if m:
            prefix, mtype, num = m.groups()
            return f"{prefix}-{mtype}-{num}"
        return slot_id.replace("_", "-")

    @staticmethod
    def missileid_to_slotid(missile_id):
        parts = missile_id.split('-')
        if len(parts) >= 3:
            return f"{parts[0]}{parts[1]}{parts[2]}"
        return missile_id.replace("-", "")

    def missile_slot_status(self, plane_id):
        slots = df.get_machine_missiles_list(plane_id)
        slot_state = df.get_missiles_device_slots_state(plane_id).get("missiles_slots", [])
        results = []
        for idx, slot_id in enumerate(slots):
            present = slot_state[idx] if idx < len(slot_state) else False
            missile_id_guess = self.slotid_to_missileid(slot_id)
            missile_state = None
            wreck = None
            pos = None
            if present:
                try:
                    missile_state = df.get_missile_state(missile_id_guess)
                    wreck = missile_state.get("wreck", None)
                    pos = missile_state.get("position", None)
                except Exception:
                    wreck = None
            results.append({
                "slot_idx": idx,
                "slot_id": slot_id,
                "missile_id": missile_id_guess,
                "slot_active": present,
                "wreck": wreck,
                "position": pos
            })
        return results

    def is_missile_wreck(self, missile_id):
        try:
            missile_state = df.get_missile_state(missile_id)
            return missile_state.get("wreck", False)
        except Exception:
            return True

    def track_all_missiles(self, side="ally", steps=10, print_missing=True):
        self.refresh_missiles()
        missile_ids = list(self.ally_missiles) if side == "ally" else list(self.enemy_missiles)
        if not missile_ids:
            print(f"[{side.upper()}] No active missiles.")
            return
        for mid in missile_ids:
            for i in range(steps):
                try:
                    state = df.get_missile_state(mid)
                    pos = state.get("position", None)
                    wreck = state.get("wreck", None)
                    if pos is not None and wreck is not True:
                        print(f"  [{side.upper()} MISSILE:{mid}] Step {i:02d}: x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f}")
                    else:
                        if print_missing:
                            print(f"  [{side.upper()} MISSILE:{mid}] Step {i:02d}: (wreck/position missing)")
                        break
                except Exception:
                    if print_missing:
                        print(f"  [{side.upper()} MISSILE:{mid}] Step {i:02d}: (state unavailable)")
                    break

    def missile_summary(self):
        self.refresh_missiles()
        print(f"ALLY missiles: {sorted(self.ally_missiles)}")
        print(f"ENEMY missiles: {sorted(self.enemy_missiles)}")
        print("Ally slot state:", self.missile_slot_status(self.ally_id))
        print("Enemy slot state:", self.missile_slot_status(self.enemy_id))


# --------------------------- Minimal enemy environment ------------------------- #
class SimpleEnemy(HarfangEnv):
    """
    Minimal adversary wrapper that still respects the finalized slot-based firing.
    """

    def __init__(self):
        super(SimpleEnemy, self).__init__()
        self.has_fired = False

    def _apply_action(self, action_ally, action_enemy):
        # Apply basic controls
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        df.set_plane_pitch(self.Plane_ID_oppo, float(action_enemy[0]))
        df.set_plane_roll(self.Plane_ID_oppo, float(action_enemy[1]))
        df.set_plane_yaw(self.Plane_ID_oppo, float(action_enemy[2]))

        # Slot-based fires (ally)
        if float(action_ally[3]) > 0.0:
            ally_unfired_slots = self._unfired_slots(self.Plane_ID_ally)
            if ally_unfired_slots:
                df.fire_missile(self.Plane_ID_ally, min(ally_unfired_slots))
                self.now_missile_state = True
                print(" === ally fired missile! ===")
        else:
            self.now_missile_state = False

        # Slot-based fires (enemy)
        if float(action_enemy[3]) > 0.0:
            oppo_unfired_slots = self._unfired_slots(self.Plane_ID_oppo)
            if oppo_unfired_slots:
                df.fire_missile(self.Plane_ID_oppo, min(oppo_unfired_slots))
                print(" === enemy fired missile! ===")

        df.update_scene()


# ----------------------------- Primitive controllers --------------------------- #
class Action_Helper:
    """
    Centralized low-level control primitives used by Agents.
    Numerics and tactical choices preserved from the original code.
    """

    def track_cmd(self, state):
        dx, dy, dz = state[0] * 10000, state[2] * 10000, state[1] * 10000
        dx_norm, dy_norm, dz_norm = state[0], state[2], state[1]
        plane_heading = state[19]             # degrees
        altitude = state[14] * 10000          # meters
        target_angle = state[6]

        # Relative bearing to target (degrees in [-180, 180])
        angle_to_enemy = np.degrees(np.arctan2(dx, dy))
        relative_bearing = (angle_to_enemy - plane_heading + 180) % 360 - 180

        # Pitch command with altitude guard
        if altitude < 1200:
            pitch_cmd = -0.3
        elif altitude > 8000:
            pitch_cmd = 0.3
        else:
            plane_pitch_norm = state[3]
            plane_pitch = math.degrees(plane_pitch_norm * math.pi) * (-1)

            horiz_dist = np.sqrt(dx**2 + dy**2)
            pitch_to_target = np.degrees(np.arctan2(dz, horiz_dist))

            relative_pitch = (pitch_to_target - plane_pitch + 90) % 180 - 90
            gain = np.interp(horiz_dist, [0, 800], [1.5, 1.1])

            if horiz_dist < 800:
                pitch_cmd = np.clip(-0.03 * relative_pitch * gain / 30, -1, 1)
            else:
                pitch_cmd = np.clip(dz_norm * -0.2, -1, 1)

            if abs(relative_pitch) > 0.5:
                pitch_cmd = -0.25 if relative_pitch > 0 else 0.25

        roll_cmd = 0.0
        yaw_gain = 0.03 if abs(relative_bearing) < 10 else 0.06
        yaw_cmd = np.clip(relative_bearing * yaw_gain, -1.0, 1.0)

        return [float(pitch_cmd), float(roll_cmd), float(yaw_cmd), -1.0]

    def evade_cmd(self, state):
        """
        Evasive controls vs inbound missiles.
        Implementation keeps all original thresholds/shaping; only reads missiles
        from the numeric state tail (converted back to list of dicts internally).
        """
        import numpy as np

        def clamp(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
            return float(max(lo, min(hi, val)))

        def wrap_deg(angle: float) -> float:
            return (angle + 180.0) % 360.0 - 180.0

        def fast_tanh(x: float) -> float:
            return float(np.tanh(x))

        dt = float(getattr(self, "dt", 0.15)) or 0.15  # fallback

        # persistent bag
        if not hasattr(self, "_ev"):
            rng = np.random.RandomState(1234)
            self._ev = {
                "rng": rng,
                "prev": {},
                "lp": np.array([0.0, 0.0, 0.0]),
                "mid_until": 0.0,
                "mid_dir": None,
                "did_snap": False,
                "snap_dir": 0.0,
            }
        S = self._ev
        rng = S["rng"]

        # Ownship
        agent_pos = state[13:16] * 10000.0
        heading = float(state[19])
        try:
            altitude = float(state[14] * 10000.0)
        except Exception:
            altitude = float(agent_pos[1])

        # Recreate missile dicts from numeric packs
        missiles = []
        start = 22
        for i in range(MAX_TRACKED_MISSILES):
            base = start + i * MISSILE_PACK_LEN
            if base + 4 >= len(state):
                break
            present = state[base] > 0.5
            if not present:
                continue
            mx, my, mz = map(float, state[base + 1: base + 4])
            if (mx, my, mz) == (0.0, 0.0, 0.0):
                continue
            missiles.append({"missile_id": f"M{i}", "position": [mx, my, mz]})

        # No threat: decay commands
        if not missiles:
            tau = 0.18
            alpha = dt / (tau + dt)
            cmd_vec = (1 - alpha) * S["lp"] + alpha * np.array([0.0, 0.0, 0.0])
            S["lp"] = cmd_vec
            return [float(cmd_vec[0]), float(cmd_vec[1]), float(cmd_vec[2]), 0.0]

        # Threat metrics
        def compute_missile_metrics(missile: dict, idx: int) -> dict:
            mid = missile.get("missile_id", f"M{idx}")
            mx, my, mz = map(float, missile["position"])
            dx = mx - agent_pos[0]
            dy = my - agent_pos[1]
            dz = mz - agent_pos[2]
            rng_3d = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            az_world = np.degrees(np.arctan2(dx, dz))
            prev = S["prev"].get(mid, {"range": rng_3d, "az": az_world})
            closure = (prev["range"] - rng_3d) / dt
            dlos = wrap_deg(az_world - prev["az"]) / dt
            rbearing = wrap_deg(az_world - heading)
            t_go = (rng_3d / max(closure, 1e-3)) if closure > 1e-3 else 1e9
            S["prev"][mid] = {"range": rng_3d, "az": az_world}
            return {"id": mid, "range": rng_3d, "closure": closure, "tgo": t_go, "az": az_world, "rb": rbearing, "dLOS": dlos}

        metrics_all = [compute_missile_metrics(m, i) for i, m in enumerate(missiles)]
        ms = min(metrics_all, key=lambda k: k["tgo"])

        # PN-mirror core
        deg2rad = np.pi / 180.0
        V_rel = max(ms["closure"], 180.0)
        lamdot = abs(ms["dLOS"]) * deg2rad
        r = max(ms["range"], 200.0)

        g_cap = 8.0
        k1, k2 = 0.9, 0.6
        a_lat = k1 * V_rel * lamdot + k2 * (V_rel ** 2) / (r + 1.0)
        a_lat = np.clip(a_lat, 3.0, g_cap)

        # Phases
        T1, T2 = 7.0, 3.0
        t_go = ms["tgo"]
        rb = ms["rb"]
        rb_sign = -1.0 if rb > 0.0 else 1.0

        def yaw_to_beam(rbearing: float, gain: float = 0.06, scale: float = 60.0) -> float:
            desired = 90.0 if rbearing >= 0.0 else -90.0
            err = desired - rbearing
            dir_sign = -1.0 if rbearing > 0.0 else 1.0
            return float(np.clip(dir_sign * np.tanh(gain * (abs(err) / scale) * 60.0), -1.0, 1.0))

        def bank_cmd_from_yaw(yaw_cmd: float, mag: float = 1.1) -> float:
            return float(np.clip(np.tanh(mag * np.sign(yaw_cmd)), -1.0, 1.0))

        def g_to_pitch_cmd(g: float) -> float:
            return float(np.clip(g / g_cap, -1.0, 1.0))

        # Rear-aspect
        rear_aspect = abs(rb) > 90.0 and t_go > T2
        if rear_aspect:
            desired_rb = 90.0 if rb >= 0.0 else -90.0
            err = desired_rb - rb

            yaw_cmd = float(np.clip(np.tanh((err / 35.0) * 1.6), -1.0, 1.0))
            roll_cmd = float(np.clip(np.tanh(np.sign(yaw_cmd) * 1.3), -1.0, 1.0))
            pitch_cmd = g_to_pitch_cmd(min(g_cap * 1.1, a_lat * 1.2))

            if altitude < 1200.0:
                pitch_cmd = max(pitch_cmd, g_to_pitch_cmd(5.0))

            tau = 0.05
            alpha = dt / (tau + dt)
            vec = np.array([pitch_cmd, roll_cmd, yaw_cmd])
            smoothed = (1 - alpha) * S["lp"] + alpha * vec
            S["lp"] = smoothed
            P, R, Y = [float(np.clip(c, -1.0, 1.0)) for c in smoothed.tolist()]
            return [P, R, Y, -1.0]

        # Reset snap
        if t_go > 6.0:
            S["did_snap"] = False
            S["snap_dir"] = 0.0

        if t_go > T1:
            yaw_cmd = yaw_to_beam(rb, gain=0.045, scale=60.0) * 0.8
            roll_cmd = bank_cmd_from_yaw(yaw_cmd, mag=1.0) * 0.85
            pitch_cmd = g_to_pitch_cmd(min(a_lat, 4.0))
        elif t_go > T2:
            now_t = float(getattr(self, "_t", 0.0))
            if (S["mid_dir"] is None) or (now_t >= S["mid_until"]):
                S["mid_dir"] = (-rb_sign if rng.rand() < 0.35 else rb_sign)
                S["mid_until"] = now_t + rng.uniform(0.6, 0.9)
            yaw_cmd = S["mid_dir"] * abs(yaw_to_beam(rb, gain=0.06, scale=55.0))
            roll_cmd = bank_cmd_from_yaw(yaw_cmd, mag=1.15)
            pitch_cmd = g_to_pitch_cmd(min(a_lat, 6.5))
        else:
            if (t_go < 1.2) and (not S["did_snap"]):
                S["did_snap"] = True
                S["snap_dir"] = -rb_sign
            snap_dir = S["snap_dir"] if S["did_snap"] else rb_sign
            yaw_cmd = snap_dir * float(np.tanh(0.14 * (min(140.0, abs(90.0 - rb)) / 40.0) * 60.0))
            roll_cmd = float(np.clip(np.tanh(1.4 * np.sign(yaw_cmd)), -1.0, 1.0))
            pitch_cmd = g_to_pitch_cmd(g_cap) * 1.2

        # Altitude guard
        ALT_FLOOR = 1200.0
        if altitude < ALT_FLOOR:
            pitch_cmd = max(pitch_cmd, g_to_pitch_cmd(5.0))
            roll_cmd *= 0.9

        # Smoothing
        tau = 0.06 if t_go < 2.0 else (0.10 if t_go < 5.0 else 0.18)
        alpha = dt / (tau + dt)
        vec = np.array([pitch_cmd, roll_cmd, yaw_cmd])
        smoothed = (1 - alpha) * S["lp"] + alpha * vec
        S["lp"] = smoothed
        P, R, Y = [float(np.clip(c, -1.0, 1.0)) for c in smoothed.tolist()]
        return [P, R, Y, -1.0]

    def climb_cmd(self, state):
        # Preserved logic and tunings
        dbg_on = bool(getattr(self, "debug_climb", False))
        dbg_every = int(getattr(self, "debug_climb_every", 10))
        dt = float(getattr(self, "dt", 0.15)) or 0.15

        if not hasattr(self, "_climb_dbg"):
            self._climb_dbg = {"step": 0, "prev_alt": None}
        D = self._climb_dbg
        D["step"] += 1

        def _dbg_print(**kw):
            if not dbg_on:
                return
            if D["step"] % max(1, dbg_every) != 0:
                return
            print("[CLIMB DBG] " + " ".join(f"{k}={v}" for k, v in kw.items()))

        target_alt_m = 4000.0
        DEAD_BAND = 200.0
        Kp_up = 0.00005
        Kp_down = 0.00012
        Kd_att = 0.0015
        MAX_UP = 0.035
        MAX_DOWN = 0.080

        altitude = float(state[14] * 10000.0)
        alt_err = target_alt_m - altitude
        pitch_deg = float(state[19] * 180)

        if D["prev_alt"] is None:
            vz = 0.0
        else:
            vz = (altitude - D["prev_alt"]) / dt
        D["prev_alt"] = altitude

        if abs(alt_err) <= DEAD_BAND:
            pitch_raw_pre = Kd_att * pitch_deg
        else:
            if alt_err > 0:
                pitch_raw_pre = -(Kp_up * alt_err) + Kd_att * pitch_deg
            else:
                pitch_raw_pre = -(Kp_down * alt_err) + Kd_att * pitch_deg

        if pitch_raw_pre < -MAX_UP:
            pitch_raw = -MAX_UP; sat = "SAT_UP"
        elif pitch_raw_pre > MAX_DOWN:
            pitch_raw = MAX_DOWN; sat = "SAT_DOWN"
        else:
            pitch_raw = pitch_raw_pre; sat = "OK"

        _dbg_print(alt=f"{altitude:.0f}m", err=f"{alt_err:+.0f}m", vz=f"{vz:+.1f}m/s",
                   pitch_deg=f"{pitch_deg:+.1f}°", raw=f"{pitch_raw_pre:+.4f}",
                   cmd=f"{pitch_raw:+.4f}", limits=f"[{-MAX_UP:.3f},{MAX_DOWN:.3f}]",
                   sat=sat, dt=f"{dt:.2f}s", step=D['step'])

        roll_cmd = 0.0
        yaw_cmd = 0.0
        fire_cmd = 0.0
        return [float(pitch_raw), roll_cmd, yaw_cmd, fire_cmd]

    def fire_cmd(self, state):
        locked = state[7]
        fire_cmd = 1.0 if locked > 0 else 0.0
        return [0.0, 0.0, 0.0, fire_cmd]

    def enemys_track_cmd(self, state):
        # Enemy flavor of track (preserved numerics)
        dx, dy, dz = state[0] * 10000, state[2] * 10000, state[1] * 10000
        dx_norm, dy_norm, dz_norm = state[0], state[2], state[1]
        plane_heading = state[19]
        altitude = state[14] * 10000
        locked = state[7]
        target_angle = state[6]

        angle_to_enemy = np.degrees(np.arctan2(dx, dy))
        relative_bearing = (angle_to_enemy - plane_heading + 180) % 360 - 180

        if altitude < 1200:
            pitch_cmd = -0.3
        elif altitude > 8000:
            pitch_cmd = 0.3
        else:
            plane_pitch_norm = state[3]
            plane_pitch = math.degrees(plane_pitch_norm * math.pi) * (-1)
            horiz_dist = np.sqrt(dx**2 + dy**2)
            pitch_to_target = np.degrees(np.arctan2(dz, horiz_dist))
            relative_pitch = (pitch_to_target - plane_pitch + 90) % 180 - 90
            gain = np.interp(horiz_dist, [0, 800], [1.5, 1.1])
            if horiz_dist < 800:
                pitch_cmd = np.clip(-0.03 * relative_pitch * gain / 30, -1, 1)
            else:
                pitch_cmd = np.clip(dz_norm * -0.2, -1, 1)
            if abs(relative_pitch) > 0.5:
                pitch_cmd = -0.25 if relative_pitch > 0 else 0.25

        roll_cmd = 0.0
        yaw_gain = 0.03 if abs(relative_bearing) < 10 else 0.06
        yaw_cmd = np.clip(relative_bearing * yaw_gain, -1.0, 1.0)
        fire_cmd = 1.0 if locked > 0 else 0.0
        return [float(pitch_cmd), float(roll_cmd), float(yaw_cmd), float(fire_cmd)]