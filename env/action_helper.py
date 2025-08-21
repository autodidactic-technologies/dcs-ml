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
from hirl.environments.constants import *  # NormStates etc.

MAX_TRACKED_MISSILES = 4
MISSILE_PACK_LEN = 5  # [present, mx, my, mz, heading]

# ----------------------------- Primitive controllers --------------------------- #
class ActionHelper:
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
        """
        Always climbs by default.
        But if:
        - altitude > 7000m → descend (nose down)
        - pitch attitude is already too steep → counter-pitch to flatten
        """
        altitude = float(state[14] * 10000.0)  # meters
        pitch_att = float(state[21] * 180.0)  # degrees (-90 to +90)

        print(f"ALT: {altitude:.0f} m | PITCH: {pitch_att:.1f}°")

        # Default: climb
        pitch_cmd = -0.1  # nose up

        # Condition 1: too high → descend
        if altitude > 7000:
            pitch_cmd = 0.25  # nose down

        # Condition 2: over-rotated → flatten
        if pitch_att > 75:
            pitch_cmd = 0.2  # nose down to flatten
        elif pitch_att < -75:
            pitch_cmd = -0.2  # nose up to flatten (too nose-down)

        # Final safety clamp
        MAX_PITCH = 0.32
        pitch_cmd = max(min(pitch_cmd, MAX_PITCH), -MAX_PITCH)

        roll_cmd = 0.0
        yaw_cmd = 0.0
        fire_cmd = 0.0
        return [pitch_cmd, roll_cmd, yaw_cmd, fire_cmd]


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
        return [float(pitch_cmd), float(roll_cmd), float(yaw_cmd), float(0)]