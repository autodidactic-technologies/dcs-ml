from combat_mission_env import CombatMissionEnv
import random
import time
import numpy as np
import pygame


class RuleBasedAgent:
    def __init__(self, env):
        self.env = env
        self.last_radar_detection = None
        self.last_action_time = {}
        self.cooldowns = {
            "radar_toggle": 20,  # frames between radar toggles
            "mode_toggle": 40,  # frames between mode toggles
            "maneuver": 15  # frames between maneuver changes
        }
        self.current_time = 0
        self.search_pattern_index = 0
        self.search_patterns = [6, 8, 7, 9]  # North, East, South, West
        self.last_known_enemy_pos = None
        self.engagement_phase = "search"  # search, approach, engage, evade
        self.engagement_timer = 0
        self.optimal_distance = {
            1: 150,  # aircraft - optimal engagement distance
            2: 120  # ground unit - optimal engagement distance
        }
        self.missile_cooldown = 0
        self.missile_cooldown_time = 10  # frames between missile launches
        self.last_action = None
        self.consecutive_same_actions = 0
        self.max_consecutive_actions = 5  # Prevent getting stuck in loops

    def act(self, obs):
        self.current_time += 1
        action = None

        # Extract state information
        radar_on = obs['radar_on']
        radar_sweep_mode = obs['radar_sweep_mode']
        enemy_detected = obs['enemy_detected']
        enemy_type = obs['enemy_type']
        active_radar_missiles = obs['active_radar_missiles']
        ir_guided_missiles = obs['ir_guided_missiles']
        distance = obs['distance'][0] if obs['distance'].size > 0 else float('inf')
        autopilot_on = obs['autopilot_on']

        # Update missile cooldown
        if self.missile_cooldown > 0:
            self.missile_cooldown -= 1

        # Turn on radar if it's off - highest priority
        if radar_on == 0:
            action = 2  # Toggle radar on

        # If enemy is detected, update tracking and engagement phase
        elif enemy_detected == 1:
            self.last_radar_detection = self.current_time
            self.last_known_enemy_pos = True  # We know enemy is around

            # For ground units, prefer IR missiles
            if enemy_type == 2:  # Ground unit
                if self.missile_cooldown == 0 and ir_guided_missiles > 0:
                    self.missile_cooldown = self.missile_cooldown_time
                    action = 1  # Fire IR missile
                # If no IR missiles left but have active radar missiles
                elif self.missile_cooldown == 0 and active_radar_missiles > 0:
                    self.missile_cooldown = self.missile_cooldown_time
                    action = 0  # Fire active radar missile as backup
                else:
                    # Move to optimal firing position if not firing
                    if autopilot_on == 1:
                        action = 4  # Turn off autopilot
                    elif distance > self.optimal_distance[enemy_type] * 1.2:
                        action = 5  # Move toward enemy
                    elif distance < self.optimal_distance[enemy_type] * 0.8:
                        action = 10  # Move away from enemy
                    else:
                        # In optimal range, hold position
                        pass

            # For air targets, prefer active radar at longer ranges, IR at closer ranges
            elif enemy_type == 1:  # Air target
                if self.missile_cooldown == 0:
                    if distance > self.optimal_distance[enemy_type] and active_radar_missiles > 0:
                        self.missile_cooldown = self.missile_cooldown_time
                        action = 0  # Fire active radar missile at longer range
                    elif ir_guided_missiles > 0:
                        self.missile_cooldown = self.missile_cooldown_time
                        action = 1  # Fire IR missile at closer range
                    elif active_radar_missiles > 0:  # Backup if no IR missiles
                        self.missile_cooldown = self.missile_cooldown_time
                        action = 0
                else:
                    # Move to optimal firing position if not firing
                    if autopilot_on == 1:
                        action = 4  # Turn off autopilot
                    elif distance > self.optimal_distance[enemy_type] * 1.2:
                        action = 5  # Move toward enemy
                    elif distance < self.optimal_distance[enemy_type] * 0.8:
                        action = 10  # Move away from enemy

        # Lost contact with enemy - search or pursue based on last detection
        elif self.last_radar_detection is not None:
            time_since_detection = self.current_time - self.last_radar_detection

            # Recently lost contact - move to last known position
            if time_since_detection < 30:
                if autopilot_on == 1:
                    action = 4  # Toggle autopilot off
                else:
                    action = 5  # Move toward last known position

            # Longer since last detection - switch to search mode
            else:
                self.engagement_phase = "search"
                # Ensure radar is in sweep mode for better search
                if radar_sweep_mode == 0 and self._check_cooldown("mode_toggle"):
                    action = 3  # Toggle to sweep mode
                # Turn off autopilot and execute search pattern
                elif autopilot_on == 1:
                    action = 4  # Toggle autopilot off
                elif self._check_cooldown("maneuver"):
                    action = self.search_patterns[self.search_pattern_index]
                    self.search_pattern_index = (self.search_pattern_index + 1) % len(self.search_patterns)

        # No detection yet - initial search behavior
        else:
            # Start with radar sweep mode for initial detection
            if radar_sweep_mode == 0 and self._check_cooldown("mode_toggle"):
                action = 3  # Toggle to sweep mode for initial search
            # Turn off autopilot for manual search pattern
            elif autopilot_on == 1:
                action = 4  # Toggle autopilot off
            # Execute search pattern
            elif self._check_cooldown("maneuver"):
                action = self.search_patterns[self.search_pattern_index]
                self.search_pattern_index = (self.search_pattern_index + 1) % len(self.search_patterns)

        # If no specific action was selected, default to a smart action
        if action is None:
            # If we're not doing anything specific, use No-op
            action = 11  # No-op

        # Prevent getting stuck in the same action repeatedly
        if action == self.last_action:
            self.consecutive_same_actions += 1
            if self.consecutive_same_actions > self.max_consecutive_actions:
                # Break the pattern by choosing a different action
                if action in [5, 6, 7, 8, 9, 10]:  # If stuck in a movement pattern
                    # Choose a different movement
                    action = random.choice([a for a in [6, 7, 8, 9] if a != action])
                    self.consecutive_same_actions = 0
        else:
            self.consecutive_same_actions = 0

        self.last_action = action
        return action

    def _check_cooldown(self, action_type):
        """Check if an action is off cooldown and if so, put it on cooldown"""
        if action_type not in self.last_action_time:
            self.last_action_time[action_type] = 0

        if self.current_time - self.last_action_time[action_type] >= self.cooldowns[action_type]:
            self.last_action_time[action_type] = self.current_time
            return True
        return False


# Global environment variable
global_env = None


def run_rule_based_simulation(episodes=5, steps_per_episode=300):
    global global_env
    global_env = CombatMissionEnv()
    env = global_env
    agent = RuleBasedAgent(env)

    total_rewards = []

    try:
        for episode in range(episodes):
            obs = env.reset()
            agent.current_time = 0
            agent.last_radar_detection = None
            agent.consecutive_same_actions = 0
            agent.last_action = None

            episode_reward = 0

            for step in range(steps_per_episode):
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
                env.render()

                episode_reward += reward

                # Convert action to readable string for display
                action_names = {
                    0: "Fire Active Radar", 1: "Fire IR Guided", 2: "Toggle Radar",
                    3: "Toggle Sweep Mode", 4: "Toggle Autopilot", 5: "Toward Enemy",
                    6: "North", 7: "South", 8: "East", 9: "West",
                    10: "Away from Enemy", 11: "No-op"
                }

                print(f"Step {step}: Action: {action_names[action]}, Reward: {reward:.2f}")

                # Add a small delay for better visualization
                pygame.time.delay(50)

                if done:
                    print(f"Episode {episode + 1} finished after {step + 1} steps with reward {episode_reward:.2f}")
                    break

            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1} total reward: {episode_reward:.2f}")
            print("-------------------------------")
    finally:
        # Do not close the environment here, it's managed by the main module
        pass

    if total_rewards:
        print(f"Average reward over {len(total_rewards)} episodes: {np.mean(total_rewards):.2f}")


if __name__ == '__main__':
    try:
        run_rule_based_simulation()
    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        # Cleanup
        if global_env is not None:
            try:
                global_env.close()
            except:
                pass
        if pygame.get_init():
            pygame.quit()
            # Choose a different movement
            action = random.choice([a for a in [6, 7, 8, 9] if a != action])
            self.consecutive_same_actions = 0


def _check_cooldown(self, action_type):
    """Check if an action is off cooldown and if so, put it on cooldown"""
    if action_type not in self.last_action_time:
        self.last_action_time[action_type] = 0

    if self.current_time - self.last_action_time[action_type] >= self.cooldowns[action_type]:
        self.last_action_time[action_type] = self.current_time
        return True
    return False


def run_rule_based_simulation(episodes=5, steps_per_episode=300):
    env = CombatMissionEnv()
    agent = RuleBasedAgent(env)

    total_rewards = []

    try:
        for episode in range(episodes):
            obs = env.reset()
            agent.current_time = 0
            agent.last_radar_detection = None
            agent.consecutive_same_actions = 0
            agent.last_action = None

            episode_reward = 0

            for step in range(steps_per_episode):
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
                env.render()

                episode_reward += reward

                # Convert action to readable string for display
                action_names = {
                    0: "Fire Active Radar", 1: "Fire IR Guided", 2: "Toggle Radar",
                    3: "Toggle Sweep Mode", 4: "Toggle Autopilot", 5: "Toward Enemy",
                    6: "North", 7: "South", 8: "East", 9: "West",
                    10: "Away from Enemy", 11: "No-op"
                }

                print(f"Step {step}: Action: {action_names[action]}, Reward: {reward:.2f}")

                # Add a small delay for better visualization
                pygame.time.delay(50)

                if done:
                    print(f"Episode {episode + 1} finished after {step + 1} steps with reward {episode_reward:.2f}")
                    break

            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1} total reward: {episode_reward:.2f}")
            print("-------------------------------")
    finally:
        env.close()

    if total_rewards:
        print(f"Average reward over {len(total_rewards)} episodes: {np.mean(total_rewards):.2f}")


if __name__ == '__main__':
    run_rule_based_simulation()
