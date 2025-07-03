import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from loguru import logger
from collections import deque
from minigrid.core.constants import OBJECT_TO_IDX, STATE_TO_IDX


class AskingPolicy(nn.Module):
    """Enhanced asking policy with better initialization and regularization"""

    def __init__(self, obs_shape: tuple, hidden_dim: int = 64):
        super().__init__()

        obs_size = np.prod(obs_shape)
        input_dim = obs_size * 3  # current, previous, difference

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Ask or Not Ask
        )

        # Better initialization - start more balanced
        with torch.no_grad():
            self.network[-1].bias[0] = -0.05  # Ask bias (slightly negative)
            self.network[-1].bias[1] = 0.05  # Not ask bias (slightly positive)

            # Xavier initialization for better gradient flow
            for layer in self.network:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, current_obs, previous_obs):
        current_flat = current_obs.flatten()
        previous_flat = previous_obs.flatten()
        diff_flat = current_flat - previous_flat
        combined_input = torch.cat([current_flat, previous_flat, diff_flat])
        return self.network(combined_input)


class Mediator:
    """
    Enhanced mediator with GRADUAL loop detection and learning-friendly interruption management.
    """

    def __init__(self,
                 obs_shape: tuple,
                 learning_rate: float = 1e-4,
                 device: str = "cpu",
                 verbose: bool = True):
        self.obs_shape = obs_shape
        self.device = device
        self.verbose = verbose

        # Initialize asking policy network
        self.asking_policy = AskingPolicy(obs_shape).to(device)
        self.optimizer = torch.optim.Adam(self.asking_policy.parameters(), lr=learning_rate)

        # State tracking attributes
        self.previous_obs = None
        self.last_llm_plan = None
        self.steps_since_last_ask = 0

        # GRADUAL loop detection (less aggressive)
        self.recent_actions = deque(maxlen=15)  # Increased from 10
        self.recent_positions = deque(maxlen=8)  # Increased from 5

        # Enhanced but gradual loop detection for LLM overrides
        self.recent_llm_overrides = deque(maxlen=25)  # Increased from 20
        self.recent_situations = deque(maxlen=15)  # Increased from 10
        self.override_count_same_situation = 0
        self.forced_rl_mode_steps = 0
        self.last_agent_pos = None
        self.position_stuck_count = 0

        # Learning phase tracking - ADDED
        self.learning_phase = "early_exploration"  # early_exploration -> guided_learning -> autonomous
        self.episodes_completed = 0

        # Training progress tracking attributes
        self.ask_history = []
        self.reward_history = []
        self.loss_history = []

        # More lenient penalty parameters for learning
        self.lambda_penalty = 0.01  # Start lower
        self.max_lambda = 0.15  # Lower max penalty
        self.agreement_penalty = 0.05  # Lower penalty to allow learning
        self.gamma = 0.99
        self.recent_loss = 0.0

        # Reward smoothing
        self.reward_buffer = deque(maxlen=100)
        self.baseline_reward = 0.0

        # Interrupt tracking with more tolerance
        self.recent_interrupts = deque(maxlen=50)
        self.recent_agreements = deque(maxlen=50)
        self.interrupt_efficiency_threshold = 0.2  # Lower threshold

        # Progressive learning parameters
        self.loop_detection_sensitivity = 0.3  # Start less sensitive
        self.emergency_threshold_multiplier = 2.0  # More lenient

    def should_ask_llm(self,
                       obs: Dict,
                       ppo_action: int,
                       use_learned_policy: bool = True,
                       force_exploration: bool = False) -> Tuple[bool, float]:
        """
        More gradual asking with LEARNING-AWARE loop detection
        """

        # Ensure action is integer
        action = int(ppo_action) if hasattr(ppo_action, '__iter__') else ppo_action

        # Gradually increase loop detection strictness
        if self.forced_rl_mode_steps > 0:
            self.forced_rl_mode_steps -= 1
            if self.verbose and self.forced_rl_mode_steps % 3 == 0:
                logger.info(f"🔒 Gradual RL mode: {self.forced_rl_mode_steps} steps remaining")
            return False, 0.1

        # GRADUAL loop detection based on learning phase
        if self._detect_gradual_loops(obs, action):
            return False, 0.1

        # PHASE-BASED asking strategy
        if self.learning_phase == "early_exploration":
            return self._early_exploration_asking(obs, action)
        elif self.learning_phase == "guided_learning":
            return self._guided_learning_asking(obs, action, use_learned_policy)
        else:  # autonomous phase
            return self._autonomous_asking(obs, action, use_learned_policy)

    def _detect_gradual_loops(self, obs: Dict, ppo_action: int) -> bool:
        """
        GRADUAL loop detection that becomes stricter as agent learns
        """
        # Track current situation
        current_situation = self._get_situation_signature(obs, ppo_action)
        self.recent_situations.append(current_situation)

        # Track agent position
        agent_pos = self._extract_features(obs).get('agent_pos')
        if agent_pos == self.last_agent_pos:
            self.position_stuck_count += 1
        else:
            self.position_stuck_count = 0
        self.last_agent_pos = agent_pos

        # Adjust thresholds based on learning phase
        if self.learning_phase == "early_exploration":
            # Very lenient during early learning
            same_situation_threshold = 8
            stuck_threshold = 20
            override_threshold = 10
        elif self.learning_phase == "guided_learning":
            # Moderate strictness
            same_situation_threshold = 6
            stuck_threshold = 15
            override_threshold = 8
        else:  # autonomous
            # Normal strictness
            same_situation_threshold = 4
            stuck_threshold = 12
            override_threshold = 6

        # DETECTION 1: Same situation repeated (gradual threshold)
        if len(self.recent_situations) >= same_situation_threshold:
            recent_situ = list(self.recent_situations)[-same_situation_threshold:]
            if len(set(recent_situ)) == 1:
                if self.verbose:
                    logger.warning(f"🔄 Gradual loop detected: Same situation {same_situation_threshold} times")
                self.forced_rl_mode_steps = max(3, same_situation_threshold // 2)  # Shorter forced mode
                return True

        # DETECTION 2: Position stuck with learning consideration
        if self.position_stuck_count >= stuck_threshold:
            recent_ask_rate = np.mean(self.ask_history[-10:]) if len(self.ask_history) >= 10 else 0
            if recent_ask_rate > 0.6:  # Only if asking too frequently
                if self.verbose:
                    logger.warning(f"🔄 Position stuck loop: {self.position_stuck_count} steps")
                self.forced_rl_mode_steps = stuck_threshold // 3
                return True

        # DETECTION 3: Override oscillation (more lenient)
        if len(self.recent_llm_overrides) >= override_threshold:
            pattern = list(self.recent_llm_overrides)[-override_threshold:]
            unique_overrides = len(set(pattern))
            if unique_overrides <= 2 and len(pattern) >= override_threshold:
                if self.verbose:
                    logger.warning(f"🔄 Override oscillation detected")
                self.forced_rl_mode_steps = override_threshold // 2
                return True

        return False

    def _early_exploration_asking(self, obs: Dict, action: int) -> Tuple[bool, float]:
        """
        Early exploration phase: Encourage learning with minimal penalties
        """
        # Always ask for clearly problematic actions
        if self._is_problematic_action(obs, action):
            return True, 0.95

        # Ask for critical situations to build understanding
        if self._is_critical_situation(obs, action):
            return True, 0.9

        # Increased exploration during early phase
        if self.steps_since_last_ask >= 12 and np.random.random() < 0.15:  # 15% chance
            return True, 0.7

        # Periodic asking to build experience
        if self.steps_since_last_ask >= 20:
            return True, 0.6

        return False, 0.3

    def _guided_learning_asking(self, obs: Dict, ppo_action: int, use_learned_policy: bool) -> Tuple[bool, float]:
        """
        Guided learning phase: Use learned policy with safety net
        """
        # Safety checks first
        if self._is_critical_situation(obs, ppo_action):
            return True, 0.9

        if not use_learned_policy:
            return self._heuristic_asking_decision(obs, ppo_action)

        # Use neural network with learning-friendly adjustments
        if self.previous_obs is None:
            return True, 1.0

        current_obs_tensor = self._obs_to_tensor(obs)
        previous_obs_tensor = self._obs_to_tensor(self.previous_obs)

        with torch.no_grad():
            logits = self.asking_policy(current_obs_tensor, previous_obs_tensor)
            probabilities = torch.softmax(logits, dim=0)
            ask_prob = probabilities[0].item()

            # Learning-friendly threshold adjustment
            base_threshold = 0.55  # Lower threshold for learning

            # Adjust based on recent performance (more forgiving)
            recent_efficiency = self._calculate_recent_efficiency()
            if recent_efficiency < 0.15:  # Very low efficiency
                threshold = 0.75
            elif recent_efficiency < 0.3:  # Low efficiency
                threshold = 0.65
            else:
                threshold = base_threshold

            should_ask = ask_prob > threshold

        self.steps_since_last_ask += 1
        if should_ask:
            self.steps_since_last_ask = 0

        return should_ask, ask_prob

    def _autonomous_asking(self, obs: Dict, ppo_action: int, use_learned_policy: bool) -> Tuple[bool, float]:
        """
        Autonomous phase: Full reliance on learned policy with efficiency considerations
        """
        # Safety checks
        if self._is_critical_situation(obs, ppo_action):
            return True, 0.9

        if not use_learned_policy:
            return self._heuristic_asking_decision(obs, ppo_action)

        # Standard neural network usage
        if self.previous_obs is None:
            return True, 1.0

        current_obs_tensor = self._obs_to_tensor(obs)
        previous_obs_tensor = self._obs_to_tensor(self.previous_obs)

        with torch.no_grad():
            logits = self.asking_policy(current_obs_tensor, previous_obs_tensor)
            probabilities = torch.softmax(logits, dim=0)
            ask_prob = probabilities[0].item()

            # Standard efficiency-based threshold
            recent_efficiency = self._calculate_recent_efficiency()
            if recent_efficiency < 0.2:
                threshold = 0.8
            elif recent_efficiency < 0.4:
                threshold = 0.7
            else:
                threshold = 0.6

            should_ask = ask_prob > threshold

        self.steps_since_last_ask += 1
        if should_ask:
            self.steps_since_last_ask = 0

        return should_ask, ask_prob

    def _calculate_recent_efficiency(self) -> float:
        """Calculate recent interrupt efficiency"""
        if len(self.recent_interrupts) < 5:
            return 0.5  # Neutral efficiency for insufficient data

        recent_interrupt_rate = np.mean(list(self.recent_interrupts)[-20:])
        recent_agreement_rate = np.mean(list(self.recent_agreements)[-20:])

        if recent_interrupt_rate < 0.01:
            return 1.0  # Perfect efficiency if not interrupting

        return 1 - (recent_agreement_rate / recent_interrupt_rate)

    def _is_problematic_action(self, obs: Dict, ppo_action: int) -> bool:
        """Detect problematic actions that need LLM intervention"""
        action = int(ppo_action) if hasattr(ppo_action, '__iter__') else ppo_action

        # Invalid actions
        if action in [4, 6]:  # Drop, Done (forbidden)
            return True

        # Don't ask if we're in forced mode (learning from natural RL)
        if self.forced_rl_mode_steps > 0:
            return False

        # Action loops (more lenient thresholds)
        self.recent_actions.append(action)
        if len(self.recent_actions) >= 8:  # Increased from 5
            # Same action repeated many times
            if len(set(list(self.recent_actions)[-6:])) == 1:
                return True

            # Oscillating between 2 actions (check longer pattern)
            last_6 = list(self.recent_actions)[-6:]
            if len(set(last_6)) == 2 and last_6[0] == last_6[2] == last_6[4] and last_6[1] == last_6[3] == last_6[5]:
                return True

        return False

    def _is_critical_situation(self, obs: Dict, ppo_action: int) -> bool:
        """Detect critical situations that always need LLM"""
        action = int(ppo_action) if hasattr(ppo_action, '__iter__') else ppo_action

        # Don't override if we're in learning mode
        if self.forced_rl_mode_steps > 0:
            return False

        # Check for problematic actions first
        if self._is_problematic_action(obs, action):
            return True

        # Extract features for situation analysis
        features = self._extract_features(obs)

        # Critical situations (but respect learning mode)
        if features.get('is_adjacent_to_key') and not features.get('has_key', False) and action != 3:
            return True

        if features.get('facing_wall') and action == 2:
            return True

        if features.get('is_adjacent_to_door') and features.get('has_key', False) and action != 5:
            return True

        # Check if trying to go through closed door
        if features.get('facing_door') and not features.get('door_is_open', False) and action == 2:
            return True

        # Been too long without asking (adjusted for learning phase)
        max_steps = 25 if self.learning_phase == "early_exploration" else 18
        if self.steps_since_last_ask > max_steps and self.position_stuck_count < 8:
            return True

        return False

    def _heuristic_asking_decision(self, obs: Dict, ppo_action: int) -> Tuple[bool, float]:
        """Enhanced heuristic asking policy"""
        action = int(ppo_action) if hasattr(ppo_action, '__iter__') else ppo_action

        if self._is_critical_situation(obs, action):
            return True, 0.9

        if self._significant_obs_change(obs):
            return True, 0.7

        # Phase-aware periodic asking
        max_steps = 25 if self.learning_phase == "early_exploration" else 18
        if self.steps_since_last_ask >= max_steps:
            return True, 0.6

        return False, 0.3

    def _significant_obs_change(self, obs: Dict) -> bool:
        """Enhanced observation change detection"""
        if self.previous_obs is None:
            return True

        current_features = self._extract_features(obs)
        previous_features = self._extract_features(self.previous_obs)

        # Important feature changes
        important_changes = [
            current_features.get('is_key_visible') != previous_features.get('is_key_visible'),
            current_features.get('is_door_visible') != previous_features.get('is_door_visible'),
            current_features.get('is_adjacent_to_key') != previous_features.get('is_adjacent_to_key'),
            current_features.get('is_adjacent_to_door') != previous_features.get('is_adjacent_to_door'),
            abs(current_features.get('dist_to_key', 999) - previous_features.get('dist_to_key', 999)) > 2,
        ]

        return any(important_changes)

    def train_asking_policy(self,
                            obs: Dict,
                            action: int,
                            reward: float,
                            next_obs: Dict,
                            asked_llm: bool,
                            llm_plan_changed: bool):
        """
        LEARNING-FRIENDLY training with gradual penalty increase
        """
        if self.previous_obs is None:
            self.previous_obs = obs
            return

        # Track LLM overrides for loop detection
        if asked_llm and llm_plan_changed:
            self.recent_llm_overrides.append(action)

        # Compute reward with learning-phase-aware penalties
        mediator_reward = self._compute_learning_aware_reward(
            task_reward=reward,
            asked_llm=asked_llm,
            llm_plan_changed=llm_plan_changed
        )

        current_obs_tensor = self._obs_to_tensor(obs)
        previous_obs_tensor = self._obs_to_tensor(self.previous_obs)

        # Get asking policy prediction
        logits = self.asking_policy(current_obs_tensor, previous_obs_tensor)
        probabilities = torch.softmax(logits, dim=0)
        ask_prob = probabilities[0]
        not_ask_prob = probabilities[1]

        # Policy gradient loss (REINFORCE)
        if asked_llm:
            policy_loss = -torch.log(ask_prob + 1e-8) * mediator_reward
        else:
            policy_loss = -torch.log(not_ask_prob + 1e-8) * mediator_reward

        # Add entropy bonus (higher during learning)
        entropy_bonus_weight = 0.03 if self.learning_phase == "early_exploration" else 0.02
        entropy = -(ask_prob * torch.log(ask_prob + 1e-8) +
                    not_ask_prob * torch.log(not_ask_prob + 1e-8))
        entropy_bonus = entropy_bonus_weight * entropy

        # L2 regularization
        l2_reg = 0.001 * sum(p.pow(2.0).sum() for p in self.asking_policy.parameters())

        total_loss = policy_loss - entropy_bonus + l2_reg

        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()

        # Learning-phase-aware gradient clipping
        grad_clip = 0.3 if self.learning_phase == "early_exploration" else 0.2
        torch.nn.utils.clip_grad_norm_(self.asking_policy.parameters(), grad_clip)
        self.optimizer.step()

        # Track statistics
        self.ask_history.append(asked_llm)
        self.reward_history.append(mediator_reward)
        self.loss_history.append(total_loss.item())
        self.recent_loss = total_loss.item()

        # Update learning phase
        self._update_learning_phase()

        # Update penalties gradually
        self._update_gradual_penalties()

        # Progress logging
        if self.verbose and len(self.ask_history) % 50 == 0:
            recent_ask_rate = np.mean(self.ask_history[-30:])
            recent_reward = np.mean(self.reward_history[-30:])
            recent_loss = np.mean(self.loss_history[-10:]) if len(self.loss_history) >= 10 else 0

            logger.info(f"Mediator Stats: Ask Rate={recent_ask_rate:.3f}, "
                        f"Avg Reward={recent_reward:.3f}, Loss={recent_loss:.3f}, "
                        f"Phase={self.learning_phase}, λ={self.lambda_penalty:.3f}, "
                        f"Forced_RL={self.forced_rl_mode_steps}, Stuck={self.position_stuck_count}")

    def _compute_learning_aware_reward(self, task_reward: float, asked_llm: bool, llm_plan_changed: bool) -> float:
        """
        Learning-phase-aware reward computation with gradual penalty increase
        """
        # Update baseline
        self.reward_buffer.append(task_reward)
        if len(self.reward_buffer) > 10:
            self.baseline_reward = np.mean(list(self.reward_buffer)[-50:])

        # Base reward (advantage)
        advantage = task_reward - self.baseline_reward
        base_reward = advantage

        # Track interrupt patterns
        self.recent_interrupts.append(asked_llm)
        self.recent_agreements.append(asked_llm and not llm_plan_changed)

        # Phase-aware penalty scaling
        if self.learning_phase == "early_exploration":
            penalty_scale = 0.5  # Reduced penalties
            bonus_scale = 1.2  # Increased bonuses
        elif self.learning_phase == "guided_learning":
            penalty_scale = 0.75
            bonus_scale = 1.0
        else:  # autonomous
            penalty_scale = 1.0
            bonus_scale = 0.8

        penalty = 0.0
        bonus = 0.0

        if asked_llm:
            if llm_plan_changed:
                # Good intervention
                if task_reward > 0:
                    bonus = 0.1 * bonus_scale
                elif task_reward <= 0:
                    penalty = 0.03 * penalty_scale  # Reduced penalty
            else:
                # Asked but agreed - scale penalty based on learning phase
                base_penalty = self.agreement_penalty * penalty_scale

                # Less aggressive escalation during learning
                recent_agreement_rate = np.mean(list(self.recent_agreements)[-20:]) if len(
                    self.recent_agreements) >= 20 else 0
                if recent_agreement_rate > 0.4:  # More lenient threshold
                    escalation_factor = 1.5 + (recent_agreement_rate - 0.4) * 2  # Reduced escalation
                    penalty = base_penalty * escalation_factor
                else:
                    penalty = base_penalty

                # Cap penalty to prevent learning disruption
                max_penalty = 0.3 if self.learning_phase == "early_exploration" else 0.5
                penalty = min(penalty, max_penalty)

        else:
            # Didn't ask
            if task_reward <= -0.1:
                # Should have asked (but more lenient during learning)
                if self.forced_rl_mode_steps == 0:
                    penalty = 0.015 * penalty_scale  # Reduced penalty
            elif task_reward > 0.5:
                # Good: didn't interrupt successful action
                bonus = 0.02 * bonus_scale

        total_reward = base_reward + bonus - penalty

        # Learning-phase-aware clipping
        if self.learning_phase == "early_exploration":
            total_reward = np.clip(total_reward, -0.5, 0.8)  # Less harsh negative clipping
        else:
            total_reward = np.clip(total_reward, -0.8, 0.8)

        return total_reward

    def _update_learning_phase(self):
        """Update learning phase based on progress"""
        num_episodes = len(self.ask_history) // 30  # Approximate episodes

        if num_episodes < 10:  # First 10 episodes
            self.learning_phase = "early_exploration"
        elif num_episodes < 25:  # Next 15 episodes
            self.learning_phase = "guided_learning"
        else:
            self.learning_phase = "autonomous"

    def _update_gradual_penalties(self):
        """Gradually increase penalties as agent learns"""
        progress = min(len(self.ask_history) / 1500.0, 1.0)  # Slower progression

        # Gradual penalty increase
        if self.learning_phase == "early_exploration":
            self.lambda_penalty = 0.005 + progress * 0.01  # Very low start
            self.agreement_penalty = 0.02 + progress * 0.03
        elif self.learning_phase == "guided_learning":
            self.lambda_penalty = 0.01 + progress * 0.05
            self.agreement_penalty = 0.05 + progress * 0.05
        else:  # autonomous
            self.lambda_penalty = 0.02 + progress * 0.08
            self.agreement_penalty = 0.1 + progress * 0.1

        # Cap penalties
        self.lambda_penalty = min(self.lambda_penalty, self.max_lambda)
        self.agreement_penalty = min(self.agreement_penalty, 0.15)

    def _get_situation_signature(self, obs: Dict, ppo_action: int) -> str:
        """Create a signature for the current situation to detect loops"""
        features = self._extract_features(obs)

        # Create a compact situation signature
        signature = f"pos:{features.get('agent_pos')}_key:{features.get('key_pos')}_door:{features.get('door_pos')}_haskey:{features.get('has_key', False)}_action:{ppo_action}"

        return signature

    def _obs_to_tensor(self, obs: Dict) -> torch.Tensor:
        """Convert observation dictionary to tensor."""
        image = obs['image']
        return torch.FloatTensor(image).to(self.device)

    def _extract_features(self, obs: Dict) -> Dict:
        """Extract features from observation for decision making."""
        obj_map = obs['image'][:, :, 0]
        state_map = obs['image'][:, :, 2]

        def find(idx):
            locs = np.argwhere(obj_map == idx)
            return tuple(int(x) for x in locs[0]) if len(locs) else None

        # In MiniGrid partial observation, agent is ALWAYS at center
        height, width = obj_map.shape
        agent_pos = (height // 2, width // 2)

        key_pos = find(OBJECT_TO_IDX["key"])
        door_pos = find(OBJECT_TO_IDX["door"])
        goal_pos = find(OBJECT_TO_IDX["goal"])

        # Door state analysis
        door_state = None
        door_is_open = False
        if door_pos:
            inv_state = {v: k for k, v in STATE_TO_IDX.items()}
            door_state_val = int(state_map[door_pos])
            door_state = inv_state.get(door_state_val, "unknown")
            door_is_open = (door_state == "open")

        def manh(p, q):
            return abs(p[0] - q[0]) + abs(p[1] - q[1]) if (p and q) else float('inf')

        def is_adjacent(p, q):
            return (p and q and manh(p, q) == 1)

        def is_facing_object(agent_pos, obj_pos):
            if not (agent_pos and obj_pos):
                return False
            return (obj_pos[0] == agent_pos[0] - 1 and obj_pos[1] == agent_pos[1])

        # Track position for loop detection
        self.recent_positions.append(agent_pos)

        features = {
            'agent_pos': agent_pos,
            'key_pos': key_pos,
            'door_pos': door_pos,
            'goal_pos': goal_pos,
            'door_state': door_state,
            'door_is_open': door_is_open,
            'dist_to_key': manh(agent_pos, key_pos),
            'dist_to_door': manh(agent_pos, door_pos),
            'dist_to_goal': manh(agent_pos, goal_pos),
            'is_key_visible': key_pos is not None,
            'is_door_visible': door_pos is not None,
            'is_adjacent_to_key': is_adjacent(agent_pos, key_pos),
            'is_adjacent_to_door': is_adjacent(agent_pos, door_pos),
            'facing_key': is_facing_object(agent_pos, key_pos),
            'facing_door': is_facing_object(agent_pos, door_pos),
            'facing_wall': False,
            'has_key': False,  # This will be overridden by TSC agent
        }

        return features

    def update_state(self, obs: Dict):
        """Update mediator's internal state."""
        self.previous_obs = obs.copy() if obs else None

    def get_statistics(self) -> Dict:
        """Enhanced statistics with learning phase and efficiency metrics"""
        if not self.ask_history:
            return {
                'total_steps': 0,
                'ask_rate': 0.0,
                'avg_reward': 0.0,
                'recent_ask_rate': 0.0,
                'recent_avg_reward': 0.0,
                'recent_loss': 0.0,
                'learning_phase': self.learning_phase,
                'lambda_penalty': self.lambda_penalty,
                'agreement_penalty': self.agreement_penalty,
                'baseline_reward': self.baseline_reward,
                'interrupt_efficiency': 1.0,
                'recent_interrupt_rate': 0.0,
                'recent_agreement_rate': 0.0,
                'efficiency_threshold': self.interrupt_efficiency_threshold,
                'successful_episodes': 0,
                'forced_rl_mode_steps': self.forced_rl_mode_steps,
                'position_stuck_count': self.position_stuck_count,
                'recent_override_count': len(self.recent_llm_overrides)
            }

        # Calculate efficiency metrics
        recent_interrupt_rate = np.mean(list(self.recent_interrupts)[-50:]) if len(
            self.recent_interrupts) >= 50 else np.mean(list(self.recent_interrupts)) if self.recent_interrupts else 0
        recent_agreement_rate = np.mean(list(self.recent_agreements)[-50:]) if len(
            self.recent_agreements) >= 50 else np.mean(list(self.recent_agreements)) if self.recent_agreements else 0

        efficiency = self._calculate_recent_efficiency()

        return {
            'total_steps': len(self.ask_history),
            'ask_rate': np.mean(self.ask_history),
            'avg_reward': np.mean(self.reward_history),
            'recent_ask_rate': np.mean(self.ask_history[-100:]) if len(self.ask_history) >= 100 else np.mean(
                self.ask_history),
            'recent_avg_reward': np.mean(self.reward_history[-100:]) if len(self.reward_history) >= 100 else np.mean(
                self.reward_history),
            'recent_loss': np.mean(self.loss_history[-10:]) if len(self.loss_history) >= 10 else 0,
            'learning_phase': self.learning_phase,
            'lambda_penalty': self.lambda_penalty,
            'agreement_penalty': self.agreement_penalty,
            'baseline_reward': self.baseline_reward,
            'interrupt_efficiency': efficiency,
            'recent_interrupt_rate': recent_interrupt_rate,
            'recent_agreement_rate': recent_agreement_rate,
            'efficiency_threshold': self.interrupt_efficiency_threshold,
            'successful_episodes': sum(1 for r in self.reward_history[-100:] if r > 0) if len(
                self.reward_history) >= 100 else sum(1 for r in self.reward_history if r > 0),
            'forced_rl_mode_steps': self.forced_rl_mode_steps,
            'position_stuck_count': self.position_stuck_count,
            'recent_override_count': len(self.recent_llm_overrides),
            'loop_detection_active': self.forced_rl_mode_steps > 0
        }

    def save_asking_policy(self, path: str):
        """Save the learned asking policy."""
        torch.save({
            'model_state_dict': self.asking_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ask_history': self.ask_history,
            'reward_history': self.reward_history,
            'loss_history': self.loss_history,
            'learning_phase': self.learning_phase,
            'lambda_penalty': self.lambda_penalty,
            'agreement_penalty': self.agreement_penalty,
            'baseline_reward': self.baseline_reward,
            'recent_interrupts': list(self.recent_interrupts),
            'recent_agreements': list(self.recent_agreements),
            'recent_llm_overrides': list(self.recent_llm_overrides),
            'forced_rl_mode_steps': self.forced_rl_mode_steps,
            'position_stuck_count': self.position_stuck_count,
        }, path)

    def load_asking_policy(self, path: str):
        """Load a pre-trained asking policy."""
        checkpoint = torch.load(path, map_location=self.device)
        self.asking_policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.ask_history = checkpoint.get('ask_history', [])
        self.reward_history = checkpoint.get('reward_history', [])
        self.loss_history = checkpoint.get('loss_history', [])
        self.learning_phase = checkpoint.get('learning_phase', 'early_exploration')
        self.lambda_penalty = checkpoint.get('lambda_penalty', 0.01)
        self.agreement_penalty = checkpoint.get('agreement_penalty', 0.05)
        self.baseline_reward = checkpoint.get('baseline_reward', 0.0)
        self.recent_interrupts = deque(checkpoint.get('recent_interrupts', []), maxlen=50)
        self.recent_agreements = deque(checkpoint.get('recent_agreements', []), maxlen=50)
        self.recent_llm_overrides = deque(checkpoint.get('recent_llm_overrides', []), maxlen=25)
        self.forced_rl_mode_steps = checkpoint.get('forced_rl_mode_steps', 0)
        self.position_stuck_count = checkpoint.get('position_stuck_count', 0)