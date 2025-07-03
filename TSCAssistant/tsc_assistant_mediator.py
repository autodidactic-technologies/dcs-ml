import re
import numpy as np
from loguru import logger
from typing import Dict, Tuple, Optional
from collections import deque

from TSCAssistant.tsc_agent_prompt import render_prompt
from TSCAssistant.feature_translator import translate_features_for_llm
from TSCAssistant.mediator import Mediator
from minigrid.core.constants import OBJECT_TO_IDX, STATE_TO_IDX


class TSCAgentWithMediator:
    """
    Enhanced TSC Agent with learning-aware mediator integration and GRADUAL loop detection.
    """

    def __init__(self,
                 llm,
                 obs_shape: tuple = (7, 7, 3),
                 device: str = "cpu",
                 verbose: bool = True,
                 train_mediator: bool = True):
        self.llm = llm
        self.verbose = verbose
        self.train_mediator = train_mediator

        # Initialize mediator with learning-friendly parameters
        self.mediator = Mediator(
            obs_shape=obs_shape,
            device=device,
            verbose=verbose,
            learning_rate=1e-4
        )

        # Track LLM interactions
        self.current_plan = None
        self.interaction_count = 0
        self.override_count = 0

        # Performance tracking for learning
        self.recent_overrides = deque(maxlen=30)  # Increased window
        self.recent_successes = deque(maxlen=30)

        # GRADUAL loop detection at TSC level
        self.consecutive_same_llm_decision = 0
        self.last_llm_action = None
        self.last_rl_action = None
        self.same_situation_count = 0
        self.last_situation_hash = None

        # Learning-aware emergency circuit breaker
        self.emergency_rl_mode = 0
        self.llm_failure_count = 0
        self.learning_tolerance_multiplier = 2.0  # More tolerance during learning

        # Context tracking for better prompts
        self.interaction_history = deque(maxlen=10)
        self.performance_trend = deque(maxlen=20)

    def extract_features(self, obs: np.ndarray, info: dict) -> dict:
        """
        Enhanced feature extraction with learning context
        """
        # Get environment for carrying state
        env = info.get("llm_env", None)

        # Use coordinate system from updated version
        obj_map = np.fliplr(obs['image'][:, :, 0])
        state_map = np.fliplr(obs['image'][:, :, 2])

        # Agent carrying state
        has_key = bool(getattr(env.unwrapped, "carrying", None)) if env else False

        def find(idx):
            locs = np.argwhere(obj_map == idx)
            return tuple(int(x) for x in locs[0]) if len(locs) else None

        # Agent position
        agent_pos = (3, 0)

        # Find objects
        key_pos = find(OBJECT_TO_IDX["key"])
        door_pos = find(OBJECT_TO_IDX["door"])
        goal_pos = find(OBJECT_TO_IDX["goal"])

        # Door state
        door_state = None
        if door_pos:
            inv_state = {v: k for k, v in STATE_TO_IDX.items()}
            door_state = inv_state.get(int(state_map[door_pos]), "unknown")

        # Distance calculations
        def manh(p, q):
            return abs(p[0] - q[0]) + abs(p[1] - q[1]) if (p and q) else None

        dist_to_key = manh(agent_pos, key_pos)
        dist_to_door = manh(agent_pos, door_pos)
        dist_to_goal = manh(agent_pos, goal_pos)

        # Enhanced spatial features
        def get_vertical_distance(p, q):
            return abs(p[0] - q[0]) if (p and q) else None

        def get_horizontal_distance(p, q):
            return abs(p[1] - q[1]) if (p and q) else None

        vert_dist_to_goal = get_vertical_distance(agent_pos, goal_pos)
        horiz_dist_to_goal = get_horizontal_distance(agent_pos, goal_pos)
        vertical_distance_to_key = get_vertical_distance(agent_pos, key_pos)
        horizontal_distance_to_key = get_horizontal_distance(agent_pos, key_pos)

        # Relative directions
        def rel_dir_agent_frame(agent_pos, q_pos):
            if not (agent_pos and q_pos):
                return None
            dy, dx = q_pos[1] - agent_pos[1], q_pos[0] - agent_pos[0]

            if dx == 0:
                if dy < 0:
                    return "down"
                elif dy > 0:
                    return "up"
            elif dx < 0:
                return "left"
            elif dx > 0:
                return "right"

        rel_dir_to_key = rel_dir_agent_frame(agent_pos, key_pos)
        rel_dir_to_door = rel_dir_agent_frame(agent_pos, door_pos)

        # Environmental analysis
        other_mask = (obj_map > OBJECT_TO_IDX["empty"]) & (obj_map != OBJECT_TO_IDX["agent"]) & (
                obj_map != OBJECT_TO_IDX["wall"])
        other_locs = [tuple(loc) for loc in np.argwhere(other_mask)]
        dists = [manh(agent_pos, loc) for loc in other_locs]
        valid_dists = [d for d in dists if d is not None]
        dist_to_nearest_object = min(valid_dists) if valid_dists else None

        grid_size = obs['image'].shape[:2]
        num_visible_objects = int(other_mask.sum())

        # Object counts
        counts = {
            name: int((obj_map == idx).sum())
            for name, idx in OBJECT_TO_IDX.items()
            if name not in ("empty", "agent")
        }

        # Path analysis
        free_mask = (obj_map == OBJECT_TO_IDX["empty"])
        frees = 0
        if agent_pos:
            y, x = agent_pos
            for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid_size[0] and 0 <= nx < grid_size[1] and free_mask[ny, nx]:
                    frees += 1
        multiple_paths_open = frees >= 2

        # Enhanced object detection
        def object_in_front(agent_pos: tuple, obj_map: np.ndarray):
            try:
                yn = agent_pos[1] + 1
                xn = agent_pos[0]
                if 0 <= xn < obj_map.shape[0] and 0 <= yn < obj_map.shape[1]:
                    if obj_map[xn, yn] == OBJECT_TO_IDX["key"]:
                        return "key"
                    elif obj_map[xn, yn] == OBJECT_TO_IDX["door"]:
                        return "door"
                    elif obj_map[xn, yn] == OBJECT_TO_IDX["wall"]:
                        return "wall"
                    elif obj_map[xn, yn] == OBJECT_TO_IDX["goal"]:
                        return "goal"
                    else:
                        return "empty"
                return "out_of_bounds"
            except (IndexError, TypeError):
                return "error"

        front_object = object_in_front(agent_pos, obj_map)

        # Adjacency
        def is_adjacent(p, q):
            return (p and q and manh(p, q) == 1)

        is_adj_key = is_adjacent(agent_pos, key_pos)
        is_adj_door = is_adjacent(agent_pos, door_pos)

        # Facing detection
        def is_facing_object_agent_frame(agent_pos: tuple, obj_map: np.ndarray, obj_idx: int) -> bool:
            try:
                yn = agent_pos[1] + 1
                xn = agent_pos[0]

                if 0 <= xn < obj_map.shape[0] and 0 <= yn < obj_map.shape[1]:
                    return obj_map[xn, yn] == obj_idx
                return False
            except (IndexError, TypeError):
                return False

        is_facing_key = is_facing_object_agent_frame(agent_pos, obj_map, OBJECT_TO_IDX["key"])
        is_facing_door = is_facing_object_agent_frame(agent_pos, obj_map, OBJECT_TO_IDX["door"])
        is_facing_wall = is_facing_object_agent_frame(agent_pos, obj_map, OBJECT_TO_IDX["wall"])

        # Return enhanced feature dictionary
        return {
            # Original features
            "grid_size": grid_size,
            "agent_pos": agent_pos,
            "key_pos": key_pos,
            "door_pos": door_pos,
            "goal_pos": goal_pos,
            "door_state": door_state,
            "dist_to_key": dist_to_key,
            "dist_to_door": dist_to_door,
            "dist_to_goal": dist_to_goal,
            "is_key_visible": key_pos is not None,
            "is_door_visible": door_pos is not None,
            "is_adjacent_to_key": is_adj_key,
            "is_adjacent_to_door": is_adj_door,
            "facing_key": is_facing_key,
            "facing_wall": is_facing_wall,
            "facing_door": is_facing_door,

            # Enhanced features
            "has_key": has_key,
            "dist_to_nearest_object": dist_to_nearest_object,
            "num_visible_objects": num_visible_objects,
            "vertical_distance_to_goal": vert_dist_to_goal,
            "horizontal_distance_to_goal": horiz_dist_to_goal,
            "vertical_distance_to_key": vertical_distance_to_key,
            "horizontal_distance_to_key": horizontal_distance_to_key,
            "rel_dir_to_key": rel_dir_to_key,
            "rel_dir_to_door": rel_dir_to_door,
            "multiple_paths_open": multiple_paths_open,
            "front_object": front_object,
            **counts
        }

    def _get_situation_hash(self, features: Dict, rl_action: int) -> str:
        """Create a hash of the current situation for loop detection"""
        key_elements = [
            features.get('agent_pos'),
            features.get('key_pos'),
            features.get('door_pos'),
            features.get('has_key', False),
            rl_action
        ]
        return str(hash(tuple(str(x) for x in key_elements)))

    def _get_learning_phase_context(self) -> Dict:
        """Get context based on current learning phase"""
        phase = self.mediator.learning_phase

        if phase == "early_exploration":
            return {
                "learning_context": "Agent is in early learning phase. Provide clear, educational explanations for decisions.",
                "decision_guidance": "Focus on teaching correct actions and explaining why certain moves are good or bad.",
                "patience_note": "Be patient with repeated questions as the agent is learning basic concepts."
            }
        elif phase == "guided_learning":
            return {
                "learning_context": "Agent is developing decision-making skills. Provide reasoning but allow some exploration.",
                "decision_guidance": "Balance between correcting clear mistakes and allowing learning opportunities.",
                "patience_note": "Agent is becoming more capable but may still need guidance in complex situations."
            }
        else:  # autonomous
            return {
                "learning_context": "Agent is in autonomous mode. Only intervene when necessary.",
                "decision_guidance": "Focus on efficiency and only override clearly problematic actions.",
                "patience_note": "Agent should be making good decisions independently."
            }

    def agent_run(self,
                  sim_step: int,
                  obs: Dict,
                  rl_action: int,
                  infos: Dict,
                  reward: Optional[float] = None,
                  use_learned_asking: bool = True) -> Tuple[int, bool, Dict]:
        """Enhanced decision logic with LEARNING-AWARE loop detection."""

        info = infos if isinstance(infos, dict) else infos[0]

        # Extract enhanced features
        raw_feats = self.extract_features(obs, info)

        # Gradual situation-based loop detection
        current_situation = self._get_situation_hash(raw_feats, rl_action)
        if current_situation == self.last_situation_hash:
            self.same_situation_count += 1
        else:
            self.same_situation_count = 0
        self.last_situation_hash = current_situation

        # Learning-aware emergency circuit breaker
        emergency_threshold = int(20 * self.learning_tolerance_multiplier)
        if self.same_situation_count > emergency_threshold:
            if self.verbose:
                logger.warning(
                    f"🚨 LEARNING-AWARE EMERGENCY: Same situation {self.same_situation_count} times - forcing RL mode for {emergency_threshold // 2} steps")
            self.emergency_rl_mode = emergency_threshold // 2
            self.same_situation_count = 0

        # Check emergency RL mode
        if self.emergency_rl_mode > 0:
            self.emergency_rl_mode -= 1
            interaction_info = {
                'asked_llm': False,
                'ask_probability': 0.05,
                'llm_plan_changed': False,
                'interaction_count': self.interaction_count,
                'override_count': self.override_count,
                'emergency_mode': True,
                'learning_phase': self.mediator.learning_phase
            }
            if self.verbose and self.emergency_rl_mode % 10 == 0:
                logger.info(f"🚨 Emergency RL mode: {self.emergency_rl_mode} steps remaining")
            return rl_action, False, interaction_info

        # Learning-aware performance adjustment
        recent_performance = self._get_recent_performance()
        learning_phase = self.mediator.learning_phase

        # Adjust asking probability based on learning phase and performance
        if learning_phase == "early_exploration":
            asking_threshold_adjustment = -0.1  # More likely to ask during learning
        elif learning_phase == "guided_learning":
            if recent_performance < 0.3:
                asking_threshold_adjustment = 0.1  # Less likely if struggling
            else:
                asking_threshold_adjustment = 0.0
        else:  # autonomous
            if recent_performance < 0.3:
                asking_threshold_adjustment = 0.2
            elif recent_performance > 0.8:
                asking_threshold_adjustment = -0.1
            else:
                asking_threshold_adjustment = 0.0

        # Mediator decides whether to interrupt RL
        should_interrupt, interrupt_confidence = self.mediator.should_ask_llm(
            obs=obs,
            ppo_action=rl_action,
            use_learned_policy=use_learned_asking
        )

        # Apply learning-aware adjustment
        if asking_threshold_adjustment != 0.0:
            adjusted_threshold = 0.6 + asking_threshold_adjustment
            if should_interrupt and interrupt_confidence < adjusted_threshold:
                should_interrupt = False
                if self.verbose:
                    logger.info(
                        f"Learning-aware asking adjustment: {interrupt_confidence:.3f} < {adjusted_threshold:.3f}")

        interaction_info = {
            'asked_llm': should_interrupt,
            'ask_probability': interrupt_confidence,
            'llm_plan_changed': False,
            'interaction_count': self.interaction_count,
            'override_count': self.override_count,
            'recent_performance': recent_performance,
            'asking_adjustment': asking_threshold_adjustment,
            'emergency_mode': False,
            'same_situation_count': self.same_situation_count,
            'learning_phase': learning_phase
        }

        if should_interrupt:
            # LLM interrupts and provides guidance
            llm_action, plan_changed = self._query_llm_with_learning_context(raw_feats, rl_action, info)

            # Enhanced loop detection for LLM decisions
            if llm_action == self.last_llm_action and rl_action == self.last_rl_action:
                self.consecutive_same_llm_decision += 1

                # Learning-aware loop threshold
                loop_threshold = 12 if learning_phase == "early_exploration" else 8

                if self.consecutive_same_llm_decision > loop_threshold:
                    if self.verbose:
                        logger.warning(
                            f"🔄 LLM decision loop detected: {self.consecutive_same_llm_decision} times - using RL action")
                    llm_action = rl_action
                    plan_changed = False
                    self.llm_failure_count += 1
            else:
                self.consecutive_same_llm_decision = 0
                self.llm_failure_count = 0

            self.last_llm_action = llm_action
            self.last_rl_action = rl_action

            final_action = llm_action
            was_interrupted = True
            interaction_info['llm_plan_changed'] = plan_changed
            self.interaction_count += 1

            if plan_changed:
                self.override_count += 1
                self.recent_overrides.append(1)
            else:
                self.recent_overrides.append(0)

            # Track interaction for learning
            self.interaction_history.append({
                'rl_action': rl_action,
                'llm_action': llm_action,
                'changed': plan_changed,
                'learning_phase': learning_phase
            })

        else:
            # Use RL action directly
            final_action = rl_action
            was_interrupted = False
            self.consecutive_same_llm_decision = 0

        # Update mediator state
        self.mediator.update_state(obs)

        # Enhanced logging with learning context
        if self.verbose:
            self._log_decision_with_learning_context(sim_step, rl_action, final_action, should_interrupt,
                                                     interrupt_confidence, was_interrupted,
                                                     interaction_info['llm_plan_changed'], raw_feats, learning_phase)

        return final_action, was_interrupted, interaction_info

    def _query_llm_with_learning_context(self, features: Dict, ppo_action: int, info: Dict) -> Tuple[int, bool]:
        """Enhanced LLM querying with learning-aware context and prompts."""

        # Handle forbidden actions
        if ppo_action in [4, 6]:
            return self._handle_forbidden_action(features, ppo_action)

        try:
            # Create learning-aware context
            enhanced_context = self._create_learning_aware_context(features, ppo_action)
            learning_context = self._get_learning_phase_context()
            enhanced_context.update(learning_context)

            # Add loop detection context
            if self.consecutive_same_llm_decision > 3:
                enhanced_context[
                    "loop_warning"] = f"This override has been repeated {self.consecutive_same_llm_decision} times. Consider if RL action might be correct."

            if self.same_situation_count > 8:
                enhanced_context[
                    "situation_warning"] = f"Agent has been in similar situation for {self.same_situation_count} steps. Consider different approach."

            # Add performance context
            recent_performance = self._get_recent_performance()
            if recent_performance < 0.3:
                enhanced_context["performance_context"] = "Agent has been struggling recently. Provide clear guidance."
            elif recent_performance > 0.7:
                enhanced_context["performance_context"] = "Agent is performing well. Only intervene if necessary."

            # Translate to natural language
            translated_feats = translate_features_for_llm(features)
            translated_feats.update(enhanced_context)

            # Generate enhanced prompt
            prompt = render_prompt(
                env_name=info.get("env", "MiniGrid"),
                features=translated_feats,
                action=int(ppo_action)
            )

            # Get LLM response
            try:
                llm_response = self.llm.invoke(prompt)
                txt = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            except Exception as llm_error:
                logger.error(f"LLM invocation failed: {llm_error}")
                return self._fallback_action(features, ppo_action)

            # Parse action
            m = re.search(r"Selected\s*action\s*[:=]?\s*(\d)", txt, re.IGNORECASE)
            if not m:
                logger.error(f"No action found in LLM response: {txt[:200]}...")
                return self._fallback_action(features, ppo_action)

            llm_action = int(m.group(1))

            # Validate LLM action
            if not self._is_valid_action(llm_action, features):
                logger.warning(f"LLM chose invalid action {llm_action}, using fallback")
                return self._fallback_action(features, ppo_action)

            logger.info(f"🤖 LLM CHOSE: {llm_action} (was {ppo_action}) [Phase: {self.mediator.learning_phase}]")

            plan_changed = (llm_action != ppo_action)
            return llm_action, plan_changed

        except Exception as e:
            logger.error(f"LLM error: {e}, using fallback")
            self.llm_failure_count += 1
            return self._fallback_action(features, ppo_action)

    def _create_learning_aware_context(self, features: Dict, ppo_action: int) -> Dict:
        """Create enhanced context with learning awareness."""

        context = {}

        # Add decision confidence
        context["ppo_confidence"] = "high" if ppo_action in [0, 1, 2] else "low"

        # Add state assessment
        has_key = features.get('has_key', False)
        key_visible = features.get('is_key_visible', False)
        door_visible = features.get('is_door_visible', False)

        if has_key and door_visible:
            context["current_objective"] = "Find and unlock the door"
        elif key_visible and not has_key:
            context["current_objective"] = "Navigate to and pick up the key"
        elif not key_visible and not has_key:
            context["current_objective"] = "Explore to find the key"
        else:
            context["current_objective"] = "Continue current plan"

        # Add learning phase specific guidance
        learning_phase = self.mediator.learning_phase

        if learning_phase == "early_exploration":
            context["learning_guidance"] = "Agent is learning. Provide educational explanations for decisions."
            context["intervention_style"] = "Be more willing to correct and guide the agent's actions."
        elif learning_phase == "guided_learning":
            context["learning_guidance"] = "Agent is developing skills. Balance guidance with learning opportunities."
            context["intervention_style"] = "Correct clear mistakes but allow some exploration."
        else:  # autonomous
            context["learning_guidance"] = "Agent is autonomous. Only intervene when necessary."
            context["intervention_style"] = "Focus on efficiency and minimal necessary interventions."

        # Add performance context
        recent_perf = self._get_recent_performance()
        if recent_perf < 0.3:
            context["performance_note"] = "Agent struggling recently. Provide clear, helpful guidance."
        elif recent_perf > 0.8:
            context["performance_note"] = "Agent performing excellently. Current strategy is working well."
        else:
            context["performance_note"] = "Agent learning steadily. Make balanced decisions."

        # Add interaction history context
        if len(self.interaction_history) > 0:
            recent_interactions = list(self.interaction_history)[-3:]
            override_rate = sum(1 for i in recent_interactions if i['changed']) / len(recent_interactions)

            if override_rate > 0.8:
                context[
                    "interaction_context"] = "You've been overriding frequently. Consider if RL actions might be acceptable."
            elif override_rate < 0.2:
                context["interaction_context"] = "You've been agreeing with RL mostly. Continue being selective."
            else:
                context["interaction_context"] = "Good balance of agreeing and overriding RL actions."
        else:
            context["interaction_context"] = "Continue making balanced decisions."

        # Add loop prevention context
        if self.llm_failure_count > 2:
            context[
                "failure_warning"] = f"LLM has had {self.llm_failure_count} recent issues. Consider simpler, safer actions."
        else:
            context["failure_warning"] = ""

        return context

    def _handle_forbidden_action(self, features: Dict, ppo_action: int) -> Tuple[int, bool]:
        """Enhanced forbidden action handling with learning context."""

        logger.warning(f"PPO suggested forbidden action {ppo_action} [Phase: {self.mediator.learning_phase}]")

        # Get current state
        has_key = features.get('has_key', False)
        is_adjacent_to_key = features.get('is_adjacent_to_key', False)
        is_adjacent_to_door = features.get('is_adjacent_to_door', False)
        facing_key = features.get('facing_key', False)
        facing_door = features.get('facing_door', False)
        front_object = features.get('front_object', 'empty')
        rel_dir_to_key = features.get('rel_dir_to_key')

        # Context-aware override decisions
        if is_adjacent_to_key and facing_key and not has_key:
            logger.info("→ Perfect key pickup situation, using PICKUP (3)")
            return 3, True
        elif is_adjacent_to_door and facing_door and has_key:
            logger.info("→ Perfect door toggle situation, using TOGGLE (5)")
            return 5, True
        elif front_object == "key" and not has_key:
            logger.info("→ Key directly in front, moving FORWARD (2)")
            return 2, True
        elif front_object == "door" and has_key:
            logger.info("→ Door directly in front with key, moving FORWARD (2)")
            return 2, True
        elif rel_dir_to_key and not has_key:
            # Smart navigation based on relative direction
            if rel_dir_to_key == "left":
                logger.info("→ Key to the left, turning LEFT (0)")
                return 0, True
            elif rel_dir_to_key == "right":
                logger.info("→ Key to the right, turning RIGHT (1)")
                return 1, True
            else:
                logger.info("→ Key ahead, moving FORWARD (2)")
                return 2, True
        else:
            logger.info("→ Default exploration, turning LEFT (0)")
            return 0, True

    def _get_recent_performance(self) -> float:
        """Get recent performance metric."""
        if len(self.recent_successes) == 0:
            return 0.5  # Neutral
        return np.mean(self.recent_successes)

    def _log_decision_with_learning_context(self, sim_step: int, rl_action: int, final_action: int,
                                            should_interrupt: bool, interrupt_confidence: float,
                                            was_interrupted: bool, llm_changed_plan: bool, features: Dict,
                                            learning_phase: str):
        """Enhanced logging with learning context."""

        # Get current state for context
        has_key = features.get('has_key', False)
        key_pos = features.get('key_pos')
        door_pos = features.get('door_pos')
        agent_pos = features.get('agent_pos')

        state_str = f"Agent@{agent_pos}, Key@{key_pos}, Door@{door_pos}, HasKey={has_key}"
        phase_str = f"[{learning_phase.upper()}]"

        if should_interrupt:
            if llm_changed_plan:
                logger.info(f"[Step {sim_step}] {phase_str} 🛑 LLM OVERRIDE: "
                            f"RL {rl_action} → LLM {final_action} "
                            f"(conf={interrupt_confidence:.2f}) | {state_str}")
            else:
                logger.info(f"[Step {sim_step}] {phase_str} 🛑 LLM AGREED: "
                            f"RL {rl_action} confirmed "
                            f"(conf={interrupt_confidence:.2f}) | {state_str}")
        else:
            logger.info(f"[Step {sim_step}] {phase_str} ✅ RL CONTINUES: "
                        f"Action {final_action} "
                        f"(conf={interrupt_confidence:.2f}) | {state_str}")

    def _is_valid_action(self, action: int, features: Dict) -> bool:
        """Validate if LLM action makes sense in current context."""

        # Always forbid actions 4 and 6
        if action in [4, 6]:
            return False

        # Validate pickup action
        if action == 3:
            has_key = features.get('has_key', False)
            is_adjacent_to_key = features.get('is_adjacent_to_key', False)
            if has_key or not is_adjacent_to_key:
                return False

        # Validate toggle action
        if action == 5:
            has_key = features.get('has_key', False)
            is_adjacent_to_door = features.get('is_adjacent_to_door', False)
            if not has_key or not is_adjacent_to_door:
                return False

        return True

    def _fallback_action(self, features: Dict, original_action: int) -> Tuple[int, bool]:
        """Learning-aware fallback action selection."""

        # If we've been in fallback too many times, try original action
        if self.llm_failure_count > 5 and self._is_valid_action(original_action, features):
            logger.info("→ Multiple LLM failures, trying original RL action")
            return original_action, False

        # If original action was valid, use it
        if self._is_valid_action(original_action, features):
            return original_action, False

        # Choose safe action based on state and learning phase
        has_key = features.get('has_key', False)
        key_visible = features.get('is_key_visible', False)
        rel_dir_to_key = features.get('rel_dir_to_key')

        # Learning phase aware fallback
        if self.mediator.learning_phase == "early_exploration":
            # More conservative fallback during learning
            if not has_key and key_visible and rel_dir_to_key:
                if rel_dir_to_key == "left":
                    return 0, True  # Turn left
                elif rel_dir_to_key == "right":
                    return 1, True  # Turn right
                else:
                    return 2, True  # Move forward
            else:
                return 0, True  # Safe default: turn left
        else:
            # Standard fallback for advanced phases
            if not has_key and key_visible and rel_dir_to_key:
                if rel_dir_to_key == "left":
                    return 0, True
                elif rel_dir_to_key == "right":
                    return 1, True
                else:
                    return 2, True
            else:
                return 1, True  # Turn right for exploration

    def _heuristic_asking_decision(self, obs: Dict, ppo_action: int) -> Tuple[bool, float]:
        """Enhanced heuristic asking policy with learning awareness"""
        action = int(ppo_action) if hasattr(ppo_action, '__iter__') else ppo_action

        if self._is_critical_situation(obs, action):
            return True, 0.9

        if self._significant_obs_change(obs):
            return True, 0.7

        # Phase-aware periodic asking
        learning_phase = self.mediator.learning_phase
        if learning_phase == "early_exploration":
            max_steps = 25
        elif learning_phase == "guided_learning":
            max_steps = 20
        else:  # autonomous
            max_steps = 18

        if self.mediator.steps_since_last_ask >= max_steps:
            return True, 0.6

        return False, 0.3

    def _significant_obs_change(self, obs: Dict) -> bool:
        """Enhanced observation change detection"""
        if self.mediator.previous_obs is None:
            return True

        current_features = self.extract_features(obs, {"llm_env": None})
        previous_features = self.extract_features(self.mediator.previous_obs, {"llm_env": None})

        # Important feature changes
        important_changes = [
            current_features.get('is_key_visible') != previous_features.get('is_key_visible'),
            current_features.get('is_door_visible') != previous_features.get('is_door_visible'),
            current_features.get('is_adjacent_to_key') != previous_features.get('is_adjacent_to_key'),
            current_features.get('is_adjacent_to_door') != previous_features.get('is_adjacent_to_door'),
            abs(current_features.get('dist_to_key', 999) - previous_features.get('dist_to_key', 999)) > 2,
        ]

        return any(important_changes)

    def _is_critical_situation(self, obs: Dict, ppo_action: int) -> bool:
        """Detect critical situations that always need LLM with learning awareness"""
        action = int(ppo_action) if hasattr(ppo_action, '__iter__') else ppo_action

        # Don't override if we're in learning mode emergency
        if self.emergency_rl_mode > 0:
            return False

        # Check for problematic actions first
        if self._is_problematic_action(obs, action):
            return True

        # Extract features for situation analysis
        features = self.extract_features(obs, {"llm_env": None})

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
        learning_phase = self.mediator.learning_phase
        if learning_phase == "early_exploration":
            max_steps = 25
        elif learning_phase == "guided_learning":
            max_steps = 22
        else:  # autonomous
            max_steps = 18

        if (self.mediator.steps_since_last_ask > max_steps and
                self.mediator.position_stuck_count < 8):
            return True

        return False

    def _is_problematic_action(self, obs: Dict, ppo_action: int) -> bool:
        """Detect problematic actions that need LLM intervention with learning awareness"""
        action = int(ppo_action) if hasattr(ppo_action, '__iter__') else ppo_action

        # Invalid actions
        if action in [4, 6]:  # Drop, Done (forbidden)
            return True

        # Don't ask if we're in emergency mode (learning from natural RL)
        if self.emergency_rl_mode > 0:
            return False

        # Action loops (more lenient thresholds based on learning phase)
        learning_phase = self.mediator.learning_phase
        if learning_phase == "early_exploration":
            loop_threshold = 8  # More tolerant during learning
        elif learning_phase == "guided_learning":
            loop_threshold = 6
        else:  # autonomous
            loop_threshold = 5

        # Track recent actions (using mediator's deque)
        if len(self.mediator.recent_actions) >= loop_threshold:
            # Same action repeated many times
            recent_actions = list(self.mediator.recent_actions)[-loop_threshold:]
            if len(set(recent_actions)) == 1:
                return True

            # Oscillating between 2 actions
            if loop_threshold >= 6:
                last_6 = recent_actions[-6:]
                if (len(set(last_6)) == 2 and
                        last_6[0] == last_6[2] == last_6[4] and
                        last_6[1] == last_6[3] == last_6[5]):
                    return True

        return False

    def get_mediator_stats(self) -> Dict:
        """Enhanced mediator statistics with learning context."""
        base_stats = self.mediator.get_statistics()

        # Add performance metrics
        recent_override_rate = np.mean(self.recent_overrides) if self.recent_overrides else 0
        recent_performance = self._get_recent_performance()

        # Calculate learning progress metrics
        learning_progress = {
            'early_exploration': 0.33,
            'guided_learning': 0.66,
            'autonomous': 1.0
        }.get(self.mediator.learning_phase, 0.0)

        base_stats.update({
            'total_interactions': self.interaction_count,
            'total_overrides': self.override_count,
            'override_rate': self.override_count / max(self.interaction_count, 1),
            'recent_override_rate': recent_override_rate,
            'recent_performance': recent_performance,
            'interaction_efficiency': self.override_count / max(self.interaction_count, 1),
            'performance_trend': 'improving' if recent_performance > 0.6 else 'declining' if recent_performance < 0.4 else 'stable',
            'emergency_rl_mode': self.emergency_rl_mode,
            'consecutive_same_llm_decision': self.consecutive_same_llm_decision,
            'same_situation_count': self.same_situation_count,
            'llm_failure_count': self.llm_failure_count,
            'learning_progress': learning_progress,
            'learning_tolerance_multiplier': self.learning_tolerance_multiplier,
            'interaction_history_length': len(self.interaction_history)
        })
        return base_stats

    def update_performance(self, success: bool):
        """Update recent performance tracking with learning context."""
        self.recent_successes.append(1 if success else 0)
        self.performance_trend.append(success)

        # Adjust learning tolerance based on performance trend
        if len(self.performance_trend) >= 10:
            recent_success_rate = np.mean(list(self.performance_trend)[-10:])
            if recent_success_rate < 0.3:
                # Struggling - increase tolerance
                self.learning_tolerance_multiplier = min(3.0, self.learning_tolerance_multiplier * 1.1)
            elif recent_success_rate > 0.7:
                # Doing well - decrease tolerance gradually
                self.learning_tolerance_multiplier = max(1.0, self.learning_tolerance_multiplier * 0.95)

    def save_mediator(self, path: str):
        """Save the trained mediator."""
        self.mediator.save_asking_policy(path)

    def load_mediator(self, path: str):
        """Load a pre-trained mediator."""
        self.mediator.load_asking_policy(path)