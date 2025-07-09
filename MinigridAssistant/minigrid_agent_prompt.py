LEARNING_AWARE_PROMPT_TEMPLATE = """You are an expert controller for a reinforcement learning agent in MiniGrid-DoorKey-6x6.

LEARNING CONTEXT:
{learning_context}
{decision_guidance}
{patience_note}

CURRENT SITUATION:
Agent Position: {agent_pos} 
Key Position: {key_pos} | Distance: {dist_to_key} | Visible: {is_key_visible}
Door Position: {door_pos} | State: {door_state} | Distance: {dist_to_door} | Visible: {is_door_visible}
Goal Position: {goal_pos} | Distance: {dist_to_goal}

AGENT STATUS:
- Carrying key: {has_key}
- Current objective: {current_objective}
- Learning phase: {learning_guidance}

SPATIAL CONTEXT:
- Key direction: {rel_dir_to_key}
- Door direction: {rel_dir_to_door}
- Object in front: {front_object}
- Facing wall: {facing_wall}

PERFORMANCE CONTEXT:
{performance_note}
{performance_context}

INTERACTION CONTEXT:
{interaction_context}

PPO AGENT SUGGESTS: Action {action} (Confidence: {ppo_confidence})
INTERVENTION STYLE: {intervention_style}

AVAILABLE ACTIONS:
0: Turn left    1: Turn right    2: Move forward    3: Pick up key    5: Toggle door
(Actions 4 and 6 are FORBIDDEN - never use them)

CRITICAL DECISION RULES (MUST FOLLOW EXACTLY):
1. PICKUP RULES:
   - Use Action 3 ONLY when: adjacent_to_key=True AND facing_key=True AND has_key=False
   - NEVER pick up if already carrying key

2. TOGGLE RULES: 
   - Use Action 5 ONLY when: adjacent_to_door=True AND has_key=True AND door is locked
   - Don't toggle if door is already open

3. NAVIGATION RULES:
   - If need key: navigate toward key position
   - If have key: navigate toward door position
   - Don't move forward into walls

4. LEARNING-AWARE RULES:
   - Early exploration: Be educational, explain reasoning clearly
   - Guided learning: Balance correction with learning opportunities
   - Autonomous: Only intervene when clearly necessary

5. EFFICIENCY RULES:
   - Consider the learning phase when deciding to override
   - In early phases: More willing to guide and correct
   - In advanced phases: Agree with RL unless action is clearly wrong

LOOP PREVENTION:
{loop_warning}
{situation_warning}
{failure_warning}

ANALYSIS STEPS:
1. What learning phase is the agent in and how should this affect my decision?
2. What is the current objective based on agent status?
3. Is the PPO action appropriate for this objective and learning phase?
4. Does the PPO action violate any critical rules?
5. Given the learning context, should I agree or override?

RESPONSE FORMAT (MUST START WITH THIS):
Selected action: <number>

Then explain your reasoning, considering the learning phase and educational value of your decision.
"""

# Standard prompt template for fallback
STANDARD_PROMPT_TEMPLATE = """You are an expert controller for a reinforcement learning agent in MiniGrid-DoorKey-6x6.

CURRENT SITUATION:
Agent Position: {agent_pos} 
Key Position: {key_pos} | Distance: {dist_to_key} | Visible: {is_key_visible}
Door Position: {door_pos} | State: {door_state} | Distance: {dist_to_door} | Visible: {is_door_visible}
Goal Position: {goal_pos} | Distance: {dist_to_goal}

AGENT STATUS:
- Carrying key: {has_key}
- Current objective: {current_objective}

SPATIAL CONTEXT:
- Key direction: {rel_dir_to_key}
- Door direction: {rel_dir_to_door}
- Object in front: {front_object}
- Facing wall: {facing_wall}

PPO AGENT SUGGESTS: Action {action} (Confidence: {ppo_confidence})

AVAILABLE ACTIONS:
0: Turn left    1: Turn right    2: Move forward    3: Pick up key    5: Toggle door
(Actions 4 and 6 are FORBIDDEN - never use them)

CRITICAL DECISION RULES (MUST FOLLOW EXACTLY):
1. PICKUP RULES:
   - Use Action 3 ONLY when: adjacent_to_key=True AND facing_key=True AND has_key=False
   - NEVER pick up if already carrying key

2. TOGGLE RULES: 
   - Use Action 5 ONLY when: adjacent_to_door=True AND has_key=True AND door is locked
   - Don't toggle if door is already open

3. NAVIGATION RULES:
   - If need key: navigate toward key position
   - If have key: navigate toward door position
   - Don't move forward into walls

4. EFFICIENCY RULES:
   - Agree with PPO unless it clearly violates above rules
   - Only override when PPO action is demonstrably wrong or forbidden

CONTEXT NOTES:
{performance_note}

ANALYSIS STEPS:
1. What is the current objective based on agent status?
2. Is the PPO action appropriate for this objective?
3. Does the PPO action violate any critical rules?
4. Should I agree or override?

RESPONSE FORMAT (MUST START WITH THIS):
Selected action: <number>

Then explain your reasoning briefly.
"""


def render_prompt(env_name: str, features: dict, action: int) -> str:
    """
    Enhanced prompt rendering with learning awareness and safe default values.
    """

    # Check if this is a learning-aware context (has learning_context key)
    has_learning_context = 'learning_context' in features

    if has_learning_context:
        template = LEARNING_AWARE_PROMPT_TEMPLATE
    else:
        template = STANDARD_PROMPT_TEMPLATE

    # Ensure all required fields have safe default values
    safe_features = {
        'env_name': env_name,
        'action': action,
        'agent_pos': features.get('agent_pos', 'unknown'),
        'key_pos': features.get('key_pos', 'unknown'),
        'door_pos': features.get('door_pos', 'unknown'),
        'goal_pos': features.get('goal_pos', 'unknown'),
        'dist_to_key': features.get('dist_to_key', 'unknown'),
        'dist_to_door': features.get('dist_to_door', 'unknown'),
        'dist_to_goal': features.get('dist_to_goal', 'unknown'),
        'is_key_visible': features.get('is_key_visible', 'unknown'),
        'is_door_visible': features.get('is_door_visible', 'unknown'),
        'door_state': features.get('door_state', 'unknown'),
        'has_key': features.get('has_key', 'unknown'),
        'rel_dir_to_key': features.get('rel_dir_to_key', 'unknown'),
        'rel_dir_to_door': features.get('rel_dir_to_door', 'unknown'),
        'front_object': features.get('front_object', 'unknown'),
        'facing_wall': features.get('facing_wall', 'unknown'),
        'current_objective': features.get('current_objective', 'Explore and complete task'),
        'ppo_confidence': features.get('ppo_confidence', 'medium'),
        'performance_note': features.get('performance_note', 'Agent is learning, make safe decisions.'),
    }

    # Add learning-aware context with safe defaults
    if has_learning_context:
        safe_features.update({
            'learning_context': features.get('learning_context', 'Agent is in learning mode.'),
            'decision_guidance': features.get('decision_guidance', 'Make balanced decisions.'),
            'patience_note': features.get('patience_note', 'Be patient with the learning process.'),
            'learning_guidance': features.get('learning_guidance', 'Agent is developing skills.'),
            'performance_context': features.get('performance_context', ''),
            'interaction_context': features.get('interaction_context', ''),
            'intervention_style': features.get('intervention_style', 'Balance guidance with autonomy.'),
            'loop_warning': features.get('loop_warning', ''),
            'situation_warning': features.get('situation_warning', ''),
            'failure_warning': features.get('failure_warning', ''),
        })

    try:
        return template.format(**safe_features)
    except KeyError as e:
        # If there's still a missing key, log it and use standard template
        logger.warning(f"Missing key in prompt template: {e}. Using standard template.")
        return STANDARD_PROMPT_TEMPLATE.format(**{k: v for k, v in safe_features.items()
                                                  if k in ['agent_pos', 'key_pos', 'door_pos', 'goal_pos',
                                                           'dist_to_key', 'dist_to_door', 'dist_to_goal',
                                                           'is_key_visible', 'is_door_visible', 'door_state',
                                                           'has_key', 'rel_dir_to_key', 'rel_dir_to_door',
                                                           'front_object', 'facing_wall', 'current_objective',
                                                           'ppo_confidence', 'performance_note', 'action']})


def render_standard_prompt(env_name: str, features: dict, action: int) -> str:
    """
    Fallback standard prompt rendering for compatibility.
    """

    prompt_features = {
        'env_name': env_name,
        'action': action,
        'agent_pos': features.get('agent_pos', 'unknown'),
        'key_pos': features.get('key_pos', 'unknown'),
        'door_pos': features.get('door_pos', 'unknown'),
        'goal_pos': features.get('goal_pos', 'unknown'),
        'dist_to_key': features.get('dist_to_key', 'unknown'),
        'dist_to_door': features.get('dist_to_door', 'unknown'),
        'dist_to_goal': features.get('dist_to_goal', 'unknown'),
        'is_key_visible': features.get('is_key_visible', 'unknown'),
        'is_door_visible': features.get('is_door_visible', 'unknown'),
        'door_state': features.get('door_state', 'unknown'),
        'has_key': features.get('has_key', 'unknown'),
        'rel_dir_to_key': features.get('rel_dir_to_key', 'unknown'),
        'rel_dir_to_door': features.get('rel_dir_to_door', 'unknown'),
        'front_object': features.get('front_object', 'unknown'),
        'facing_wall': features.get('facing_wall', 'unknown'),
        'current_objective': features.get('current_objective', 'Explore and complete task'),
        'ppo_confidence': features.get('ppo_confidence', 'medium'),
        'performance_note': features.get('performance_note', 'Agent is learning, make safe decisions.'),
    }

    return STANDARD_PROMPT_TEMPLATE.format(**prompt_features)