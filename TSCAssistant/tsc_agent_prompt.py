# TSCAssistant/tsc_agent_prompt.py

PROMPT_TEMPLATE = """You are controlling a reinforcement learning agent in the MiniGrid-DoorKey-6x6 environment. 
This environment has only one key object that the agent must pick up in order to unlock the door and then get to the green goal square. 


Your job is to decide the **next action** the agent should take based on its current observations and the proposed action from the PPO policy.
You are the decision maker whether PPO Agent's Suggested Action is reasonable. Feel free to override if PPO Agent's Suggested Action is not reasonable.
✔️ Agree with PPO Agent's Suggested Action if the action is safe and reasonable (e.g., moving forward into an empty space, turning toward a target, or toggling a reachable door)  
❌ Override PPO Agent's Suggested Action if the action leads into a wall, repeats without progress, or violates logic (e.g., trying to pick up a key when none is adjacent to agent)

---

0: Turn left (rotate counterclockwise)
1: Turn right (rotate clockwise)  
2: Move forward (step in facing direction)
3: Pick up key (only works when adjacent to key)
4: Drop key — (disabled, do not choose, FORBIDDEN)  
5: Toggle (open/close door when adjacent)
6: Done — (disabled, do not choose, FORBIDDEN)

🧠 Environment: {env_name}

🧩 **Observable area size**: {grid_size}

🔍 **Visibility**:
- Key visible?           {is_key_visible}
- Door visible?          {is_door_visible}
- Number of objects:     {num_visible_objects}

📍 **Positions**:
- Agent at:              {agent_pos}
- Key at:                {key_pos}
- Door at:               {door_pos} (state: {door_state})
- Goal at:               {goal_pos}

📐 **Distances**:
- Manhattan distance to key:                {dist_to_key}
    ↳ Horizontal distance to key: {horizontal_distance_to_key}
    ↳ Vertical distance to key:   {vertical_distance_to_key}
- Manhattan distance to door:               {dist_to_door}
- Manhattan distance to goal:               {dist_to_goal}
    ↳ Horizontal distance to goal: {horizontal_distance_to_goal}
    ↳ Vertical distance to goal:   {vertical_distance_to_goal}
- Manhattan distance to Nearest object:        {dist_to_nearest_object}

↔️ **Relative direction**:
- To key:                {rel_dir_to_key}
- To door:               {rel_dir_to_door}
⚠️ **Facing key**:
- Facing key?            {facing_key}
- Facing wall?           {facing_wall}
- Facing door?           {facing_door}

🧭 **Agent facing direction like a compass**:      {facing_direction_compass}

🌿 **Paths open?**:       {multiple_paths_open}

🤖 PPO Agent's Suggested Action: {action}

---
📌 **Helpful Strategy Hints** (use these and your intuition when deciding actions):
- If agent is facing the key, use action 3 (pick up key), override if PPO Agent's Suggested Action different than action 3 (pick up key).
- To pickup key it is enough to face with key and distance = 1. 
- The agent can't move towards the tile with key before picking up the key.
- If agent is facing the door AND adjacent to it, use action 5 (toggle)
- If agent is facing a wall, use action 0 (turn left) or 1 (turn right) - NEVER use action 2 (move forward)

📝 Pick the best **next action** from the available actions
You can only choose ONE action you cant respond with like "action 1 or action 2".
Respond with **exactly** this format on the first line of your reply:

Selected action: <number>

For example: `Selected action: 2`

⚠️ If your response does not begin with `Selected action: <number>`, it will be ignored.

After this line, make your explanation why you override or agree with PPO Agent's Suggested Action.
"""

def render_prompt(env_name: str, features: dict, action: int) -> str:
    """
    env_name: the MiniGrid env name
    features: dict matching all placeholders above
    action: PPO’s integer suggestion
    """
    return PROMPT_TEMPLATE.format(
        env_name=env_name,
        action=action,
        **features
    )
