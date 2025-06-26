# tsc_agent_prompt.py

PROMPT_TEMPLATE = """You are controlling a reinforcement learning agent in the MiniGrid-DoorKey-6x6 environment. 
This environment has a key that the agent must pick up in order to unlock a door and then get to the green goal square. 
The agent must pick up a key, use it to unlock a door, and reach the green goal square.

Your job is to decide the **next action** the agent should take based on its current observation and the proposed action from the PPO policy.

---

Actions available:
0: Turn left
1: Turn right
2: Move forward
3: Pick up key (only works when key is directly under the agent and not already picked up)
4: Drop key (unused)
5: Toggle (used to open door with key)
6: Done (unused)

🧠 Environment: {env_name}
📸 Observation (7×7 grid): {observation}
🤖 PPO Agent's Suggested Action: {action}

---

📝 Instructions:

1. Your job is to select the best action for the agent using only the current observation grid and the action suggested by the PPO agent.

2. If the suggested action appears to lead to progress based on the current observation (e.g., a clear path ahead or interactable object), you may agree with it.

3. If the suggested action appears ineffective or blocked — for example, attempting to move into a wall or a closed door — choose a more appropriate action such as turning (action 0 or 1).

4. Only select or agree with action 3 (Pick up key) if a key is **visibly in the same tile as the agent**.

5. Do not select action 3 (Pick up key) if the agent has likely already picked up the key (e.g., no key is visible in the grid).

6. If a closed door is clearly visible and directly reachable in front of the agent, consider using action 5 (Toggle) to open it.

7. If it is unclear what to do (e.g., when the area looks blocked or ambiguous), prefer turning actions (0 = turn left, 1 = turn right) to explore.

Respond with **exactly** this format on the first line of your reply:

Selected action: <number>

For example: `Selected action: 2`

⚠️ If your response does not begin with `Selected action: <number>`, it will be ignored.

After this line, you may explain your choice if desired.
"""

def render_prompt(env_name, observation,action):
    return PROMPT_TEMPLATE.format(
        env_name=env_name,
        observation=str(observation),
        action=action
    )
