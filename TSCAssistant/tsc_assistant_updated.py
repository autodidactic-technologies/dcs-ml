import re
import numpy as np
from TSCAssistant.tsc_agent_prompt import render_prompt
from loguru import logger

class TSCAgent:
    def __init__(self, llm, verbose=True):
        self.llm = llm
        self.verbose = verbose

    def agent_run(self, sim_step, obs, action, infos):
        """
        - sim_step: current simulation step (int)
        - obs: the raw observation (numpy array)
        - action: single action from PPO (int or array)
        - infos: metadata dict or list with "env" key (optional)

        Returns:
            final_action (int): action to be taken
            is_overridden (bool): True if LLM overrides PPO, else False
        """

        # Extract environment name
        env_name = "MiniGrid"
        if isinstance(infos, dict):
            env_name = infos.get("env", env_name)
        elif isinstance(infos, list) and len(infos) > 0 and isinstance(infos[0], dict):
            env_name = infos[0].get("env", env_name)

        # Check observation validity
        if obs is None or not isinstance(obs, np.ndarray):
            logger.warning("Missing or invalid observation. Falling back to PPO action.")
            return int(action), False

        prompt = render_prompt(
            env_name=env_name,
            observation=obs.tolist(),
            action=int(action)
        )


        try:
            llm_response = self.llm.invoke(prompt)
            llm_response_str = str(llm_response)

            # Regex to extract action from LLM response
            match = re.search(r"Selected\s*action\s*[:=]?\s*(\d)", llm_response_str, re.IGNORECASE)
            if match:
                llm_action = int(match.group(1))
                overridden = llm_action != int(action)

                if self.verbose:
                    if overridden:
                        logger.info(f"[LLM @ Step {sim_step}] Overriding PPO: {int(action)} → {llm_action}")
                    else:
                        logger.info(f"[LLM @ Step {sim_step}] Agrees with PPO: {int(action)}")

                return llm_action, overridden

            else:
                raise ValueError(f"No valid action found in LLM output: {llm_response_str}")

        except Exception as e:
            logger.error(f"[LLM ERROR] Falling back to PPO action: {e}")
            return int(action), False
