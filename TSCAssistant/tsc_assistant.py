import re
from TSCAssistant.tsc_agent_prompt import render_prompt
from loguru import logger
import numpy as np

class TSCAgent:
    def __init__(self, llm, verbose=True):
        self.llm = llm
        self.verbose = verbose

    def agent_run(self, sim_step, obs, action, infos):
        """
        - sim_step: current simulation step (int)
        - obs: the raw observation (numpy array or dict)
        - action: single action from PPO (int or array)
        - infos: metadata dict or list with "env" key (may be unused in MiniGrid)
        """

        # Extract env_name safely
        env_name = "MiniGrid"
        if isinstance(infos, dict):
            env_name = infos.get("env", env_name)
        elif isinstance(infos, list) and len(infos) > 0 and isinstance(infos[0], dict):
            env_name = infos[0].get("env", env_name)

        raw_obs = obs

        if raw_obs is None or not isinstance(raw_obs, np.ndarray):
            logger.warning("Missing or invalid observation. Falling back to PPO action.")
            return int(action)

        prompt = render_prompt(
            env_name=env_name,
            observation=raw_obs.tolist(),
            action=int(action)
        )

        try:
            llm_response = self.llm.invoke(prompt)
            llm_response_str = str(llm_response)
            # Extract first digit (0–4) using regex
            match = re.search(r"\baction\s*\(?\s*(\d)\s*\)?", llm_response_str, re.IGNORECASE)
            if match:
                llm_action = int(match.group(1))
                if self.verbose:
                    logger.info(f"[LLM @ Step {sim_step}] Overriding PPO with action: {llm_action}")
                return llm_action
            else:
                raise ValueError(f"No valid action found in LLM output: {llm_response_str}")
        except Exception as e:
            logger.error(f"[LLM ERROR] Falling back to PPO action: {e}")
            return int(action)
